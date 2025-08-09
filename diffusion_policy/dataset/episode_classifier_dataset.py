from typing import Any, Dict, Union, List, Optional, Tuple

import os
import copy
import glob
import yaml
import pickle
import pathlib
import collections
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.episode_dataset_utils import ep_lens_to_idxs
from diffusion_policy.common.sampler import get_dataset_masks


def load_pickle(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_yaml(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


class EpisodeDatasetBase(Dataset):

    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path],
        episode_cache_size: Optional[int] = None,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        seed: int = 42,
        preload_episode_cache: bool = True,
        dataset_mask_kwargs: Dict[str, Any] = {}
    ):
        """Construct EpisodeDataset."""
        super().__init__()
        self._dataset_path = pathlib.Path(dataset_path)
        self._init_episode_cache_size = episode_cache_size
        self._preload_episode_cache = preload_episode_cache

        # Episode files.
        _, episode_files = self.get_dataset_files(dataset_path)
        
        # Get train, validation, holdout masks.
        train_mask, val_mask, holdout_mask = get_dataset_masks(
            dataset_path=dataset_path,
            num_episodes=len(episode_files),
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            seed=seed,
            **dataset_mask_kwargs,
        )
        self.setup_dataset(train_mask)

        # General parameters.
        self._train_mask = train_mask
        self._val_mask = val_mask
        self._holdout_mask = holdout_mask
    
    @staticmethod
    def get_dataset_files(dataset_path: Union[str, pathlib.Path]) -> Tuple[pathlib.Path, List[pathlib.Path]]:
        """Return dataset metadata and episode file paths."""
        dataset_path = pathlib.Path(dataset_path)
        metadata_path = dataset_path / "metadata.yaml"
        episode_files = sorted(glob.glob(os.path.join(dataset_path, "*")))
        episode_files = [pathlib.Path(file) for file in episode_files if os.path.basename(file) != "metadata.yaml"]
        return metadata_path, episode_files

    def setup_dataset(self, train_mask: np.ndarray) -> None:
        """Setup dataset for sampling."""
        # Load episode files and metadata.
        metadata_path, episode_files = self.get_dataset_files(self._dataset_path)
        metadata = load_yaml(metadata_path)
        
        # Setup training set metadata.
        self._episode_files = [x for i, x in enumerate(episode_files) if train_mask[i]]
        self._episode_lengths = [x for i, x in enumerate(metadata["episode_lengths"]) if train_mask[i]]
        self._episode_successes = [x for i, x in enumerate(metadata["episode_successes"]) if train_mask[i]]
        self._episode_indices = ep_lens_to_idxs(np.array(self._episode_lengths))
        assert self._episode_indices[-1][-1] == sum(self._episode_lengths) - 1
        self._length = sum(self._episode_lengths)
        
        # Setup episode cache.
        if self._init_episode_cache_size is None:
            self._episode_cache_size = len(self._episode_files)
        else:
            self._episode_cache_size = self._init_episode_cache_size
        self._episode_cache = collections.deque()

        # Optionally preload episode cache.
        if self._preload_episode_cache:
            for episode, episode_file in enumerate(self._episode_files):
                assert self._search_cache(episode) is None, "Episode doubly cached."
                episode_frame = load_pickle(episode_file)
                self._cache_episode(episode, episode_frame)
            
        # Accelerate __getitem__.
        idx_to_episode: Dict[int, int] = {}
        idx_to_episode_idx: Dict[int, int] = {}
        for episode, episode_indices in enumerate(self._episode_indices):
            for episode_idx, idx in enumerate(episode_indices):
                assert idx not in idx_to_episode, "Indices must be unique."
                idx_to_episode[int(idx)] = int(episode)
                idx_to_episode_idx[int(idx)] = int(episode_idx)
        self._idx_to_episode = idx_to_episode
        self._idx_to_episode_idx = idx_to_episode_idx
    
    def get_validation_dataset(self) -> "EpisodeDatasetBase":
        """Return validation dataset."""
        val_set = copy.copy(self)
        val_set._train_mask = self._val_mask
        val_set.setup_dataset(self._val_mask)
        return val_set
    
    def get_holdout_dataset(self) -> "EpisodeDatasetBase":
        """Return validation dataset."""
        holdout_set = copy.copy(self)
        holdout_set._train_mask = self._holdout_mask
        holdout_set.setup_dataset(self._holdout_mask)
        return holdout_set

    def __len__(self):
        return self.len
    
    @property
    def len(self) -> int:
        """Length of dataset."""
        return self._length

    @property
    def episode_lengths(self) -> List[int]:
        """Length of episodes."""
        return self._episode_lengths
    
    @property
    def episode_idxs(self) -> List[np.ndarray]:
        """Dataset indices associated with each episode."""
        return self._episode_indices

    @property
    def episode_successes(self) -> List[int]:
        """Episode successes."""
        return self._episode_successes
    
    def _search_cache(self, episode: int) -> Optional[pd.DataFrame]:
        """Search cache for episode pd.DataFrame."""
        for _episode, _episode_frame in self._episode_cache:
            if episode == _episode:
                return _episode_frame
        return None

    def _cache_episode(self, episode: int, episode_frame: pd.DataFrame) -> None:
        """Add episode pd.DataFrame to the dataset cache."""
        self._episode_cache.appendleft((episode, episode_frame))
        if len(self._episode_cache) > self._episode_cache_size:
            self._episode_cache.pop()
        assert len(self._episode_cache) <= self._episode_cache_size

    def _preprocess_sample(self, idx: int, episode: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return sample processed for downstream task."""
        raise NotImplementedError("Implement in subclass.")
    
    def __getitem__(self, idx: int) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Return sample."""        
        # Search cache for episode.
        episode = self._idx_to_episode[idx]
        episode_frame = self._search_cache(episode)

        # Optionally add episode frame to cache.
        if episode_frame is None and not self._preload_episode_cache:
            episode_frame = load_pickle(self._episode_files[episode])
            self._cache_episode(episode, episode_frame)
        elif episode_frame is None and self._preload_episode_cache:
            raise ValueError(f"Episode cache preloading failed. Could not find episode {episode}")

        # Retrieve sample within episode.
        sample_idx = self._idx_to_episode_idx[idx]
        sample = episode_frame.iloc[sample_idx].to_dict()

        # Process sample for task.
        sample = self._preprocess_sample(idx, episode, sample)
        
        return dict_apply(sample, torch.from_numpy)


class EpisodeClassifierDataset(EpisodeDatasetBase):

    def __init__(self, *args: Any, **kwargs: Any):
        """Construct EpisodeClassifierDataset."""
        super().__init__(*args, **kwargs)
    
    def _preprocess_sample(self, idx: int, episode: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return sample processed for classification task."""
        return {
            "obs": sample["obs"], 
            "action": sample["action"], 
            "success": np.array(self._episode_successes[episode], dtype=np.int64)[None]
        }
