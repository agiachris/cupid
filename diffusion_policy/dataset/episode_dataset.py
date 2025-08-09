from typing import Any, Dict, Union, List, Generator, Optional, Tuple

import os
import glob
import yaml
import pickle
import pathlib
import collections
import numpy as np
import pandas as pd

import torch
from torch.utils.data import IterableDataset, default_collate

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.episode_dataset_utils import ep_lens_to_idxs


def load_pickle(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_yaml(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


class EpisodeDataset(IterableDataset):

    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path],
        exec_horizon: int,
        sample_history: int,
        filter_success: bool = False,
        filter_failure: bool = False,
        filter_episodes: Optional[List[int]] = None,
        max_episode_length: Optional[int] = None,
        max_num_episodes: Optional[int] = None,
        episode_cache_size: int = 1,
    ):
        """Construct EpisodeDataset."""
        super().__init__()
        assert exec_horizon >= 1 and sample_history >= 0
        self._dataset_path = dataset_path        
        
        # Episode files.
        episode_files = sorted(glob.glob(os.path.join(dataset_path, "*")))
        self._episode_files = [file for file in episode_files if os.path.basename(file) != "metadata.yaml"]
        self._metadata = load_yaml(os.path.join(dataset_path, "metadata.yaml"))
        self._episode_idxs = ep_lens_to_idxs(np.array(self.episode_lengths))
        self._episode_cache = collections.deque()
        self._episode_cache_size = episode_cache_size

        # Sampling parameters.
        self._exec_horizon = exec_horizon
        self._sample_history = sample_history
        self._filter_success = filter_success
        self._filter_failure = filter_failure
        self._filter_episodes = filter_episodes
        self._max_episode_length = max_episode_length
        self._max_num_episodes = max_num_episodes
    
    def __len__(self):
        return self.len
    
    @property
    def len(self) -> int:
        """Length of dataset."""
        return self._metadata["length"]

    @property
    def episode_lengths(self) -> List[int]:
        """Length of episodes."""
        return self._metadata["episode_lengths"]
    
    @property
    def episode_idxs(self) -> List[np.ndarray]:
        """Dataset indices associated with each episode."""
        return self._episode_idxs

    @property
    def episode_successes(self) -> List[int]:
        """Episode successes."""
        return self._metadata["episode_successes"]
    
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
    
    def __getitem__(self, idx: int) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Return sample."""
        # Find corresponding episode.
        episode_mask = np.array([idx in x for x in self.episode_idxs])
        if episode_mask.sum() == 0:
            raise ValueError(f"Sample index {idx} not found in dataset.")
        if episode_mask.sum() > 1:
            raise ValueError(f"Multiple instances of sample index {idx} found in dataset.")

        # Search cache for episode.
        episode = int(episode_mask.argmax())
        episode_frame = self._search_cache(episode)

        # Optionally episode frame to cache.
        if episode_frame is None:
            episode_frame = load_pickle(self._episode_files[episode])
            self._cache_episode(episode, episode_frame)

        # Retrieve sample within episode.
        sample_idx: int = int(np.where(self.episode_idxs[episode] == idx)[0][0])
        sample = [
            episode_frame.iloc[j].to_dict()
            for j in range(
                sample_idx - self._exec_horizon * self._sample_history,
                sample_idx + 1,
                self._exec_horizon,
            )
        ]
        return sample[0] if len(sample) == 1 else sample

    def __iter__(
        self,
    ) -> Generator[Union[Dict[str, Any], List[Dict[str, Any]]], None, None]:
        """Return sample."""
        num_episodes = 0
        for i, file_path in enumerate(self._episode_files):
            # if self._max_num_episodes is not None and num_episodes >= self._max_num_episodes:
            if self._max_num_episodes is not None and i >= self._max_num_episodes:
                continue

            episode = load_pickle(file_path)
            success = episode.iloc[0].to_dict().get("success", True)
            if (
                (self._filter_success and success)
                or (self._filter_failure and not success)
                or (
                    self._filter_episodes is not None
                    and not isinstance(self._filter_episodes, str)
                    and i in self._filter_episodes
                )
            ):
                continue
            else:
                num_episodes += 1

            for idx in range(
                self._exec_horizon * self._sample_history,
                len(episode),
                self._exec_horizon,
            ):
                if (
                    self._max_episode_length is not None
                    and episode.iloc[idx].to_dict()["timestep"]
                    >= self._max_episode_length
                ):
                    continue

                sample = [
                    episode.iloc[j].to_dict()
                    for j in range(
                        idx - self._exec_horizon * self._sample_history,
                        idx + 1,
                        self._exec_horizon,
                    )
                ]
                assert all(x["episode"] == i for x in sample)
                yield sample[0] if len(sample) == 1 else sample


class BatchEpisodeDataset(EpisodeDataset):
    
    def __init__(self, batch_size: int, collate_non_batch: bool = True, *args: Any, **kwargs: Any):
        """Construct BatchableEpisodeDataset."""
        super().__init__(*args, **kwargs)
        self._batch_size = batch_size
        self._collate_non_batch = collate_non_batch

    def __iter__(self) -> Generator[Union[Dict[str, Any], List[Dict[str, Any]]], None, None]:
        """Return batched sample."""
        if self._batch_size == 1 and not self._collate_non_batch:
            for sample in super().__iter__():
                assert isinstance(sample, dict)
                yield dict_apply(sample, torch.from_numpy)
                
        else:
            batch = []
            for sample in super().__iter__():
                batch.append(sample)

                if len(batch) == self._batch_size:
                    yield default_collate(batch)
                    batch = []

            if batch:
                yield default_collate(batch)
