from typing import Any, List, Union, Tuple, Dict, Iterable

import dill
import torch
import hydra
import pathlib
import omegaconf
import numpy as np
from torch import nn
from copy import deepcopy

from diffusion_policy.common import trak_util
from diffusion_policy.dataset import episode_dataset_utils as data_util

from diffusion_policy.workspace import base_workspace
from diffusion_policy.policy import base_lowdim_policy, base_image_policy

from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.dataset.real_franka_image_dataset import RealFrankaImageDataset

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy

from diffusion_policy.classifier.lowdim_state_success_classfier import LowdimStateSuccessClassifier
from diffusion_policy.classifier.image_state_success_classfier import ImageStateSuccessClassifier
from diffusion_policy.classifier.hybrid_image_state_success_classfier import HybridImageStateSuccessClassifier


SUPPORTED_DEMO_DATASETS = (
    PushTLowdimDataset,
    PushTImageDataset,
    RobomimicReplayLowdimDataset,
    RobomimicReplayImageDataset,
    RealFrankaImageDataset,
)


SUPPORTED_POLICIES = (
    DiffusionUnetLowdimPolicy,
    DiffusionUnetImagePolicy,
    DiffusionUnetHybridImagePolicy,
)


SUPPORTED_CLASSIFIERS = (
    LowdimStateSuccessClassifier,
    ImageStateSuccessClassifier,
    HybridImageStateSuccessClassifier,
)


DemoDatasetType = Union[
    PushTLowdimDataset,
    PushTImageDataset,
    RobomimicReplayLowdimDataset,
    RobomimicReplayImageDataset,
    RealFrankaImageDataset,
]


PolicyType = Union[
    base_lowdim_policy.BaseLowdimPolicy, 
    base_image_policy.BaseImagePolicy
]


ClassifierType = Union[
    LowdimStateSuccessClassifier,
    ImageStateSuccessClassifier,
    HybridImageStateSuccessClassifier,
]


def get_index_checkpoint(checkpoints: List[pathlib.Path], idx: int) -> pathlib.Path:
    """Return checkpoint by sorted index."""
    checkpoints = sorted([ckpt for ckpt in checkpoints if "latest" not in ckpt.name])
    return checkpoints[idx]


def get_best_checkpoint(checkpoints: List[pathlib.Path]) -> pathlib.Path:
    """Return best checkpoint by mean score."""
    best_checkpoint = None
    best_score = float("-inf")
    for checkpoint in checkpoints:
        try:
            # Assuming filenames are in the form: epoch=<epoch>-test_mean_score=<score>.ckpt
            parts = checkpoint.stem.split("-")
            score_part = next((part for part in parts if "test_mean_score=" in part), None)
            if score_part is None:
                continue
            score = float(score_part.split("=")[1])

            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint

        except (ValueError, IndexError):
            continue

    if best_checkpoint is None:
        raise ValueError("No valid checkpoints with test mean scores found.")

    return best_checkpoint


def get_policy_from_checkpoint(
    checkpoint: Union[str, pathlib.Path],
    return_cfg: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> Union[nn.Module, Tuple[nn.Module, omegaconf.DictConfig]]:
    """Load policy from checkpoint."""
    payload = torch.load(open(str(checkpoint), 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    # Construct workspace.
    workspace: base_workspace.BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
    # Get policy from workspace.
    policy: PolicyType = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy: PolicyType = workspace.ema_model

    if isinstance(device, str):
        device = torch.device(device)
    policy.to(device)
    policy.eval()

    if return_cfg:
        return policy, cfg
    return policy


def get_parameter_names(model: nn.Module, keys: List[str]) -> List[str]:
    """Return parameters named specified by keys."""   
    parameter_names = list(dict(model.named_parameters()).keys())
    # Important: Remove dummy parameters to avoid None-type gradients.
    return sorted([k for k in parameter_names if any(_k in k for _k in keys) and "dummy" not in k])


def batch_dict_to_iterable(batch: Dict[str, Any]) -> Tuple[Iterable[torch.Tensor], int]:
    raise NotImplementedError("Deprecated function.")
    """Convert batch dictionary to tuple."""
    image: torch.Tensor = batch["obs"]["image"]
    agent_pos: torch.Tensor = batch["obs"]["agent_pos"]
    action: torch.Tensor = batch["action"]
    assert image.shape[0] == agent_pos.shape[0] == action.shape[0]
    return (image, agent_pos, action), action.shape[0]


def get_train_episode_lengths(
    episode_ends: np.ndarray, 
    episode_mask: np.ndarray,
    sequence_length: int, 
    pad_before: int = 0, 
    pad_after: int = 0,
    debug: bool = True,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Return lengths of all training demonstrations."""
    assert episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    episode_lengths = []
    episode_indices = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:    
            continue

        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        
        indices = []
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
        
        episode_lengths.append(len(indices))
        episode_indices.append(np.array(indices))

    episode_lengths = np.array(episode_lengths)
    
    if return_indices:
        return episode_lengths, episode_indices
    
    return episode_lengths


def get_dataset_metadata(
    cfg: omegaconf.DictConfig,
    dataset: Union[BatchEpisodeDataset, DemoDatasetType], 
) -> Dict[str, Any]:
    """Return dataset metadata, including episode lengths and indices."""
    metadata = {}
    
    if isinstance(dataset, BatchEpisodeDataset):
        metadata["success_mask"] = np.array(dataset.episode_successes, dtype=bool)
        metadata["ep_lens"] = np.array(dataset.episode_lengths)
        metadata["ep_idxs"] = deepcopy(dataset.episode_idxs)
        assert metadata["ep_idxs"][-1][-1] == len(dataset) - 1
        metadata["num_eps"] = len(metadata["ep_idxs"])
        metadata["num_samples"] = len(dataset)

    elif isinstance(dataset, SUPPORTED_DEMO_DATASETS):
        if len(dataset) > 0:
            metadata["ep_lens"] = trak_util.get_train_episode_lengths(
                episode_ends=dataset.replay_buffer.episode_ends,
                episode_mask=dataset.train_mask,
                sequence_length=cfg.task.dataset.horizon,
                pad_before=cfg.task.dataset.pad_before,
                pad_after=cfg.task.dataset.pad_after,
                debug=True,
                return_indices=False,
            )
            metadata["ep_idxs"] = data_util.ep_lens_to_idxs(metadata["ep_lens"])
            assert metadata["ep_idxs"][-1][-1] == len(dataset) - 1
            metadata["num_eps"] = len(metadata["ep_idxs"])
            metadata["num_samples"] = len(dataset)
        else:
            metadata["ep_lens"] = np.array([])
            metadata["ep_idxs"] = []
            metadata["num_eps"] = 0
            metadata["num_samples"] = 0

    else:
        raise ValueError(f"Unsupported dataset of type {type(dataset)}.")
        
    return metadata