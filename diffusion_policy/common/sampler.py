from typing import Optional, Tuple, Union

import yaml
import h5py
import numba
import pathlib
import numpy as np
from collections import defaultdict

from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
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
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


def filter_training_episodes(
    train_mask: np.ndarray,
    filter_ratio: float,
    curation_config: pathlib.Path,
    curation_method: str,
    seed: int,
) -> np.ndarray:
    """Filter training data by curation method."""
    if filter_ratio <= 0.0:
        return train_mask
    
    # Load curation config.
    with open(curation_config, "+r") as f:
        config = yaml.safe_load(f)
    
    # Filter training episodes.
    num_filter = int(train_mask.sum() * filter_ratio)
    filter_idxs = np.array(config[curation_method][seed])
    assert np.all(train_mask[filter_idxs]), "Indexing non-training data."
    train_mask[filter_idxs[:num_filter]] = False

    return train_mask


def select_holdout_episodes(
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    select_ratio: float,
    curation_config: pathlib.Path,
    curation_method: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select training data by curation method."""
    if select_ratio <= 0.0:
        return train_mask, holdout_mask

    # Load curation config.
    with open(curation_config, "+r") as f:
        config = yaml.safe_load(f)

    # Select holdout episodes.
    num_select = int(holdout_mask.sum() * select_ratio)
    select_idxs = np.array(config[curation_method][seed])
    assert np.all(holdout_mask[select_idxs]), "Indexing non-holdout data."
    assert not np.any(train_mask[select_idxs]), "Indexing training data."
    holdout_mask[select_idxs[:num_select]] = False
    train_mask[select_idxs[:num_select]] = True
    
    return train_mask, holdout_mask
    

# Note: On Line 269, we sample training, then validation, and then holdout demos. 
# Optionally for future experiments, we should swap the order in which validation 
# demos and holdout demos are sampled. That is, we want a sliding window between 
# sampled training and holdout demos, without validation demos being sampled in
# between. Doing so would result in greater coherence across the filtering and
# selection experiments because they would a) be curating from the same pool of 
# demos and b) they would share the same validation set.
def get_dataset_masks(
    dataset_path: Union[str, pathlib.Path],
    num_episodes: int,
    val_ratio: float,
    max_train_episodes: Optional[int] = None,
    train_ratio: Optional[float] = None,
    uniform_quality: bool = False,
    curate_dataset: bool = False,
    curation_config_dir: Optional[str] = None,
    curation_method: Optional[str] = None,
    filter_ratio: Optional[float] = None,
    select_ratio: Optional[float] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training, validation, and holdout masks."""
    assert not (max_train_episodes is not None and train_ratio is not None), \
    "One or neither of max_train_episodes or train_ratio should be specified."

    # Dataset splits.
    if max_train_episodes is not None:
        num_train = max_train_episodes
        num_val = int(num_episodes * val_ratio)
        num_holdout = num_episodes - num_train - num_val
    else:
        train_ratio = 1.0 - val_ratio if train_ratio is None else train_ratio
        num_train = int(num_episodes * train_ratio)
        num_val = int(num_episodes * val_ratio)
        num_holdout = num_episodes - num_train - num_val

    assert_str =f"num_train ({num_train}) + num_val ({num_val}) + num_holdout ({num_holdout}) != num_episodes ({num_episodes})"
    assert num_train + num_val + num_holdout == num_episodes, assert_str

    # Dataset info.
    dataset_path = pathlib.Path(dataset_path)
    dataset_name = dataset_path.parts[1]
    if dataset_name in ["robomimic", "hardware"]:
        task_name = dataset_path.parts[-3]
        task_type = dataset_path.parts[-2]
    elif dataset_name == "pusht":
        task_name = dataset_name
        task_type = "ph"
    elif dataset_path.parts[2] == "eval_save_episodes":
        task_name = ""
        task_type = "ph"
    elif dataset_path.parts[2] == "eval_save_episodes_real":
        task_name = ""
        task_type = "ph_real"
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}.")
    
    if task_type == "ph_real":
        # Samples in consecutive order (from evaluation).
        train_mask = np.zeros(num_episodes, dtype=bool)
        train_mask[:num_train] = True
        val_mask = np.zeros(num_episodes, dtype=bool)
        val_mask[num_train:num_train+num_val] = True
        holdout_mask = ~np.logical_or(train_mask, val_mask)

    elif task_type == "ph" or not uniform_quality:
        # i.i.d. sampling across all quality tiers.
        val_mask = get_val_mask(num_episodes, val_ratio, seed=seed)        
        train_mask = ~val_mask
        if max_train_episodes is not None:
            train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
        else:
            train_mask = downsample_mask(train_mask, num_train, seed=seed)
        holdout_mask = ~np.logical_or(train_mask, val_mask)

    elif task_type == "mh" and uniform_quality:
        # i.i.d. sampling within quality tiers.
        assert max_train_episodes is None, "Does not support max_train_episodes."
        if dataset_name == "robomimic":
            with h5py.File(dataset_path) as file:
                if any(x in task_name for x in ["lift", "can", "square"]):
                    demo_quality_sets = ["worse", "okay", "better"]
                elif "transport" in task_name:
                    demo_quality_sets = ["worse", "worse_okay", "okay", "okay_better", "better", "worse_better"]
                else:
                    raise ValueError(f"Task {task_name} is not of type {task_type}.")
                decode_fn = lambda x: np.array(
                    [int(name.decode().split("_")[-1]) for name in x]
                )
                demo_quality_idxs = [decode_fn(file["mask"][s][:]) for s in demo_quality_sets]
        
        elif dataset_name == "hardware":
            if task_name == "figure8_v3":
                demo_quality_sets = ["0", "1", "2"]
            elif task_name in ["figure8_v4", "bookshelf_v2", "bookshelf_v3"]:
                demo_quality_sets = ["0", "1", "2", "3"]
            elif any(x in task_name for x in ["figure8", "tuckbox", "bookshelf"]):
                demo_quality_sets = ["0", "1"]
            else:
                raise ValueError(f"Task {task_name} is not of type {task_type}.")
            
            quality_to_episode_idx = defaultdict(list)
            for episode_path in sorted((dataset_path.parent / "episodes").iterdir()):
                if episode_path.is_dir():
                    with open(episode_path / "quality.txt", "r") as file:
                        quality_label = file.read().strip()
                        assert quality_label in demo_quality_sets, f"Unexpected quality label: {quality_label}"
                    quality_to_episode_idx[quality_label].append(int(episode_path.stem))
            demo_quality_idxs = [np.array(quality_to_episode_idx[s]) for s in demo_quality_sets]
            
        # Dataset masks.
        train_mask = np.zeros(num_episodes, dtype=bool)
        val_mask = np.zeros(num_episodes, dtype=bool)
        holdout_mask = np.zeros(num_episodes, dtype=bool)

        # Samples per quality tier (accounts for quality sets of varying sizes).
        demo_quality_counts = np.array([len(idxs) for idxs in demo_quality_idxs], dtype=float)
        demo_quality_ratios = demo_quality_counts / demo_quality_counts.sum()
        num_samples_per_set = defaultdict(list)
        for i, quality_label in enumerate(demo_quality_sets):
            for k in [num_train, num_val, num_holdout]:
                num_samples_per_set[quality_label].append(round(k * demo_quality_ratios[i]))

        rng = np.random.default_rng(seed=seed)
        for idxs, quality_label in zip(demo_quality_idxs, demo_quality_sets):
            shuffle_idxs = idxs.copy()
            rng.shuffle(shuffle_idxs)
            start_idx = 0
            for split_mask, split_size in zip(
                [train_mask, val_mask, holdout_mask], 
                num_samples_per_set[quality_label],
            ):
                end_idx = start_idx + split_size
                split_mask[shuffle_idxs[start_idx:end_idx]] = True
                start_idx = end_idx
    else: 
        raise ValueError(f"Unsupport task type {task_type}.")

    # Assert no remainder demos.
    assert (
        train_mask.sum() == num_train and
        val_mask.sum() == num_val and
        holdout_mask.sum() == num_holdout and
        not np.logical_and(train_mask, val_mask).any() and
        not np.logical_and(train_mask, holdout_mask).any() and
        not np.logical_and(val_mask, holdout_mask).any()
    ), "Remainder demos!"

    # Dataset curation.
    if curate_dataset:
        assert (
            (curation_config_dir is not None) and
            (curation_method is not None) and
            (filter_ratio is not None and 0.0 <= filter_ratio <= 1.0) and
            (select_ratio is not None and 0.0 <= select_ratio <= 1.0)
        ), "Curation arguments must be set together"

        # Filter episodes from training data.
        train_mask = filter_training_episodes(
            train_mask=train_mask,
            filter_ratio=filter_ratio,
            curation_config=pathlib.Path(curation_config_dir) / "train_config.yaml",
            curation_method=curation_method,
            seed=seed,
        )

        # Select episodes from holdout data.
        train_mask, holdout_mask = select_holdout_episodes(
            train_mask=train_mask,
            holdout_mask=holdout_mask,
            select_ratio=select_ratio,
            curation_config=pathlib.Path(curation_config_dir) / "holdout_config.yaml",
            curation_method=curation_method,
            seed=seed,
        )

    return train_mask, val_mask, holdout_mask


class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
