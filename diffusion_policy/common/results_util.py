from typing import Dict, Optional, Tuple, List, Union

import json
import tqdm
import pathlib
import numpy as np
from numpy.lib.format import open_memmap


DEMO_RESULT_KEYS = [
    "offline_policy_loss",
    "offline_action_diversity",
    "offline_state_diversity",
    "online_state_similarity",
    "online_demo_score",
    "online_trak_influence",
]


ROLLOUT_RESULT_KEYS = [] # Dynamically set.


LOAD_POLICY_LOSS_KWARGS = {
    "default": {
        "dtype": np.float16,
        "seed": 0,
    },
    "default_diffusion": {
        "dtype": np.float16,
        "seed": 0,
        "exp_type": "diffusion",
        "num_timesteps": 64
    },
}


LOAD_ACTION_LIKELIHOOD_KWARGS = {
    "default": {
        "dtype": np.float16,
        "seed": 0,
    },
    "default_diffusion": {
        "dtype": np.float16,
        "seed": 0,
        "exp_type": "diffusion",
        "num_timesteps": 64
    },
}


CURATION_KEY_NAME_FN = lambda x: f"curation={x[0]}-filter={x[1]:.2f}-select={x[2]:.2f}"
for base_method, base_method_kwargs in zip(
    ["policy_loss", "action_likelihood"], 
    [LOAD_POLICY_LOSS_KWARGS, LOAD_ACTION_LIKELIHOOD_KWARGS]
):
    for base_key in list(base_method_kwargs.keys()):
        ROLLOUT_RESULT_KEYS.append(f"{base_method}-{base_key}")
        for curation_method in ["random", "state_similarity", "influence_mean", "influence_quality", "influence_sum", "influence_quality++"]:
            for filter_ratio, select_ratio in zip([0.25, 0.50, 0.66, 0.75, 0.00], [0.00, 0.00, 0.00, 0.00, 0.33]):
                load_key = f"{CURATION_KEY_NAME_FN((curation_method, filter_ratio, select_ratio))}-{base_key}"
                ROLLOUT_RESULT_KEYS.append(f"{base_method}-{load_key}")
                base_method_kwargs[load_key] = {
                    "curate_dataset": True,
                    "curation_method": curation_method,
                    "filter_ratio": filter_ratio,
                    "select_ratio": select_ratio,
                    **base_method_kwargs[base_key],
                }


LOAD_ACTION_VARIANCE_KWARGS = {
    "default": {
        "dtype": np.float16,
        "seed": 0,
    },
    "default_diffusion": {
        "dtype": np.float16,
        "seed": 0,
        "exp_type": "diffusion",
        "num_timesteps": 32
    },
}


LOAD_STATE_EMBEDDING_KWARGS = {
    "default_state": {
        "embedding_names": [],
        "embedding_dims": [],
        "dtype": np.float16,
        "seed": 0,
    },
    "default_image": {
        "embedding_names": ["dinov2"],
        "embedding_dims": [384],
        "dtype": np.float16,
        "seed": 0,
    },
}


LOAD_DEMO_SCORE_KWARGS = {
    "default": {
        "dtype": np.float16,
        "seed": 0,
    },
}


LOAD_TRAK_KWARGS = {
    "default": {
        "proj_dim": 4000,
        "lambda_reg": 0.0,
        "num_ckpts": 1,
        "trak_exp_name": "all_episodes",
        "dtype": np.float32,
        "seed": 0,
    },
    "default_diffusion": {
        "proj_dim": 4000,
        "lambda_reg": 0.0,
        "num_ckpts": 1,
        "trak_exp_name": "all_episodes",
        "dtype": np.float32,
        "seed": 0,
        "exp_type": "diffusion",
        "loss_fn": "square",
        "num_timesteps": 64
    },
}


def get_last_n_log_keys(
    train_dir: pathlib.Path, 
    key: str = "test/mean_score", 
    n: int = 5,
    required_epochs: Optional[int] = None,
) -> np.ndarray:
    """Return key values for the last `n` rollouts during policy training."""
    assert (train_dir / "checkpoints" / "latest.ckpt").exists(), f"Run crashed {train_dir}"

    log_values = []
    log_epochs = []
    with open(train_dir / "logs.json.txt", "r") as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                if key in log_entry:
                    log_values.append(log_entry[key])
                    log_epochs.append(log_entry["epoch"])
            except json.JSONDecodeError:
                continue
    
    if required_epochs is not None and log_epochs[-1] < required_epochs:
        raise ValueError(f"Run {train_dir} did not reach the required number of epochs: {log_epochs[-1]}/{required_epochs}")
    
    return np.array(log_values[-n:])


def get_offline_state_diversity_exp_key(
    exp_key: Optional[str] = None,
    embedding_name: Optional[str] = None,
    score_fn: Optional[str] = None,
    reverse: bool = False,
    method_prefix: bool = False,
) -> Union[str, Dict[str, str]]:
    """Construct or deconstruct exp_key for offline_state_diversity."""
    if not reverse:
        assert exp_key is None
        exp_key = f"{embedding_name}-{score_fn}"
        if method_prefix:
            exp_key = f"offline_state_diversity-{exp_key}"
        return exp_key
    else:
        assert isinstance(exp_key, str)
        start_idx = 0 if not method_prefix else 1
        embedding_name, score_fn = exp_key.split("-")[start_idx:]
        return {
            "embedding_name": embedding_name,
            "score_fn": score_fn,
        }


def get_online_state_similarity_exp_key(
    exp_key: Optional[str] = None,
    embedding_name: Optional[str] = None,
    score_fn: Optional[str] = None,
    aggr_fn: Optional[str] = None,
    metric: Optional[str] = None,
    num_rollouts: Optional[Union[int, str]] = None,
    reverse: bool = False,
    method_prefix: bool = False,
) -> Union[str, Dict[str, str]]:
    """Construct or deconstruct exp_key for online_state_similarity."""
    if not reverse:
        assert exp_key is None
        exp_key = f"{embedding_name}-{score_fn}-{aggr_fn}-{metric}-{num_rollouts}"  
        if method_prefix:
            exp_key = f"online_state_similarity-{exp_key}"
        return exp_key
    else:
        assert isinstance(exp_key, str)
        start_idx = 0 if not method_prefix else 1
        embedding_name, score_fn, aggr_fn, metric, num_rollouts = exp_key.split("-")[start_idx:]
        return {
            "embedding_name": embedding_name,
            "score_fn": score_fn,
            "aggr_fn": aggr_fn,
            "metric": metric,
            "num_rollouts": num_rollouts
        }


def get_online_trak_influence_exp_key(
    exp_key: Optional[str] = None,
    aggr_fn: Optional[str] = None,
    metric: Optional[str] = None,
    num_rollouts: Optional[Union[int, str]] = None,
    reverse: bool = False,
    method_prefix: bool = False,
) -> Union[str, Dict[str, str]]:
    """Construct or deconstruct exp_key for online_trak_influnece."""
    if not reverse:
        assert exp_key is None
        exp_key = f"{aggr_fn}-{metric}-{num_rollouts}"  
        if method_prefix:
            exp_key = f"online_trak_influence-{exp_key}"
        return exp_key
    else:
        assert isinstance(exp_key, str)
        start_idx = 0 if not method_prefix else 1
        aggr_fn, metric, num_rollouts = exp_key.split("-")[start_idx:]
        return {
            "aggr_fn": aggr_fn,
            "metric": metric,
            "num_rollouts": num_rollouts
        }
    

def check_mmap(x: np.ndarray) -> None:
    """Check if array contains nans or infs."""
    if np.any(np.isnan(x)):
        raise ValueError(f"Array contains {np.isnan(x).sum()} nans.")
    elif np.any(np.isinf(x)):
        raise ValueError(f"Array contains {np.isinf(x).sum()} infs.")


def load_mmap(
    filename: pathlib.Path,
    shape: Tuple[int, int],
    dtype: type,
    mode="r",
) -> np.ndarray:
    """Load memory-mapped array."""
    array = np.array(
        open_memmap(
            filename=filename,
            shape=shape,
            dtype=dtype,
            mode=mode,
        )
    )
    check_mmap(array)
    return array


def get_policy_losses(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str = "25.03.03",
    dtype: type = np.float16,
    seed: int = 0,
    return_dtype: type = np.float32,
    load_train: bool = True,
    load_test: bool = True,
    # Dataset curation.
    curate_dataset: bool = False,
    curation_method: Optional[str] = None,
    filter_ratio: Optional[float] = None,
    select_ratio: Optional[float] = None,
    # Task params: Diffusion.
    exp_type: Optional[str] = None,
    num_timesteps: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return numpy arrays of policy losses."""
    exp_name = f"{exp_date}_policy_loss-seed={seed}"

    if curate_dataset:
        assert (
            (curation_method is not None) and
            (filter_ratio is not None and 0.0 <= filter_ratio <= 1.0) and
            (select_ratio is not None and 0.0 <= select_ratio <= 1.0)
        ), "Curation arguments must be set together"
        exp_name = f"{exp_name}-curation={curation_method}-filter={filter_ratio:.2f}-select={select_ratio:.2f}"

    if exp_type is not None:
        if exp_type == "diffusion":
            exp_name = f"{exp_name}-num_timesteps={num_timesteps}"
        else:
            raise ValueError(f"Experiment type {exp_type} is not supported.")
    exp_dir = eval_dir / exp_name

    train_set_scores = (
        load_mmap(
            filename=exp_dir / "train_set_loss.mmap",
            shape=(train_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_train else None

    test_set_scores = (
        load_mmap(
            filename=exp_dir / "test_set_loss.mmap",
            shape=(test_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_test else None

    return train_set_scores, test_set_scores


def get_action_likelihoods(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str = "25.03.03",
    dtype: type = np.float16,
    seed: int = 0,
    return_dtype: type = np.float32,
    load_train: bool = True,
    load_test: bool = True,
    # Dataset curation.
    curate_dataset: bool = False,
    curation_method: Optional[str] = None,
    filter_ratio: Optional[float] = None,
    select_ratio: Optional[float] = None,
    # Task params: Diffusion.
    exp_type: Optional[str] = None,
    num_timesteps: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return numpy arrays of action likelihoods."""
    exp_name = f"{exp_date}_action_likelihood-seed={seed}"

    if curate_dataset:
        assert (
            (curation_method is not None) and
            (filter_ratio is not None and 0.0 <= filter_ratio <= 1.0) and
            (select_ratio is not None and 0.0 <= select_ratio <= 1.0)
        ), "Curation arguments must be set together"
        exp_name = f"{exp_name}-curation={curation_method}-filter={filter_ratio:.2f}-select={select_ratio:.2f}"

    if exp_type is not None:
        if exp_type == "diffusion":
            exp_name = f"{exp_name}-num_timesteps={num_timesteps}"
        else:
            raise ValueError(f"Experiment type {exp_type} is not supported.")
    exp_dir = eval_dir / exp_name

    train_set_scores = (
        load_mmap(
            filename=exp_dir / "train_set_likelihood.mmap",
            shape=(train_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_train else None

    test_set_scores = (
        load_mmap(
            filename=exp_dir / "test_set_likelihood.mmap",
            shape=(test_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_test else None

    return train_set_scores, test_set_scores


def get_action_variances(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str = "25.03.03",
    dtype: type = np.float16,
    seed: int = 0,
    return_dtype: type = np.float32,
    load_train: bool = True,
    load_test: bool = True,
    # Task params: Diffusion.
    exp_type: Optional[str] = None,
    num_timesteps: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return numpy arrays of action variances."""
    exp_name = f"{exp_date}_action_variance-seed={seed}"
    if exp_type is not None:
        if exp_type == "diffusion":
            exp_name = f"{exp_name}-num_timesteps={num_timesteps}"
        else:
            raise ValueError(f"Experiment type {exp_type} is not supported.")
    exp_dir = eval_dir / exp_name

    train_set_scores = (
        load_mmap(
            filename=exp_dir / "train_set_var.mmap",
            shape=(train_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_train else None

    test_set_scores = (
        load_mmap(
            filename=exp_dir / "test_set_var.mmap",
            shape=(test_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_test else None

    return train_set_scores, test_set_scores


def get_state_embeddings(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    embedding_dims: List[int],
    embedding_names: List[str],
    exp_date: str = "25.03.03",
    dtype: type = np.float16,
    seed: int = 0,
    return_dtype: type = np.float32,
    load_train: bool = True,
    load_test: bool = True,
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Return numpy arrays of state embeddings."""
    exp_name = f"{exp_date}_embeddings-seed={seed}"
    exp_dir = eval_dir / exp_name

    state_embeddings = {}
    for embedding_dim, embedding_name in zip(embedding_dims, embedding_names):
        try: 
            state_embeddings[embedding_name] = (
                load_mmap(
                    filename=exp_dir / f"train_set_{embedding_name}_emb.mmap",
                    shape=(train_set_size, embedding_dim),
                    dtype=dtype,
                ).squeeze().astype(return_dtype) if load_train else None,
                load_mmap(
                    filename=exp_dir / f"test_set_{embedding_name}_emb.mmap",
                    shape=(test_set_size, embedding_dim),
                    dtype=dtype,
                ).squeeze().astype(return_dtype) if load_test else None
            )
        except FileNotFoundError:
            print(f"Did not find {embedding_name} embeddings in {eval_dir}.")
    return state_embeddings


def get_demo_score_scores(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str = "25.03.03",
    dtype: type = np.float16,
    seed: int = 0,
    return_dtype: type = np.float32,
    load_train: bool = True,
    load_test: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return numpy arrays of DemoScore scores."""
    exp_name = f"{exp_date}_demo_score-seed={seed}"
    exp_dir = eval_dir / exp_name

    train_set_scores = (
        load_mmap(
            filename=exp_dir / "train_set_demo_scores.mmap",
            shape=(train_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_train else None

    test_set_scores = (
        load_mmap(
            filename=exp_dir / "test_set_demo_scores.mmap",
            shape=(test_set_size, 1),
            dtype=dtype,
        )
        .squeeze()
        .astype(return_dtype)
    ) if load_test else None

    return train_set_scores, test_set_scores


def get_trak_scores(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str,
    proj_dim: int = 4000,
    lambda_reg: float = 0.0,
    num_ckpts: int = 1,
    trak_exp_name: str = "all_episodes",
    dtype: type = np.float32,
    return_dtype: type = np.float32,
    seed: int = 0,
    debug: bool = True,
    # Task params: Diffusion.
    exp_type: Optional[str] = None,
    loss_fn: Optional[str] = None,
    num_timesteps: Optional[int] = None,
) -> np.ndarray:
    """Return numpy array of TRAK scores."""
    exp_name = f"{exp_date}_trak_results-proj_dim={proj_dim}-lambda_reg={lambda_reg}-num_ckpts={num_ckpts}-seed={seed}"
    if exp_type is not None:
        if exp_type == "diffusion":
            exp_name = f"{exp_name}-loss_fn={loss_fn}-num_timesteps={num_timesteps}"
        else:
            raise ValueError(f"Experiment type {exp_type} is not supported.")
    exp_dir = eval_dir / exp_name

    if debug:
        # Store directories.
        print("Checking if memory maps contain nans or infs.")
        for i in tqdm.tqdm(range(num_ckpts)):
            trak_seed_dir = exp_dir / str(i)
            load_mmap(
                filename=trak_seed_dir / "grads.mmap",
                shape=(train_set_size, proj_dim),
                dtype=dtype,
            )
            load_mmap(
                filename=trak_seed_dir / "out_to_loss.mmap",
                shape=(train_set_size, 1),
                dtype=dtype,
            )
            load_mmap(
                filename=trak_seed_dir / "features.mmap",
                shape=(train_set_size, proj_dim),
                dtype=dtype,
            )
            load_mmap(
                filename=trak_seed_dir / "_is_featurized.mmap",
                shape=(train_set_size, 1),
                dtype=np.int32,
            )

    scores_path = exp_dir / "scores" / f"{trak_exp_name}.mmap"
    scores = load_mmap(
        filename=scores_path,
        shape=(train_set_size, test_set_size),
        dtype=dtype,
    )

    return scores.T.astype(return_dtype)