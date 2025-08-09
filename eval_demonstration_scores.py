from typing import Union, Dict, Any, Callable, List, Optional

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import click
import hydra
import pickle
import random
import pathlib
import numpy as np
from copy import deepcopy
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset

from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

from diffusion_policy.common import results_util
from diffusion_policy.common import trak_util
from diffusion_policy.common import error_util


TEST_EMBEDDING_SCORE_FNS: Dict[str, Dict[str, Any]] = {
    "mahal": {"method": "mahal"},
    "top1_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 1}},
    "top5_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 5}},
    "top10_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 10}},
    "top1_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 1}},
    "top5_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 5}},
    "top10_cosine": {"method": "topk","method_kwargs": {"error_fn": "cosine", "k": 10}},
}


PAIRWISE_EMBEDDING_SCORE_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "l2": error_util.compute_l2_distance,
    "mahal": error_util.compute_mahal_distance,
    "cosine": error_util.compute_cosine_error,
}


def get_rollout_idx(
    metric: str, 
    num_rollouts: Union[str, int],
    success_mask: np.ndarray,
    random: bool = False,
) -> Optional[np.ndarray]:
    """Return rollout indices."""
    valid_idx = np.arange(len(success_mask))

    if isinstance(num_rollouts, str):
        assert num_rollouts == "all"
        return valid_idx

    if metric == "succ":
        valid_idx = valid_idx[success_mask]
    
    elif metric == "fail":
        valid_idx = valid_idx[~success_mask]
    
    if num_rollouts > len(valid_idx):
        return None
    
    if not random:
        return valid_idx[:num_rollouts]
    
    return np.random.choice(valid_idx, num_rollouts, replace=False)


def get_test_embedding_score_fns(embedding_name: str) -> List[str]:
    """Return test embedding score functions."""
    if embedding_name == "dinov2":
        return ["top5_cosine", "top10_cosine"]
    elif embedding_name == "policy":
        return ["mahal", "top5_l2", "top10_l2"]
    else:
        raise ValueError(f"Unsupported embedding {embedding_name}.")
    

def get_pairwise_embedding_score_fns(embedding_name: str) -> List[str]:
    """Return pairwise embedding score functions."""
    if embedding_name == "dinov2":
        return ["cosine"]
    elif embedding_name == "policy":
        return ["l2"]
    else:
        raise ValueError(f"Unsupported embedding {embedding_name}.")


def offline_policy_loss_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute demonstration scores by policy loss."""
    # Load policy losses.
    train_holdout_loss, _ = results_util.get_policy_losses(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **results_util.LOAD_POLICY_LOSS_KWARGS[load_key],
    )
    assert isinstance(train_holdout_loss, np.ndarray)

    # Compute demo scores.
    result = {
        "train": error_util.sample_to_trajectory_scores(
            sample_scores=train_holdout_loss[:train_set_metadata["num_samples"]],
            num_eps=train_set_metadata["num_eps"],
            ep_idxs=train_set_metadata["ep_idxs"],
            ep_lens=train_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        result["holdout"] = error_util.sample_to_trajectory_scores(
            sample_scores=train_holdout_loss[train_set_metadata["num_samples"]:],
            num_eps=holdout_set_metadata["num_eps"],
            ep_idxs=holdout_set_metadata["ep_idxs"],
            ep_lens=holdout_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )

    assert not any(np.any(v == 0.0) for v in result.values()), "Result contains a zero score"
    
    print(f"Completed: Offline Policy Loss.")
    return result


def offline_action_diversity_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute demonstration scores by action variance."""
    # Load action variances.
    train_holdout_var, _ = results_util.get_action_variances(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **results_util.LOAD_ACTION_VARIANCE_KWARGS[load_key],
    )
    assert isinstance(train_holdout_var, np.ndarray)

    # Compute demo scores.
    result = {
        "train": error_util.sample_to_trajectory_scores(
            sample_scores=train_holdout_var[:train_set_metadata["num_samples"]],
            num_eps=train_set_metadata["num_eps"],
            ep_idxs=train_set_metadata["ep_idxs"],
            ep_lens=train_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        result["holdout"] = error_util.sample_to_trajectory_scores(
            sample_scores=train_holdout_var[train_set_metadata["num_samples"]:],
            num_eps=holdout_set_metadata["num_eps"],
            ep_idxs=holdout_set_metadata["ep_idxs"],
            ep_lens=holdout_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )

    assert not any(np.any(v == 0.0) for v in result.values()), "Result contains a zero score"
    
    print(f"Completed: Offline Action Diversity.")
    return result


def offline_state_diversity_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
    policy_embedding_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute demonstration scores by state diversity."""
    # Online update load kwargs.
    load_kwargs = deepcopy(results_util.LOAD_STATE_EMBEDDING_KWARGS[load_key])
    load_kwargs["embedding_names"].append("policy")
    load_kwargs["embedding_dims"].append(policy_embedding_dim)

    # Load state embeddings.
    state_embeddings = results_util.get_state_embeddings(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **load_kwargs,
    )
    embedding_names = list(state_embeddings.keys())

    # Compute demo scores.
    result = defaultdict(dict)

    # Compute train demo state diversity in leave-episode-out fashion.
    for embedding_name in embedding_names:
        train_holdout_embeddings = state_embeddings[embedding_name][0]
        assert isinstance(train_holdout_embeddings, np.ndarray)
        train_embeddings = train_holdout_embeddings[:train_set_metadata["num_samples"]]

        for score_fn in get_test_embedding_score_fns(embedding_name):
            mask = np.zeros(train_set_metadata["num_samples"], dtype=bool)
            sample_scores = np.zeros(train_set_metadata["num_samples"], dtype=return_dtype)

            for ep_idx, ep_len in zip(train_set_metadata["ep_idxs"], train_set_metadata["ep_lens"]):
                # Extract episode and non-episode embeddings.
                mask[ep_idx] = True
                ep_embeddings = train_embeddings[mask]
                non_ep_embeddings = train_embeddings[~mask]
                assert len(ep_embeddings) == ep_len
                assert len(non_ep_embeddings) == train_set_metadata["num_samples"] - ep_len

                # Compute state diversity of episode states w.r.t. non-episode states.
                sample_scores[ep_idx] = error_util.compute_test_embedding_scores(
                    data_embeddings=non_ep_embeddings,
                    test_embeddings=ep_embeddings,
                    **TEST_EMBEDDING_SCORE_FNS[score_fn]
                )

                # Reset episode mask.
                mask[:] = False

            assert not np.any(sample_scores == 0.0), "Result contains a zero score"

            exp_key = results_util.get_offline_state_diversity_exp_key(
                embedding_name=embedding_name,
                score_fn=score_fn,
            )
            result["train"][exp_key] = error_util.sample_to_trajectory_scores(
                sample_scores=sample_scores,
                num_eps=train_set_metadata["num_eps"],
                ep_idxs=train_set_metadata["ep_idxs"],
                ep_lens=train_set_metadata["ep_lens"],
                aggr_fn=np.mean,
                return_dtype=return_dtype,
            )
            print(f"Completed: Offline State Diversity [Train] [{exp_key}].")

    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        # Compute holdout demo state diversity with respect to training set.
        for embedding_name in embedding_names:
            train_holdout_embeddings = state_embeddings[embedding_name][0]
            assert isinstance(train_holdout_embeddings, np.ndarray)
            train_embeddings = train_holdout_embeddings[:train_set_metadata["num_samples"]]
            holdout_embeddings = train_holdout_embeddings[train_set_metadata["num_samples"]:]

            for score_fn in get_test_embedding_score_fns(embedding_name):
                sample_scores = error_util.compute_test_embedding_scores(
                    data_embeddings=train_embeddings,
                    test_embeddings=holdout_embeddings,
                    **TEST_EMBEDDING_SCORE_FNS[score_fn]
                )
                assert sample_scores.ndim == 1 and len(sample_scores) == holdout_set_metadata["num_samples"]
                assert not np.any(sample_scores == 0.0), "Result contains a zero score"

                exp_key = results_util.get_offline_state_diversity_exp_key(
                    embedding_name=embedding_name,
                    score_fn=score_fn,
                )
                result["holdout"][exp_key] = error_util.sample_to_trajectory_scores(
                    sample_scores=sample_scores,
                    num_eps=holdout_set_metadata["num_eps"],
                    ep_idxs=holdout_set_metadata["ep_idxs"],
                    ep_lens=holdout_set_metadata["ep_lens"],
                    aggr_fn=np.mean,
                    return_dtype=return_dtype,
                )
                print(f"Completed: Offline State Diversity [Holdout] [{exp_key}].")

    return result


def online_state_similarity_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
    policy_embedding_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute demonstration scores by state similarity (resp., disimilarity) 
    w.r.t. successful on-policy rollouts (resp., failed on-policy rollouts)."""
    # Online update load kwargs.
    load_kwargs = deepcopy(results_util.LOAD_STATE_EMBEDDING_KWARGS[load_key])
    load_kwargs["embedding_names"].append("policy")
    load_kwargs["embedding_dims"].append(policy_embedding_dim)

    # Load state embeddings.
    state_embeddings = results_util.get_state_embeddings(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=True,
        **load_kwargs,
    )
    embedding_names = list(state_embeddings.keys())

    # Requires success mask.
    metrics = ["succ", "fail", "net"]
    aggr_fns = {
        "mean_of_mean": error_util.mean_of_mean_influence,
        "mean_of_mean_success": error_util.mean_of_mean_influence_success,
        "sum_of_sum": error_util.sum_of_sum_influence,
        "sum_of_sum_success": error_util.sum_of_sum_influence_success,
    }
    test_rollout_fractions = {
        "succ": ["all"],
        "fail": ["all"],
        "net": [0.01, 0.05, 0.10, 0.25, 0.5, 1.00, "all"],
    }
    
    # Compute demo scores.
    result = defaultdict(dict)

    def online_state_similarity_routine(split: str) -> None:
        """Compute trajectory-level state similarity scores between demos and rollouts."""
        assert split in ["train", "holdout"], f"Split {split} must be either train or holdout."
        demo_set_metadata = train_set_metadata if split == "train" else holdout_set_metadata

        for embedding_name in embedding_names:
            demo_embeddings = state_embeddings[embedding_name][0]
            test_embeddings = state_embeddings[embedding_name][1]
            assert isinstance(test_embeddings, np.ndarray)
            assert isinstance(demo_embeddings, np.ndarray)

            test_embeddings = test_embeddings[:test_set_metadata["num_samples"]]
            if split == "train":
                demo_embeddings = demo_embeddings[:train_set_metadata["num_samples"]]
            elif split == "holdout":
                demo_embeddings = demo_embeddings[train_set_metadata["num_samples"]:]
            else: 
                raise ValueError(f"Split {split} must be either train or holdout.")
            
            for score_fn in get_pairwise_embedding_score_fns(embedding_name):
                test_demo_scores = -1 * PAIRWISE_EMBEDDING_SCORE_FNS[score_fn](test_embeddings, demo_embeddings)
                assert test_demo_scores.shape == (test_set_metadata["num_samples"], demo_set_metadata["num_samples"])
                
                for aggr_fn in aggr_fns.keys():
                    test_demo_traj_scores = error_util.pairwise_sample_to_trajectory_scores(
                        pairwise_sample_scores=test_demo_scores,
                        num_test_eps=test_set_metadata["num_eps"],
                        num_train_eps=demo_set_metadata["num_eps"],
                        test_ep_idxs=test_set_metadata["ep_idxs"],
                        train_ep_idxs=demo_set_metadata["ep_idxs"],
                        test_ep_lens=test_set_metadata["ep_lens"],
                        train_ep_lens=demo_set_metadata["ep_lens"],
                        success_mask=test_set_metadata["success_mask"],
                        aggr_fn=aggr_fns[aggr_fn],
                        return_dtype=return_dtype,
                    )
                    assert test_demo_traj_scores.shape == (test_set_metadata["num_eps"], demo_set_metadata["num_eps"])                  

                    for metric in metrics:
                        for fraction in test_rollout_fractions[metric]:
                            if isinstance(fraction, str):
                                num_rollouts = fraction
                            else:
                                num_rollouts = int(test_set_metadata["num_eps"] * fraction)

                            exp_key = results_util.get_online_state_similarity_exp_key(
                                embedding_name=embedding_name,
                                score_fn=score_fn,
                                aggr_fn=aggr_fn,
                                metric=metric,
                                num_rollouts=num_rollouts,
                            )

                            rollout_idx = get_rollout_idx(
                                metric=metric,
                                num_rollouts=num_rollouts,
                                success_mask=test_set_metadata["success_mask"],
                                random=False
                            )
                            
                            if rollout_idx is None:
                                raise ValueError("Not enough successful or failed rollouts.")

                            result[split][exp_key] = error_util.compute_demo_quality_scores(
                                traj_scores=test_demo_traj_scores[rollout_idx, :],
                                success_mask=test_set_metadata["success_mask"][rollout_idx],
                                metric=metric,
                            )
                            print(f"Completed: Online State Similarity [{split.title()}] [{exp_key}].")

    # Compute train demo state similarity scores.
    online_state_similarity_routine("train")

    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        # Compute holdout demo state similarity scores.
        online_state_similarity_routine("holdout")

    return result


def online_demo_score_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute demonstration scores by the average (binary classifier) 
    predicted success rate across all states in the demonstration."""
    # Load DemoScore scores.
    train_holdout_scores, _ = results_util.get_demo_score_scores(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **results_util.LOAD_DEMO_SCORE_KWARGS[load_key],
    )
    assert isinstance(train_holdout_scores, np.ndarray)

    # Compute demo scores.
    result = {
        "train": error_util.sample_to_trajectory_scores(
            sample_scores=train_holdout_scores[:train_set_metadata["num_samples"]],
            num_eps=train_set_metadata["num_eps"],
            ep_idxs=train_set_metadata["ep_idxs"],
            ep_lens=train_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        result["holdout"] = error_util.sample_to_trajectory_scores(
            sample_scores=train_holdout_scores[train_set_metadata["num_samples"]:],
            num_eps=holdout_set_metadata["num_eps"],
            ep_idxs=holdout_set_metadata["ep_idxs"],
            ep_lens=holdout_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )

    if any(np.any(v == 0.0) for v in result.values()):
        print("Result contains a zero score")
    
    print(f"Completed: Online DemoScore.")
    return result


def online_trak_influence_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute demonstration scores by positive influence (resp., negative influence) 
    w.r.t. successful on-policy rollouts (resp., failed on-policy rollouts)."""
    # Load TRAK scores.
    train_holdout_scores = results_util.get_trak_scores(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        **results_util.LOAD_TRAK_KWARGS[load_key]
    )

    # Requires success mask.
    metrics = ["succ", "fail", "net"]
    aggr_fns = {
        "mean_of_mean": error_util.mean_of_mean_influence,
        "mean_of_mean_success": error_util.mean_of_mean_influence_success,
        "sum_of_sum": error_util.sum_of_sum_influence,
        "sum_of_sum_success": error_util.sum_of_sum_influence_success,
        "min_of_max": error_util.min_of_max_influence,
        "max_of_min": error_util.max_of_min_influence,
    }
    test_rollout_fractions = {
        "succ": ["all"],
        "fail": ["all"],
        "net": [0.01, 0.05, 0.10, 0.25, 0.5, 1.00, "all"],
    }

    # Compute demo scores.
    result = defaultdict(dict)

    def online_trak_influence_routine(split: str) -> None:
        """Compute trajectory-level influence scores between demos and rollouts."""
        assert split in ["train", "holdout"], f"Split {split} must be either train or holdout."
        demo_set_metadata = train_set_metadata if split == "train" else holdout_set_metadata
        if split == "train":
            test_demo_scores = train_holdout_scores[:, :train_set_metadata["num_samples"]]
        elif split == "holdout":
            test_demo_scores = train_holdout_scores[:, train_set_metadata["num_samples"]:]
        else: 
            raise ValueError(f"Split {split} must be either train or holdout.")

        for aggr_fn in aggr_fns.keys():
            test_demo_traj_scores = error_util.pairwise_sample_to_trajectory_scores(
                pairwise_sample_scores=test_demo_scores,
                num_test_eps=test_set_metadata["num_eps"],
                num_train_eps=demo_set_metadata["num_eps"],
                test_ep_idxs=test_set_metadata["ep_idxs"],
                train_ep_idxs=demo_set_metadata["ep_idxs"],
                test_ep_lens=test_set_metadata["ep_lens"],
                train_ep_lens=demo_set_metadata["ep_lens"],
                success_mask=test_set_metadata["success_mask"],
                aggr_fn=aggr_fns[aggr_fn],
                return_dtype=return_dtype,
            )
            assert test_demo_traj_scores.shape == (test_set_metadata["num_eps"], demo_set_metadata["num_eps"])

            for metric in metrics:
                for fraction in test_rollout_fractions[metric]:
                    if isinstance(fraction, str):
                        num_rollouts = fraction
                    else:
                        num_rollouts = int(test_set_metadata["num_eps"] * fraction)

                    exp_key = results_util.get_online_trak_influence_exp_key(
                        aggr_fn=aggr_fn,
                        metric=metric,
                        num_rollouts=num_rollouts,
                    )

                    rollout_idx = get_rollout_idx(
                        metric=metric,
                        num_rollouts=num_rollouts,
                        success_mask=test_set_metadata["success_mask"],
                        random=False
                    )
                        
                    if rollout_idx is None:
                        raise ValueError("Not enough successful or failed rollouts.")
                    
                    result[split][exp_key] = error_util.compute_demo_quality_scores(
                        traj_scores=test_demo_traj_scores[rollout_idx, :],
                        success_mask=test_set_metadata["success_mask"][rollout_idx],
                        metric=metric,
                    )
                    print(f"Completed: Online TRAK Influence [{split.title()}] [{exp_key}].")

    # Compute train demo trak influence scores.
    online_trak_influence_routine("train")

    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        # Compute holdout demo trak influence scores.
        online_trak_influence_routine("holdout")

    return result


@click.command()
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, required=True)
@click.option("--result_date", type=str, required=True)
@click.option('--overwrite', type=bool, default=False)
@click.option("--device", type=str, default="cpu")
@click.option("--seed", type=int, default=0)
@click.option("--use_half_precision", type=bool, default=False)
@click.option("--compute_holdout", type=bool, default=False)
# Method 1 [Offline]: Policy loss.
@click.option("--eval_offline_policy_loss", type=bool, required=True)
# Method 2 [Offline]: Action diversity.
@click.option("--eval_offline_action_diversity", type=bool, required=True)
# Method 3 [Offline]: State diversity.
@click.option("--eval_offline_state_diversity", type=bool, required=True)
# Method 4 [Online]: State similarity.
@click.option("--eval_online_state_similarity", type=bool, required=True)
# Method 5 [Online]: DemoScore.
@click.option("--eval_online_demo_score", type=bool, required=True)
# Method 6 [Online]: TRAK influence.
@click.option("--eval_online_trak_influence", type=bool, required=True)
def main(
    exp_name: str,
    eval_dir: str,
    train_dir: str, 
    train_ckpt: str,
    result_date: str,
    overwrite: bool,
    device: str,
    seed: int,
    use_half_precision: bool,
    compute_holdout: bool,
    # Method 1 [Offline]: Policy loss.
    eval_offline_policy_loss: bool,
    # Method 2 [Offline]: Action diversity.
    eval_offline_action_diversity: bool,
    # Method 3 [Offline]: State diversity.
    eval_offline_state_diversity: bool,
    # Method 4 [Online]: State similarity.
    eval_online_state_similarity: bool,
    # Method 5 [Online]: DemoScore.
    eval_online_demo_score: bool,
    # Method 6 [Online]: TRAK influence (CUPID).
    eval_online_trak_influence: bool,
):
    # Set random seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)

    # Create save dir.
    save_dir = pathlib.Path(eval_dir) / exp_name
    save_dir.mkdir(exist_ok=True)
    
    # Load policy checkpoint.
    checkpoint_dir = pathlib.Path(train_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if isinstance(train_ckpt, str) and train_ckpt.isdigit(): 
        checkpoint = trak_util.get_index_checkpoint(checkpoints, int(train_ckpt))
    elif isinstance(train_ckpt, str):
        if train_ckpt == "best":
            checkpoint = trak_util.get_best_checkpoint(checkpoints)
        else:
            checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"
    else:
        raise ValueError(f"Checkpoint type {train_ckpt} is not supported.")
    policy, cfg = trak_util.get_policy_from_checkpoint(checkpoint, device=device)
    assert isinstance(policy, trak_util.SUPPORTED_POLICIES), f"Unsupported policy of type {type(policy)}"

    # Load training set and metadata.
    train_set: trak_util.DemoDatasetType = hydra.utils.instantiate(cfg.task.dataset)
    train_set_metadata = trak_util.get_dataset_metadata(cfg, train_set)
    
    # Load holdout set and metadata.
    holdout_set = train_set.get_holdout_dataset()
    holdout_set_metadata = trak_util.get_dataset_metadata(cfg, holdout_set)
    
    # Load test set and metadata.
    test_set: IterableDataset = BatchEpisodeDataset(
        batch_size=1,
        dataset_path=pathlib.Path(eval_dir) / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    test_set_metadata = trak_util.get_dataset_metadata(cfg, test_set)

    # Save memory; only need metadata.
    del train_set
    del holdout_set
    del test_set
    
    # Compute demonstration scores.
    return_dtype = np.float16 if use_half_precision else np.float32

    def save_result(result: Dict[str, Any], result_name: str) -> None:
        """Pickle result."""
        save_fname = save_dir / f"{result_name}.pkl"
        if save_fname.exists():
            if not overwrite:
                raise ValueError(f"Output path {save_fname} already exists.")
            os.remove(save_fname)
        with open(save_fname, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Results saved [{result_name}]")

    # Method 1 [Offline]: Policy loss.
    if eval_offline_policy_loss:
        load_key = "default_diffusion"
        save_result(
            offline_policy_loss_method(
                load_key=load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "offline_policy_loss"
        )

    # Method 2 [Offline]: Action diversity.
    if eval_offline_action_diversity:
        load_key = "default_diffusion"
        save_result(
            offline_action_diversity_method(
                load_key=load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "offline_action_diversity"
        )

    # Method 3 [Offline]: State diversity.
    if eval_offline_state_diversity:
        load_key = "default_state" if isinstance(policy, DiffusionUnetLowdimPolicy) else "default_image"
        save_result(
            offline_state_diversity_method(
                load_key=load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
                policy_embedding_dim=policy.emb_dim,
            ),
            "offline_state_diversity"
        )

    # Method 4 [Online]: State similarity.
    if eval_online_state_similarity:
        load_key = "default_state" if isinstance(policy, DiffusionUnetLowdimPolicy) else "default_image"
        save_result(
            online_state_similarity_method(
                load_key=load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
                policy_embedding_dim=policy.emb_dim,
            ),
            "online_state_similarity"
        )

    # Method 5 [Online]: DemoScore.
    if eval_online_demo_score:
        load_key = "default"
        save_result(
            online_demo_score_method(
                load_key=load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "online_demo_score"
        )

    # Method 6 [Online]: TRAK influence (CUPID).
    if eval_online_trak_influence:
        load_key = "default_diffusion"
        save_result(
            online_trak_influence_method(
                load_key=load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "online_trak_influence"
        )

    print(f"Experiment Complete: All results saved to {save_dir}.")


if __name__ == '__main__':
    main()
