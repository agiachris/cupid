from typing import Dict, Any

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

import torch
from torch.utils.data import IterableDataset
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset

from diffusion_policy.common import results_util
from diffusion_policy.common import trak_util
from diffusion_policy.common import error_util


def policy_loss_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute rollout scores by policy loss."""
    # Load policy losses.
    _, test_loss = results_util.get_policy_losses(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=False,
        load_test=True,
        **results_util.LOAD_POLICY_LOSS_KWARGS[load_key],
    )
    assert isinstance(test_loss, np.ndarray)

    # Compute rollout scores.
    result = {
        "test": error_util.sample_to_trajectory_scores(
            sample_scores=test_loss[:test_set_metadata["num_samples"]],
            num_eps=test_set_metadata["num_eps"],
            ep_idxs=test_set_metadata["ep_idxs"],
            ep_lens=test_set_metadata["ep_lens"],
            aggr_fn=np.sum,
            return_dtype=return_dtype,
        )
    }

    assert not any(np.any(v == 0.0) for v in result.values()), "Result contains a zero score"
    
    print(f"Completed: Policy Loss.")
    return result


def action_likelihood_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute rollout scores by action likelihood."""
    # Load action likelihoods.
    _, test_likelihood = results_util.get_action_likelihoods(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=False,
        load_test=True,
        **results_util.LOAD_ACTION_LIKELIHOOD_KWARGS[load_key],
    )
    assert isinstance(test_likelihood, np.ndarray)

    # Compute rollout scores.
    result = {
        "test": error_util.sample_to_trajectory_scores(
            sample_scores=test_likelihood[:test_set_metadata["num_samples"]],
            num_eps=test_set_metadata["num_eps"],
            ep_idxs=test_set_metadata["ep_idxs"],
            ep_lens=test_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }

    assert not any(np.any(v == 0.0) for v in result.values()), "Result contains a zero score"
    
    print(f"Completed: Action Likelihood.")
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
# Method 1: Policy loss.
@click.option("--eval_policy_loss", type=bool, required=True)
@click.option("--eval_policy_loss_load_key", type=str, required=True)
# Method 2: Action likelihood.
@click.option("--eval_action_likelihood", type=bool, required=True)
@click.option("--eval_action_likelihood_load_key", type=str, required=True)
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
    # Method 1: Policy loss.
    eval_policy_loss: bool,
    eval_policy_loss_load_key: str,
    # Method 2: Action likelihood.
    eval_action_likelihood: bool,
    eval_action_likelihood_load_key: str,
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
    del policy

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
    
    # Compute rollout scores.
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

    # Method 1: Policy loss.
    if eval_policy_loss:
        save_result(
            policy_loss_method(
                load_key=eval_policy_loss_load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            f"policy_loss-{eval_policy_loss_load_key}"
        )
    
    # Method 2: Action likelihood.
    if eval_action_likelihood:
        save_result(
            action_likelihood_method(
                load_key=eval_action_likelihood_load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            f"action_likelihood-{eval_action_likelihood_load_key}"
        )

    print(f"Experiment Complete: All results saved to {save_dir}.")


if __name__ == '__main__':
    main()
