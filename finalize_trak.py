from typing import Optional

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import trak
import click
import hydra
import random
import pathlib
import numpy as np
import torch

from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.common.trak_util import (
    get_index_checkpoint,
    get_policy_from_checkpoint,
    get_best_checkpoint,
    get_parameter_names,
)

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy


POLICIES = (DiffusionUnetLowdimPolicy, DiffusionUnetHybridImagePolicy)
MODELOUT_FN_DIR = "diffusion_policy.data_attribution.modelout_functions"
GRADIENT_CO_DIR = "diffusion_policy.data_attribution.gradient_computers"


@click.command()
# Experiment params.
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, required=True)
@click.option("--model_keys", type=str, default=None)
# TRAK params.
@click.option("--num_ckpts", type=int, required=True)
@click.option("--modelout_fn", type=str, required=True)
@click.option("--gradient_co", type=str, required=True) 
@click.option("--proj_dim", type=int, default=2048)
@click.option("--proj_max_batch_size", type=int, default=32)
@click.option("--lambda_reg", type=float, default=0.0)
@click.option("--use_half_precision", type=bool, default=False)
# Other params.
@click.option("--batch_size", type=int, default=32)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
# Task params.
@click.option("--loss_fn", type=str, default=None)
# Optional: Finalize.
@click.option("--featurize_holdout", type=bool, default=False)
def main(
    # Experiment params.
    exp_name: str,
    eval_dir: str,
    train_dir: str, 
    train_ckpt: str, 
    model_keys: Optional[str],
    # TRAK params.
    num_ckpts: int,
    modelout_fn: str,
    gradient_co: str,
    proj_dim: int,
    proj_max_batch_size: int,
    lambda_reg: float,
    use_half_precision: bool,
    # Other params.
    batch_size: int,
    device: str,
    seed: int,
    # Task params.
    loss_fn: Optional[str],
    # Optional: Scores.
    featurize_holdout: bool,
):
    # Set random seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)

    # Find checkpoint.
    checkpoint_dir = pathlib.Path(train_dir) / "checkpoints"
    checkpoints = list(checkpoint_dir.iterdir())
    if isinstance(train_ckpt, str) and train_ckpt.isdigit(): 
        checkpoint = get_index_checkpoint(checkpoints, int(train_ckpt))
    elif isinstance(train_ckpt, str):
        if train_ckpt == "best":
            checkpoint = get_best_checkpoint(checkpoints)
        else:
            checkpoint = checkpoint_dir / f"{train_ckpt}.ckpt"
    else:
        raise ValueError(f"Checkpoint type {train_ckpt} is not supported.")

    # Load policy checkpoint.
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)
    assert isinstance(policy, POLICIES), f"Policy class {type(policy)} not supported."
    if model_keys:
        assert isinstance(model_keys, str)
        model_keys = model_keys.split(',')
    grad_wrt = get_parameter_names(policy, model_keys) if model_keys is not None else None

    # Train, holdout, and test set lengths.
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    train_set_size = len(train_set)
    holdout_set_size = 0
    if featurize_holdout:
        holdout_set = train_set.get_holdout_dataset()
        holdout_set_size = len(holdout_set)
    test_set_size = len(
        BatchEpisodeDataset(
            batch_size=batch_size,
            dataset_path=pathlib.Path(eval_dir) / "episodes",
            exec_horizon=1,
            sample_history=0,
        )
    )

    # Construct computers.
    task_kwargs = {}
    if loss_fn is not None:
        task_kwargs["loss_fn"] = loss_fn
    task = hydra.utils.get_class(f"{MODELOUT_FN_DIR}.{modelout_fn}")(**task_kwargs)
    gradient_computer = hydra.utils.get_class(f"{GRADIENT_CO_DIR}.{gradient_co}")

    # Create policy traker.
    save_dir = pathlib.Path(eval_dir) / exp_name
    traker = trak.TRAKer(
        model=policy,
        task=task,
        train_set_size=train_set_size + holdout_set_size,
        gradient_computer=gradient_computer,
        device=device,
        grad_wrt=grad_wrt,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_max_batch_size,
        lambda_reg=lambda_reg,
        save_dir=str(save_dir),
        use_half_precision=use_half_precision,
    )
 
    # Finalize test set scores.
    scores = np.array(
        traker.finalize_scores(
            exp_name="all_episodes",
            model_ids=list(range(num_ckpts)),
            allow_skip=False
        )
    )
    assert scores.shape == (train_set_size + holdout_set_size, test_set_size)
    print(f"Completed scoring for traker at {save_dir}")


if __name__ == '__main__':
    main()
