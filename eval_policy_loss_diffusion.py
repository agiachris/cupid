from typing import Union

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import tqdm
import click
import hydra
import shutil
import random
import pathlib

import numpy as np
from numpy.lib.format import open_memmap

import torch
from torch.utils.data import DataLoader, IterableDataset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.common.trak_util import (
    get_index_checkpoint,
    get_best_checkpoint,
    get_policy_from_checkpoint,
    SUPPORTED_POLICIES,
)


@click.command()
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option("--num_timesteps", type=int, required=True)
@click.option('--overwrite', type=bool, default=False)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
# Optionals.
@click.option("--use_half_precision", type=bool, default=True)
@click.option("--compute_holdout", type=bool, default=False)
def main(
    exp_name: str,
    eval_dir: str,
    train_dir: str, 
    train_ckpt: str,
    batch_size: int,
    num_timesteps: int,
    overwrite: bool,
    device: str,
    seed: int,
    # Optionals.
    use_half_precision: bool,
    compute_holdout: bool,
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
    assert isinstance(policy, SUPPORTED_POLICIES), f"Unsupported policy of type {type(policy)}"

    # Load training set (no shuffle).
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    train_set_size = len(train_set)
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
    )

    # Optional: Load holdout set (no shuffle).
    holdout_set_size = 0
    if compute_holdout:
        holdout_set = train_set.get_holdout_dataset()
        holdout_set_size = len(holdout_set)
        holdout_loader = DataLoader(
            holdout_set, 
            batch_size=batch_size,
            num_workers=8,
            persistent_workers=False,
            pin_memory=True,
            shuffle=False,
        )

    # Load test set (no shuffle).
    test_set: IterableDataset = BatchEpisodeDataset(
        batch_size=batch_size,
        dataset_path=pathlib.Path(eval_dir) / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    test_set_size = len(test_set)

    # Create save dir and loss arrays.
    save_dir = pathlib.Path(eval_dir) / exp_name
    if save_dir.exists():
        if not overwrite:
            raise ValueError(f"Output path {save_dir} already exists.")
        shutil.rmtree(save_dir)        
    save_dir.mkdir()

    dtype = np.float16 if use_half_precision else np.float32
    train_set_loss = open_memmap(
        filename=save_dir / "train_set_loss.mmap", 
        mode="w+", 
        shape=(train_set_size + holdout_set_size, 1), 
        dtype=dtype
    )
    test_set_loss = open_memmap(
        filename=save_dir / "test_set_loss.mmap", 
        mode="w+", 
        shape=(test_set_size, 1), 
        dtype=dtype
    )

    def compute_dataset_losses(
        dataloader: Union[DataLoader, IterableDataset], 
        loss_store: np.memmap,
        start_idx: int = 0,
        dataset_name: str = "train"
    ) -> int:
        """Compute loss over all dataset samples."""
        last_idx = start_idx
        loader = iter(dataloader) if isinstance(dataloader, IterableDataset) else dataloader
        for batch in tqdm.tqdm(loader, desc=f"Scoring {dataset_name} set"):
            num_samples = batch["action"].shape[0]
            batch = dict_apply({k: batch[k] for k in ["obs", "action"]}, lambda x: x.to(device))
            # Compute policy loss.
            for _ in range(num_timesteps):
                loss = policy.compute_loss(batch, return_batch_loss=True).detach().cpu().numpy()
                assert loss.shape == (num_samples,)
                loss_store[last_idx:last_idx+num_samples, :] += loss[:, None].astype(dtype) / num_timesteps
            last_idx += num_samples
        return last_idx

    # Compute training set losses.
    last_idx = compute_dataset_losses(
        dataloader=train_loader,
        loss_store=train_set_loss,
    )
    assert last_idx == train_set_size

    # Optionally compute holdout set losses.
    if compute_holdout:
        last_idx = compute_dataset_losses(
            dataloader=holdout_loader,
            loss_store=train_set_loss,
            start_idx=last_idx,
            dataset_name="holdout"
        )
        assert last_idx == train_set_size + holdout_set_size
    
    # Compute test set losses.
    last_idx = compute_dataset_losses(
        dataloader=test_set,
        loss_store=test_set_loss,
        dataset_name="test"
    )
    assert last_idx == test_set_size

    print(f"Completed computing losses for {checkpoint}")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
