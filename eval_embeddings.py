from typing import Union, Callable, Dict, Any

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import tqdm
import click
import hydra
import random
import pathlib
from functools import partial

import numpy as np
from numpy.lib.format import open_memmap

import torch
from torch.utils.data import DataLoader, IterableDataset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.common.trak_util import (
    get_index_checkpoint,
    get_best_checkpoint,
    get_policy_from_checkpoint,
    SUPPORTED_POLICIES,
)
from diffusion_policy.model.embedding.huggingface import DinoV2FeatureExtractor



def batch_embedding_wrapper(
    batch: Dict[str, Any],
    embedding_fn: Callable[[torch.Tensor], torch.Tensor],
    render_name: str,
) -> torch.Tensor:
    """Computes image embedding from batch."""
    images: torch.Tensor = batch["obs"][render_name]
    images = images[:, -1, ...]
    embeddings = embedding_fn(images)
    assert images.shape[0] == embeddings.shape[0]
    return embeddings


@click.command()
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, required=True)
@click.option("--batch_size", type=int, required=True)
@click.option('--overwrite', type=bool, default=False)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
# Embeddings.
@click.option("--policy_embeddings", type=bool, default=True)
@click.option("--dinov2_embeddings", type=bool, default=True)
# Optionals.
@click.option("--use_half_precision", type=bool, default=True)
@click.option("--compute_holdout", type=bool, default=False)
def main(
    exp_name: str,
    eval_dir: str,
    train_dir: str, 
    train_ckpt: str,
    batch_size: int,
    overwrite: bool,
    device: str,
    seed: int,
    # Embeddings.
    policy_embeddings: bool,
    dinov2_embeddings: bool,
    # Optionals.
    use_half_precision: bool,
    compute_holdout: bool,
):
    # Set random seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)

    # Load policy checkpoint.
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
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)

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

    # Embedding methods.
    embedding_fns = []
    embedding_dims = []
    embedding_names = []
    if policy_embeddings:
        assert isinstance(policy, SUPPORTED_POLICIES), f"Unsupported policy of type {type(policy)}"
        assert cfg.policy.obs_as_global_cond, "Assumes the use of global observation conditioning."
        embedding_fns.append(policy.compute_obs_embedding)
        embedding_dims.append(policy.emb_dim)
        embedding_names.append("policy")
    if dinov2_embeddings:
        assert isinstance(policy, (DiffusionUnetImagePolicy, DiffusionUnetHybridImagePolicy))
        dinov2 = DinoV2FeatureExtractor(device=device)
        embedding_fns.append(partial(
            batch_embedding_wrapper, 
            embedding_fn=dinov2.extract_features, 
            render_name=getattr(cfg.task.env_runner, "render_obs_key", "image"),
        ))
        embedding_dims.append(dinov2.feature_dim)
        embedding_names.append("dinov2")
    
    # Create save dir.
    save_dir = pathlib.Path(eval_dir) / exp_name
    save_dir.mkdir(exist_ok=True)

    # Iterate over embedding methods and store embeddings.
    dtype = np.float16 if use_half_precision else np.float32
    for embedding_fn, embedding_dim, embedding_name in zip(embedding_fns, embedding_dims, embedding_names):
        
        # Create embedding arrays.
        train_set_emb_fname = save_dir / f"train_set_{embedding_name}_emb.mmap"
        test_set_emb_fname = save_dir / f"test_set_{embedding_name}_emb.mmap"
        if any(fname.exists() for fname in [train_set_emb_fname, test_set_emb_fname]):
            if not overwrite:
                raise ValueError(f"{embedding_name.title()} embeddings at {save_dir} already exists.")
            os.remove(train_set_emb_fname)
            os.remove(test_set_emb_fname)
        train_set_emb = open_memmap(
            filename=train_set_emb_fname,
            mode="w+", 
            shape=(train_set_size + holdout_set_size, embedding_dim), 
            dtype=dtype
        )
        test_set_emb = open_memmap(
            filename=test_set_emb_fname,
            mode="w+", 
            shape=(test_set_size, embedding_dim), 
            dtype=dtype
        )

        # Compute embeddings.
        def compute_dataset_embeddings(
            dataloader: Union[DataLoader, IterableDataset], 
            emb_store: np.memmap,
            start_idx: int = 0,
            dataset_name: str = "train"
        ) -> int:
            """Compute loss over all dataset samples."""
            last_idx = start_idx
            loader = iter(dataloader) if isinstance(dataloader, IterableDataset) else dataloader
            for batch in tqdm.tqdm(loader, desc=f"Embedding {dataset_name} set"):
                num_samples = batch["action"].shape[0]
                batch = dict_apply({k: batch[k] for k in ["obs", "action"]}, lambda x: x.to(device))
                # Compute embeddings.
                emb: np.ndarray = embedding_fn(batch).detach().cpu().numpy()
                assert emb.shape == (num_samples, embedding_dim)
                emb_store[last_idx:last_idx+num_samples, :] = emb.astype(dtype)
                last_idx += num_samples
            return last_idx
        
        # Compute training set embeddings.
        last_idx = compute_dataset_embeddings(
            dataloader=train_loader,
            emb_store=train_set_emb,
        )
        assert last_idx == train_set_size

        # Optionally compute holdout set embeddings.
        if compute_holdout:
            last_idx = compute_dataset_embeddings(
                dataloader=holdout_loader,
                emb_store=train_set_emb,
                start_idx=last_idx,
                dataset_name="holdout"
            )
            assert last_idx == train_set_size + holdout_set_size
        
        # Compute test set embeddings.
        last_idx = compute_dataset_embeddings(
            dataloader=test_set,
            emb_store=test_set_emb,
            dataset_name="test"
        )
        assert last_idx == test_set_size

        print(f"Completed computing {embedding_name} embeddings for {checkpoint}")
        print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
