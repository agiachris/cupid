from typing import Union, Tuple, Optional

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
import omegaconf

import numpy as np
from numpy.lib.format import open_memmap

import torch
from torch.utils.data import DataLoader, IterableDataset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.episode_dataset import BatchEpisodeDataset
from diffusion_policy.dataset.episode_classifier_dataset import EpisodeClassifierDataset
from diffusion_policy.classifier.lowdim_state_success_classfier import LowdimStateSuccessClassifier
from diffusion_policy.common.trak_util import (
    get_index_checkpoint,
    get_best_checkpoint,
    get_policy_from_checkpoint,
    SUPPORTED_POLICIES,
    SUPPORTED_CLASSIFIERS,
    PolicyType,
    ClassifierType,
)


def get_model_and_config(
    train_dir: str,
    train_ckpt: str,
    device: str,
    assert_type: Optional[str] = "policy",
) -> Tuple[Union[PolicyType, ClassifierType], omegaconf.DictConfig]:
    """Return model and configuration from checkpoint."""
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
    
    # Load model checkpoint.
    model, cfg = get_policy_from_checkpoint(checkpoint, device=device)

    if assert_type == "policy":
        assert isinstance(model, SUPPORTED_POLICIES), f"Unsupported policy of type {type(model)}"
    elif assert_type == "classifier":
        assert isinstance(model, SUPPORTED_CLASSIFIERS), f"Unsupported classifier of type {type(model)}"

    return model, cfg


@click.command()
@click.option("--exp_name", type=str, required=True)
# Policies.
@click.option("--eval_dir", type=str, required=True)
@click.option("--train_dir", type=str, required=True)
@click.option("--train_ckpt", type=str, required=True)
# Classifiers.
@click.option("--classifier_train_dirs", type=str, multiple=True, required=True)
@click.option("--classifier_train_ckpts", type=str, multiple=True, required=True)
@click.option("--classifier_max_val_episodes", type=int, required=True)
# General.
@click.option("--batch_size", type=int, required=True)
@click.option('--overwrite', type=bool, default=False)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
# Optionals.
@click.option("--use_half_precision", type=bool, default=True)
@click.option("--compute_holdout", type=bool, default=False)
def main(
    exp_name: str,
    # Policies.
    eval_dir: str,
    train_dir: str, 
    train_ckpt: str,
    # Classifiers.
    classifier_train_dirs: Tuple[str, ...],
    classifier_train_ckpts: Tuple[str, ...],
    classifier_max_val_episodes: int,
    # General.
    batch_size: int,
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

    ### Step 1) DemoScore classifier cross-validation.

    # Load classifier validation set.
    val_set = EpisodeClassifierDataset(
        dataset_path=pathlib.Path(eval_dir) / "episodes",
        episode_cache_size=classifier_max_val_episodes,
        max_train_episodes=classifier_max_val_episodes,
        val_ratio=0.0,
        seed=seed,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
    )

    # Perform cross-validation for the best classifier checkpoint.
    best_train_dir = None
    best_train_ckpt = None
    best_val_score = float("inf")

    for cand_train_dir in classifier_train_dirs:
        for cand_train_ckpt in classifier_train_ckpts:
            # Load classifier.
            classifier, _ = get_model_and_config(
                train_dir=cand_train_dir,
                train_ckpt=cand_train_ckpt,
                device=device,
                assert_type="classifier"
            )
            assert isinstance(classifier, SUPPORTED_CLASSIFIERS)

            # Compute validation set loss.
            last_idx = 0
            loss_store = np.zeros(len(val_set), np.float32)
            for batch in tqdm.tqdm(val_loader, desc=f"Cross-validating C={cand_train_dir} ckpt={cand_train_ckpt}"):
                num_samples = batch["action"].shape[0]
                batch = dict_apply({k: batch[k] for k in ["obs", "action", "success"]}, lambda x: x.to(device))
                # Compute BCE loss.
                loss = classifier.compute_loss(batch, return_batch_loss=True).detach().cpu().numpy()
                loss_store[last_idx:last_idx+num_samples] = loss.squeeze().astype(np.float32)
                last_idx += num_samples
            
            val_score = loss_store.mean()
            if val_score < best_val_score:
                best_train_dir = cand_train_dir
                best_train_ckpt = cand_train_ckpt
                best_val_score = val_score
            del loss_store
    del val_set
    del val_loader
    assert best_train_dir is not None
    assert best_train_ckpt is not None

    # Load best DemoScore classifier.
    classifier, _ = get_model_and_config(
        train_dir=best_train_dir,
        train_ckpt=best_train_ckpt,
        device=device,
        assert_type="classifier"
    )

    ### Step 2) Score policy demonstration dataset and test dataset.
        
    # Load training set (no shuffle).
    _, test_cfg = get_model_and_config(
        train_dir=train_dir, 
        train_ckpt=train_ckpt,
        device=device,
        assert_type="policy",
    )

    train_set = hydra.utils.instantiate(test_cfg.task.dataset)
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
    train_set_demo_score = open_memmap(
        filename=save_dir / "train_set_demo_scores.mmap", 
        mode="w+", 
        shape=(train_set_size + holdout_set_size, 1), 
        dtype=dtype
    )
    test_set_demo_score = open_memmap(
        filename=save_dir / "test_set_demo_scores.mmap", 
        mode="w+", 
        shape=(test_set_size, 1), 
        dtype=dtype
    )

    def compute_dataset_demo_scores(
        dataloader: Union[DataLoader, IterableDataset], 
        demo_score_store: np.memmap,
        start_idx: int = 0,
        dataset_name: str = "train"
    ) -> int:
        """Compute DemoScore scores over all dataset samples."""
        last_idx = start_idx
        loader = iter(dataloader) if isinstance(dataloader, IterableDataset) else dataloader
        for batch in tqdm.tqdm(loader, desc=f"Scoring {dataset_name} set"):
            num_samples = batch["action"].shape[0]
            batch = dict_apply({k: batch[k] for k in ["obs"]}, lambda x: x.to(device))
            obs_dict = batch if isinstance(classifier, LowdimStateSuccessClassifier) else batch["obs"]
            # Compute DemoScore score.
            score = classifier.predict(obs_dict).detach().cpu().numpy()
            assert score.shape == (num_samples, 1)
            demo_score_store[last_idx:last_idx+num_samples, :] = score.astype(dtype)
            last_idx += num_samples

        return last_idx

    # Compute training set losses.
    last_idx = compute_dataset_demo_scores(
        dataloader=train_loader,
        demo_score_store=train_set_demo_score,
    )
    assert last_idx == train_set_size

    # Optionally compute holdout set losses.
    if compute_holdout:
        last_idx = compute_dataset_demo_scores(
            dataloader=holdout_loader,
            demo_score_store=train_set_demo_score,
            start_idx=last_idx,
            dataset_name="holdout"
        )
        assert last_idx == train_set_size + holdout_set_size
    
    # Compute test set losses.
    last_idx = compute_dataset_demo_scores(
        dataloader=test_set,
        demo_score_store=test_set_demo_score,
        dataset_name="test"
    )
    assert last_idx == test_set_size

    print(f"Completed computing DemoScores for {pathlib.Path(train_dir) / 'checkpoints' / train_ckpt}")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
