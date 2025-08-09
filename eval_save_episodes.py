import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import shutil
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.trak_util import get_best_checkpoint, get_index_checkpoint


@click.command()
@click.option('--output_dir', type=str, required=True)
@click.option('--train_dir', type=str, required=True)
@click.option('--train_ckpt', type=str, required=True)
@click.option('--num_episodes', type=int, default=100)
@click.option('--test_start_seed', type=int, default=100000)
@click.option('--overwrite', type=bool, default=False)
@click.option('--device', type=str, default='cuda:0')
def main(
    output_dir: str, 
    train_dir: str, 
    train_ckpt: str, 
    num_episodes: int,
    test_start_seed: int,
    overwrite: bool,
    device: str,
):  
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

    # Create output dir.
    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output path {output_dir} already exists!")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint.
    payload = torch.load(open(str(checkpoint), 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    # Update configuration for evaluation.
    cfg.task.env_runner.n_envs = 1
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = num_episodes
    cfg.task.env_runner.n_test_vis = num_episodes
    cfg.task.env_runner.test_start_seed = test_start_seed

    # Construct workspace.
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy from workspace.
    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # Run eval.
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        save_episodes=True)
    runner_log = env_runner.run(policy)
    
    # Dump log to json.
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
