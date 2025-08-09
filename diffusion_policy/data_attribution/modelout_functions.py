from typing import List, Dict, Iterable

import torch
from einops import reduce
import torch.nn.functional as F
from trak.modelout_functions import AbstractModelOutput

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy


class DiffusionLowdimFunctionalModelOutput(AbstractModelOutput):
    """Empirical DDPM loss for diffusion policies."""

    LOSS_FNS = {"ddpm", "square", "average"}

    def __init__(self, loss_fn: str):
        """Construct DiffusionLowdimFunctionalModelOutput."""
        super().__init__()
        assert loss_fn in self.LOSS_FNS
        self._loss_fn = loss_fn

    def get_output(
        self,
        model: DiffusionUnetLowdimPolicy,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        tsteps: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the empirical DDPM loss for a given observation-action pair."""
        # Extract weights / buffers of interest.
        model_weights = {k[len("model."):]: v for k, v in weights.items() if k.startswith("model.")}
        model_buffers = {k[len("model."):]: v for k, v in buffers.items() if k.startswith("model.")}

        # Batchify inputs.
        batch = {
            "obs": obs.unsqueeze(0),
            "action": action.unsqueeze(0)
        }   
        tsteps = tsteps.long()

        # Normalize input.
        assert 'valid_mask' not in batch
        nbatch = model.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # Handle different ways of passing observation.
        local_cond = None
        global_cond = None
        trajectory = action
        if model.obs_as_local_cond:
            # Zero out observations after n_obs_steps.
            local_cond = obs
            local_cond[:,model.n_obs_steps:,:] = 0
        elif model.obs_as_global_cond:
            global_cond = obs[:,:model.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if model.pred_action_steps_only:
                To = model.n_obs_steps
                start = To
                if model.oa_step_convention:
                    start = To - 1
                end = start + model.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # Generate impainting mask.
        if model.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = model.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images.
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Add noise to the clean images according to the noise magnitude at each timestep.
        noisy_trajectory = model.noise_scheduler.add_noise(trajectory, noise, tsteps)
        
        # Compute loss mask.
        loss_mask = ~condition_mask

        # Apply conditioning.
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual.
        pred: torch.Tensor = torch.func.functional_call(
            model.model,
            (model_weights, model_buffers),
            (noisy_trajectory, tsteps),
            {
                "local_cond": local_cond,
                "global_cond": global_cond
            }
        )

        pred_type = model.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Compute model output function.
        loss = None
        if self._loss_fn == "ddpm":
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        elif self._loss_fn == "square":
            loss = pred * loss_mask.type(pred.dtype)
            loss = reduce(loss ** 2, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        elif self._loss_fn == "average":
            loss = pred * loss_mask.type(pred.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        else:
            raise ValueError(f"Unsupported loss function {self._loss_fn}.")
        return loss
    
    def get_out_to_loss_grad(
        self, 
        model: DiffusionUnetLowdimPolicy,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Computes the (reweighting term Q in the paper)."""        
        return torch.ones(batch["action"].shape[0]).to(batch["action"].device).unsqueeze(-1)


class DiffusionHybridImageFunctionalModelOutput(AbstractModelOutput):
    """Empirical DDPM loss for diffusion policies."""

    LOSS_FNS = {"ddpm", "square", "average"}

    def __init__(self, loss_fn: str):
        """Construct DiffusionHybridImageFunctionalModelOutput."""
        super().__init__()
        assert loss_fn in self.LOSS_FNS
        self._loss_fn = loss_fn

    def get_output(
        self,
        model: DiffusionUnetHybridImagePolicy,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        obs_keys: List[str],
        tsteps: torch.Tensor,
        action: torch.Tensor,
        *obs: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the empirical DDPM loss for a given observation-action pair."""
        # Extract weights / buffers of interest.
        obs_encoder_weights = {k[len("obs_encoder."):]: v for k, v in weights.items() if k.startswith("obs_encoder.")}
        obs_encoder_buffers = {k[len("obs_encoder."):]: v for k, v in buffers.items() if k.startswith("obs_encoder.")}
        model_weights = {k[len("model."):]: v for k, v in weights.items() if k.startswith("model.")}
        model_buffers = {k[len("model."):]: v for k, v in buffers.items() if k.startswith("model.")}

        # Batchify inputs.
        batch = {
            "obs": {k: v.unsqueeze(0) for k, v in zip(obs_keys, obs)},
            "action": action.unsqueeze(0)
        }
        tsteps = tsteps.long()

        # normalize input
        assert 'valid_mask' not in batch
        nobs = model.normalizer.normalize(batch['obs'])
        nactions = model.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if model.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:model.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = torch.func.functional_call(
                model.obs_encoder,
                (obs_encoder_weights, obs_encoder_buffers),
                this_nobs
            )
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = torch.func.functional_call(
                model.obs_encoder,
                (obs_encoder_weights, obs_encoder_buffers),
                this_nobs
            )
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = model.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = model.noise_scheduler.add_noise(trajectory, noise, tsteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual.
        pred: torch.Tensor = torch.func.functional_call(
            model.model,
            (model_weights, model_buffers),
            (noisy_trajectory, tsteps),
            {
                "local_cond": local_cond,
                "global_cond": global_cond
            }
        )

        pred_type = model.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Compute model output function.
        loss = None
        if self._loss_fn == "ddpm":
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        elif self._loss_fn == "square":
            loss = pred * loss_mask.type(pred.dtype)
            loss = reduce(loss ** 2, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        elif self._loss_fn == "average":
            loss = pred * loss_mask.type(pred.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        else:
            raise ValueError(f"Unsupported loss function {self._loss_fn}.")
        return loss
    
    def get_out_to_loss_grad(
        self, 
        model: DiffusionUnetHybridImagePolicy,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Computes the (reweighting term Q in the paper)."""        
        return torch.ones(batch["action"].shape[0]).to(batch["action"].device).unsqueeze(-1)
