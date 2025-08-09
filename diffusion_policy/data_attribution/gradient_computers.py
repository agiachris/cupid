from typing import Iterable, Optional, Dict

import logging

import torch
from torch import Tensor

from trak.gradient_computers import AbstractGradientComputer

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.data_attribution.modelout_functions import DiffusionLowdimFunctionalModelOutput

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.data_attribution.modelout_functions import DiffusionHybridImageFunctionalModelOutput


def _accumulate_vectorize(g: Tensor, arr: Tensor):
    """Accumulates result into arrary.

    Gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]`.
    """
    pointer = 0
    for param in g.values():
        num_param = param[0].numel()
        arr[:, pointer : pointer + num_param] += param.flatten(start_dim=1).data
        pointer += num_param


def _average_vectorize(g: Tensor, arr: Tensor, num_timesteps: int):
    """Accumulates averaged result into array.

    Gradients are given as a dict :code:`(name_w0: grad_w0, ... name_wp:
    grad_wp)` where :code:`p` is the number of weight matrices. each
    :code:`grad_wi` has shape :code:`[batch_size, ...]` this function flattens
    :code:`g` to have shape :code:`[batch_size, num_params]` and averages the
    contributions across timesteps.

    Args:
        g (Tensor): Dictionary of gradients for each parameter
        arr (Tensor): Array to accumulate results into
        num_timesteps (int): Number of timesteps to average over
    """
    pointer = 0
    for param in g.values():
        num_param = param[0].numel()
        arr[:, pointer : pointer + num_param] += param.flatten(start_dim=1).data / num_timesteps
        pointer += num_param


class DiffusionLowdimFunctionalGradientComputer(AbstractGradientComputer):
    """Functional gradient computer for DiffusionUnetLowdimPolicy."""

    def __init__(
        self,
        model: DiffusionUnetLowdimPolicy,
        task: DiffusionLowdimFunctionalModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,   
    ):
        """Construct DiffusionLowdimFunctionalGradientComputer."""
        super().__init__(model, task, grad_dim, dtype, device)
        self.logger = logging.getLogger("DiffusionLowdimFunctionalGradientComputer")
        self.grad_wrt = grad_wrt
        if self.grad_wrt is not None:
            assert isinstance(self.grad_wrt, list)
            self.logger.info(f"Computing gradients for {len(self.grad_wrt)} parameters.")
        self.load_model_params(model)

    def load_model_params(self, model: DiffusionUnetLowdimPolicy) -> None:
        """Load model parameters and filter by grad_wrt."""
        # Get functional weights and buffers.
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

        # Filter weights based on grad_wrt.
        if self.grad_wrt is not None:
            self.func_weights = {k: self.func_weights[k] for k in self.grad_wrt if k in self.func_weights.keys()}
            missing_keys = [k for k in self.grad_wrt if k not in self.func_weights.keys()]
            if len(missing_keys) > 0:
                self.logger.warning(f"Weights not found in the model: {missing_keys}")
        
        # Check number of parameters.
        assert self.grad_dim == sum([v.numel() for v in self.func_weights.values()])

    def compute_per_sample_grad(self, batch: Dict[str, Tensor]) -> Tensor:
        """Computes per-sample gradients of the model output function."""
        # Taking the gradient wrt weights (second argument of get_output, hence argnums=1).
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output, has_aux=False, argnums=1
        )

        # Extract batch elements.
        timesteps = batch["timesteps"]  # (B, num_timesteps)
        action = batch["action"]        # (B, Ta, Da)
        obs = batch["obs"]              # (B, To, Do)
        
        # Create gradient store.
        batch_size = action.shape[0]
        grads = torch.zeros(
            size=(batch_size, self.grad_dim),
            dtype=action.dtype,
            device=action.device,
        )

        for i_tstep in range(timesteps.shape[1]):
            tsteps = timesteps[:, i_tstep:i_tstep+1]
            # Map over batch dimensions (hence 0 for each batch dimension, and None for model params).
            _average_vectorize(
                g=torch.func.vmap(
                    grads_loss,
                    in_dims=(None, None, None, 0, 0, 0),
                    randomness="different",
                )(
                    self.model,
                    self.func_weights,
                    self.func_buffers,
                    tsteps,
                    action,
                    obs,
                ),
                arr=grads,
                num_timesteps=timesteps.shape[1]
            )
        return grads

    def compute_loss_grad(self, batch: Dict[str, Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output."""
        return self.modelout_fn.get_out_to_loss_grad(self.model, self.func_weights, self.func_buffers, batch)
    

class DiffusionHybridImageFunctionalGradientComputer(AbstractGradientComputer):
    """Functional gradient computer for DiffusionUnetHybridImagePolicy."""

    def __init__(
        self,
        model: DiffusionUnetHybridImagePolicy,
        task: DiffusionHybridImageFunctionalModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,   
    ):
        """Construct DiffusionHybridImageFunctionalGradientComputer."""
        super().__init__(model, task, grad_dim, dtype, device)
        self.logger = logging.getLogger("DiffusionHybridImageFunctionalGradientComputer")
        self.grad_wrt = grad_wrt
        if self.grad_wrt is not None:
            assert isinstance(self.grad_wrt, list)
            self.logger.info(f"Computing gradients for {len(self.grad_wrt)} parameters.")
        
        self.obs_keys = model.obs_keys
        self.load_model_params(model)

    def load_model_params(self, model: DiffusionUnetHybridImagePolicy) -> None:
        """Load model parameters and filter by grad_wrt."""
        # Get functional weights and buffers.
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

        # Filter weights based on grad_wrt.
        if self.grad_wrt is not None:
            self.func_weights = {k: self.func_weights[k] for k in self.grad_wrt if k in self.func_weights.keys()}
            missing_keys = [k for k in self.grad_wrt if k not in self.func_weights.keys()]
            if len(missing_keys) > 0:
                self.logger.warning(f"Weights not found in the model: {missing_keys}")
        
        # Check number of parameters.
        assert self.grad_dim == sum([v.numel() for v in self.func_weights.values()])

    def compute_per_sample_grad(self, batch: Dict[str, Tensor]) -> Tensor:
        """Computes per-sample gradients of the model output function."""
        # Taking the gradient wrt weights (second argument of get_output, hence argnums=1).
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output, has_aux=False, argnums=1
        )

        # Extract batch elements.
        timesteps = batch["timesteps"]                  # (B, num_timesteps)
        action = batch["action"]                        # (B, Ta, Da)
        obs = [batch["obs"][k] for k in self.obs_keys]  # List[(B, To, Do)]
        
        # Create gradient store.
        batch_size = action.shape[0]
        grads = torch.zeros(
            size=(batch_size, self.grad_dim),
            dtype=action.dtype,
            device=action.device,
        )

        for i_tstep in range(timesteps.shape[1]):
            tsteps = timesteps[:, i_tstep:i_tstep+1]
            # Map over batch dimensions (hence 0 for each batch dimension, and None for model params).
            _average_vectorize(
                g=torch.func.vmap(
                    grads_loss,
                    in_dims=(None, None, None, None, 0, 0, *([0] * len(self.obs_keys))),
                    randomness="different",
                )(
                    self.model,
                    self.func_weights,
                    self.func_buffers,
                    self.obs_keys,
                    tsteps,
                    action,
                    *obs,
                ),
                arr=grads,
                num_timesteps=timesteps.shape[1]
            )
        return grads

    def compute_loss_grad(self, batch: Dict[str, Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output."""
        return self.modelout_fn.get_out_to_loss_grad(self.model, self.func_weights, self.func_buffers, batch)
