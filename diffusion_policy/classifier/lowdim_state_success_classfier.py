from typing import Dict, Any

import torch
import torch.nn.functional as F

from diffusion_policy.classifier.base_lowdim_state_success_classfier import BaseLowdimStateSuccessClassifier
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.classifier.mlp import MLPBinaryClassifier


class LowdimStateSuccessClassifier(BaseLowdimStateSuccessClassifier):
    
    def __init__(self, model: MLPBinaryClassifier, obs_dim: int, n_obs_steps: int):
        """Construct LowdimStateSuccessClassifier."""
        super().__init__()
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_dim = obs_dim
        self.n_obs_steps = n_obs_steps
        self.emb_dim = obs_dim * n_obs_steps

        print("Classifier params: %e" % sum(p.numel() for p in self.model.parameters()))

    # ========= inference  ============
    def predict(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs_dict: must include "obs" key"""
        
        # normalize input
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim

        # build input
        cond = nobs[:,:To].reshape(nobs.shape[0], -1)
        cond = cond.to(self.device).type(self.dtype)

        # make prediction
        pred = self.model(cond)
        return pred

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, Any], return_batch_loss: bool = False) -> torch.Tensor:
        # extract target
        target = batch["success"].clone().type(self.dtype)
        del batch["success"]

        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)

        # build input
        obs = nbatch['obs']
        cond = obs[:,:self.n_obs_steps,:].reshape(obs.shape[0], -1)
        
        # make prediction and compute loss
        pred = self.model(cond)
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        return loss.mean(axis=1) if return_batch_loss else loss.mean()
    