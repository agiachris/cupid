from typing import Dict, Any, List

import torch
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.classifier.base_image_state_success_classfier import BaseImageStateSuccessClassifier
from diffusion_policy.model.classifier.mlp import MLPBinaryClassifier
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class ImageStateSuccessClassifier(BaseImageStateSuccessClassifier):
    
    def __init__(
        self, 
        obs_encoder: MultiImageObsEncoder,
        n_obs_steps: int,
        hidden_dims: List[int] = [32, 32, 32],
        dropout: float = 0.3,
    ):
        """Construct ImageStateSuccessClassifier."""
        super().__init__()

        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps

        # create classifier model
        self.obs_encoder = obs_encoder
        self.model = MLPBinaryClassifier(
            input_dim=global_cond_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.n_obs_steps = n_obs_steps
        self.emb_dim = obs_feature_dim * n_obs_steps

        print("Classifier params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def predict(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs_dict: must include "obs" key"""
        assert 'past_action' not in obs_dict # not implemented yet
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps

        # build input
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        cond: torch.Tensor = self.obs_encoder(this_nobs).reshape(B, -1)
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
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]

        # build input
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        cond: torch.Tensor = self.obs_encoder(this_nobs).reshape(batch_size, -1)

        # make prediction and compute loss
        pred = self.model(cond)
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        return loss.mean(axis=1) if return_batch_loss else loss.mean()
