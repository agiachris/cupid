from typing import Dict, List, Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.classifier.base_image_state_success_classfier import BaseImageStateSuccessClassifier
from diffusion_policy.model.classifier.mlp import MLPBinaryClassifier

from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class HybridImageStateSuccessClassifier(BaseImageStateSuccessClassifier):
    
    def __init__(
        self, 
        shape_meta: Dict[str, Any],
        n_obs_steps: int,
        crop_shape: Tuple[int, int] = (76, 76),
        hidden_dims: List[int] = [32, 32, 32],
        dropout: float = 0.3,
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
    ):
        """Construct HybridImageStateSuccessClassifier."""
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create classifier model
        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps
        model = MLPBinaryClassifier(
            input_dim=global_cond_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.n_obs_steps = n_obs_steps
        self.emb_dim = obs_feature_dim * n_obs_steps
        self.obs_keys = list(obs_shape_meta.keys())

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
        Do = self.obs_feature_dim
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
        cond = self.obs_encoder(this_nobs).reshape(batch_size, -1)
        
        # make prediction and compute loss
        pred = self.model(cond)
        loss = F.binary_cross_entropy(pred, target, reduction="none")
        return loss.mean(axis=1) if return_batch_loss else loss.mean()
