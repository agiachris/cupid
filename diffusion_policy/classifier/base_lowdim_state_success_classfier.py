from typing import Dict

import torch
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BaseLowdimStateSuccessClassifier(ModuleAttrMixin):  
    
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict:
            obs: B,To,Do
        return: 
            pred: B, 1
        To = 3
        |o|o|o|
        | | |p|
        """
        raise NotImplementedError()

    # reset state for stateful success classifiers
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()

    