from typing import Dict

import torch
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BaseImageStateSuccessClassifier(ModuleAttrMixin):
    
    # ========= inference  ============
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml
    def predict(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict:
            str: B,To,*
        return: B, 1
        """
        raise NotImplementedError()

    # reset state for stateful success classifiers
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
