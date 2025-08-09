from typing import Optional, Union

import torch
from transformers import AutoImageProcessor, AutoModel


class DinoV2FeatureExtractor:

    MODEL_FEATURE_DIMS = {
        "facebook/dinov2-small": 384,
        "facebook/dinov2-base": 768,
        "facebook/dinov2-large": 1024,
    }

    def __init__(
        self, 
        model_name: str = "facebook/dinov2-small", 
        device: Optional[Union[str, torch.device]] = None
    ):
        """Construct DINOv2 feature extractor."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(device, torch.device):
            self.device = torch.device(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Determine feature dimension from model name
        self.feature_dim = self.MODEL_FEATURE_DIMS.get(model_name, None)
        if self.feature_dim is None:
            raise ValueError(f"Unknown feature dimension for model: {model_name}.")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts DINOv2 features from a batch of images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]

        return features
