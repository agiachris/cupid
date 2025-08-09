from typing import List

import torch
from torch import nn


class MLPBinaryClassifier(nn.Module):

    def __init__(
        self, 
        input_dim: int,
        hidden_dims: List[int] = [16, 16, 16],
        dropout: float = 0.3,
    ):
        """Construct MLPBinaryClassifier."""
        super().__init__()

        # Create hidden layers with ReLU and dropout.
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Create output layer.
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        # Sequential module.
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)