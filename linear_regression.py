import torch
from torch import nn


class LinearRegressionModel(nn.module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    # Forward method to define the computation in the model.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
