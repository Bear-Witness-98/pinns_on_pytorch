import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        logits = self.linear_relu_stack(x)
        return logits
