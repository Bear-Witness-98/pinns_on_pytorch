import numpy as np
import torch
from torch import nn


def pinn_loss(output: torch.Tensor, target: torch.Tensor):
    # start creating a function to adjust to a simple straight line
    # (known functions), and then try to adjust to the complex diff
    # equation loss functions.
    # Separate in: 1) initial conditions 2) line appox.
    mse_loss = nn.MSELoss()
    return mse_loss(output, target)
