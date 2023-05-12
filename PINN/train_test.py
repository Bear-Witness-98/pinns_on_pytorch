import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from loss_function import pinn_loss

BATCH_SIZE = 10


def train_loop(
    input_data: torch.Tensor, model: nn.Module, optimizer: torch.optim
) -> None:
    # I must do the "dataloader" myself, as all the data is to be stored in RAM
    # i.e. in a torch.Tensor variable.
    num_datapoints = input_data.shape[0]
    permutation = np.random.permutation(num_datapoints)
    input_data_permuted = input_data[permutation]
    batches = torch.split(input_data_permuted, BATCH_SIZE)
    size = len(batches)
    for batch_num, X in enumerate(batches):
        # compute prediction and loss
        pred = model(X)
        target = X / (-2) + 1
        loss = pinn_loss(pred, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 2 == 0:
            loss, current = loss.item(), (batch_num + 1) * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn) -> None:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    """
    return
