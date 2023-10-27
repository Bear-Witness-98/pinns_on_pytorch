
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn

NUM_SAMPLES = 50
MAX_TIME = 10
R = 1
C = 1
V0 = 0

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


def main():

    # get data
    init_conditions = (0, 0)
    # one batch of data
    T = torch.tensor(np.sort(np.random.rand(NUM_SAMPLES) * MAX_TIME).reshape(NUM_SAMPLES, 1), dtype=torch.float,requires_grad=True)
    # easy dimensions fix (NUM_SAMPLES x 1)
    # apparently, this thing messes up with the gradient function of the tensor
    # the reshaping makes this not work? how wierd
    # needed to make all reshaping and re-typing operations in numpy, previous to tensor convert.
    # T = T.reshape(T.shape[0], 1).type(torch.float)
    print("The shape of the input is: ", end='')
    print(T.shape)

    # define model
    v = PINN().to("cpu")

    # define the optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(v.parameters(), lr=learning_rate)

    loss_accum = 0
    ipdb.set_trace()
    v_T = v(T)
    V = torch.sum(v_T)
    V.backward()
    loss = (T.grad + T + 1) ** 2
    L = torch.mean(loss) + (v(torch.tensor([0.0])) - V0) ** 2
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    optimizer.zero_grad()


    for t in T:
        v_t = v(t)
        v_t.backward()
        print(t.grad)
        loss_t = t.grad + t + 1
        
        
    


    pass


if __name__ == "__main__":
    main()