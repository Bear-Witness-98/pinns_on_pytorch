import numpy as np
import torch

# generate the data in numpy arrays
T0 = 0
X0 = 1
X_dot0 = 1


def get_initial_condition() -> list[list[float, float], list[float, float]]:
    return {
        "time-positon": (T0, X0),
        "time-speed": (T0, X_dot0),
    }


def get_time_sampling(max_time: float = 2.0, num_samples: int = 50) -> torch.Tensor:
    # need an input of shape (B, 1), being B the Batch size, and 1 is the only input
    # dimension of the Net
    time_samples = torch.tensor(np.sort(np.random.rand(num_samples) * max_time))
    return time_samples.reshape(time_samples.shape[0], 1).type(torch.float)
