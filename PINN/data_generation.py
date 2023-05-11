import numpy as np

# generate the data in numpy arrays
T0 = 0
X0 = 1
X_dot0 = 1

def initial_conditions():
    return [[T0, X0], [T0, X_dot0]]

def time_values(max_time:float=2.0, num_samples:int=50) -> np.ndarray:
    return np.random.rand(num_samples)*max_time
