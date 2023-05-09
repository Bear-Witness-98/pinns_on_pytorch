import torch
import numpy as np


def tensor_init():
    data = [[1, 2], [3, 4]]
    print("Original data:")
    print(data)
    print()

    # from list
    x_data = torch.tensor(data)
    print("Directly tensor data")
    print(x_data)
    print()

    # from numpy array
    np_array = np.array(data)
    x_np = torch.tensor(np_array)
    print("Tensor data via numpy")
    print(x_np)
    print()

    # datatype is inferred

    # can initialize tensor from another tensor
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor \n {x_ones} \n")

    x_rand = torch.rand_like(
        x_data, dtype=torch.float
    )  # retains properties of x_data but overrides the dtype
    print(f"Random Tensor \n {x_rand} \n")

    # Other useful initialitions
    shape = (
        2,
        3,
    )
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor \n {rand_tensor} \n")
    print(f"Ones Tensor \n {ones_tensor} \n")
    print(f"Zeros Tensor \n {zeros_tensor} \n")
    # it appears that the tensors are initialize with dtype float[sth]


def tensor_attr():
    # these attributes may also be its own datatypes (torch specific dtypes/objects)
    tensor = torch.rand(3, 4)
    print(f"The shape of the tensor is: {tensor.shape}")
    print(f"The datatype of the tensor is: {tensor.dtype}")
    print(f"The device of the tensor is: {tensor.device}")


def tensor_ops():
    # lots and lots of tensor operations (arithmetic, array-like(indexing, slicing), sampling, etc. )
    # further described in pytorch.org/docs/stable/torch.html

    # operations typically run better on GPU (because of parallelization reasons)

    # move to gpu
    tensor = torch.rand(3, 4)
    if torch.cuda.is_available():
        tensor = tensor.to("cuda")
    print(f"The tensor is now stored in {tensor.device}")

    # indexing and slicing works just as in numpy
    tensor = torch.rand(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:,0]}")
    print(f"Last column: {tensor[...,-1]}")
    tensor[0, 0] = 0
    print(tensor)

    # joining tensors
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    # arithmetic operations
    ## matrix multiplication, multiple ways. But may leave different states
    y1 = tensor @ tensor.T

    y2 = tensor.matmul(tensor.T)

    y3 = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)

    ## element-wise multiplication
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(z1)
    torch.mul(tensor, tensor, out=z3)

    # single-element tensors
    agg = tensor.sum() # tensor with one element
    agg_item = agg.item() # convert to python numerical value
    print(agg_item, type(agg_item))

    # in-place operations
    ## These operations are denoted by a "_" suffix. They will change the variable used
    print(f"{tensor}\n")
    tensor.add_(5)
    print(tensor)

def tensor_numpy_bridge():
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    # a change in the tensor is reflected in the ndarray
    t.add_(5)
    print(f"t: {t}")
    print(f"n: {n}")

    # the other way works similar
    n = np.ones(5)
    t = torch.from_numpy(n) # this shares memory, different than the first init from np array

    # changes in array, reflects in tensor
    np.add(n,1,out=n)
    print(f"t: {t}") # this prints a little bit different than the others (why?)
    print(f"n: {n}")



def main():
    print("Here the processing starts")
    print()
    tensor_init()
    tensor_attr()
    tensor_ops()
    tensor_numpy_bridge()


if __name__ == "__main__":
    main()
