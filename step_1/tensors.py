import torch
import numpy

def tensor_init():
    data = [[1,2],[3,4]]
    print(data)

    x_data = torch.tensor(data)
    print(x_data)
    

def main():
    print('Here the processing starts')

    tensor_init()




if __name__ == '__main__':
    main()