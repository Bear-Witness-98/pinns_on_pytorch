import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


def obtain_dataset():
    ds = datasets.FashionMNIST(
        root="../step_2/data/",
        train=True,
        download=False,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0,
                torch.tensor(y),
                value=1,  # replace all elements across the 0th dimension that are at the positions [torch.tensor(y) with the value 1]
            )
        ),
    )
    return ds
    # ToTensor == converts PIL image or Numpy ndarray into a FloatTensor. Resacels image's intensity values in the range
    # [0,1] -> hay que ver si reescala con el max o el min, o tomando en cuenta el encodeo de la imagen original
    # (no importante para este tutorial)
    # Lambda == applies any user-defined lambda function. Here it transforms an integer into a one hot encoded tensor
    # i.e. label = 3 (from the possible values [0,1,2,3,4,5]) -> f(label) = [0,0,0,1,0,0]


def main():
    ds = obtain_dataset()
    print(ds[0])
    return


if __name__ == "__main__":
    main()
