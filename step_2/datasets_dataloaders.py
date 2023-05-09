import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import ipdb

import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader


def loading_a_dataset():
    # this brings the data somehow to the program

    training_data = datasets.FashionMNIST(
        root="data",  # path where the data is stored
        train=True,  # specifies training or testing dataset
        download=True,  # download data from the internet if it's not available at root
        transform=ToTensor(),  # (or target_transofrmation() ?) specify the feature and label transformations
    )
    test_data = datasets.FashionMNIST(
        root="data",  # path where the data is stored
        train=False,  # specifies training or testing dataset
        download=True,  # download data from the internet if it's not available at root
        transform=ToTensor(),  # (or target_transofrmation() ?) specify the feature and label transformations
    )

    # the transform works like so: The data is stored in whatever format (dataset dependand).
    # may be PIL image, jpg image, .mp3 sound, etc. The transform converts it to whatever format
    # we need. It is "sort of" a function that takes each data point in a specific format, and
    # converts it to a usable data format (e.g. ndarray, tensor, etc.). Usually this function is
    # dataset dependant, as different datasets have different formats. But the ToTensor function
    # is pretty versatile in that sense, so may be used for a wide range of cases.

    # in this case in particualr, with the data loaded, the object is an iterable (with lots of methods) of tuples.
    # each tuple is of the form [tensor, label]. Where tensor contains the data (image's pixel's intensity) and label
    # is the classification for each image. The "description" of each datapoint is later done by hand while iterating.

    return training_data, test_data


def iterating_dataset(training_data):
    labels_map = {
        0: "T-shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneakers",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    # cannot visualize images remotely jeje. Lost some of the magic but guareber.
    plt.show()


# a custom Dataset class must impolement three functions: __init__, __len__ and __getitem__.
# images stored in img_dir (/raw_images)
# labels in a csv file annotations_file (labels.csv)


class CustomImageDataset(Dataset):
    # inits the lables file, the images directory and the transformations to be done (whatever that is) (same as before ?).
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(
            annotations_file
        )  # pandas file retaining name and label
        self.img_dir = img_dir  # image directory
        self.transform = transform  # transformation
        self.target_transform = target_transform  # other transformation?

    # gets the number of items in the dataset.
    def __len__(self):
        return len(self.img_labels)

    # loads and returns sa sample from the dataset at index idx.
    ## based on idx, it identifies the image location in disc
    ## converts to tensor using read_image
    ## retrieves the corresponding label from the csv data in self.img_labels
    ## calls the transform function (if applicable)
    ## return tensor image and corresponding label in tuple
    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_labels.iloc[idx, 0]
        )  # the join just concatenates strings
        image = read_image(img_path)  # reads an image and converts it to a tensor
        label = self.img_labels.iloc[idx, 1]  # gets the label for the given index
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    # depending on the datatype you have, how this Dataset object should behave.
    # transform and target transform should be similar to the dataset retrieval one.
    # functions defined specifically for the dataset you are working with.

    # The Dataset object does not store all the data in RAM. Just when prompted to
    # and index by index (or appropiate slice)


def custom_dataset():
    new_dataset = CustomImageDataset("./labels.csv", "./raw_images/")
    sth = new_dataset[2]
    return


# after having the dataset class already constructed, we may have a DataLoader class, that
# is supposed to get us multiple datapoints at a time, reshuffle the data at every epoch, and
# use python Python's multiprocessing to speed up retrieval.


def preparate_dataloaders(training_data, test_data):
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader


def main():
    print("Here the processing starts")
    print()
    training_data, test_data = loading_a_dataset()
    print(type(training_data))
    iterating_dataset(training_data)
    custom_dataset()
    train_dataloader, test_dataloader = preparate_dataloaders(training_data, test_data)
    train_features, train_labels = next(iter(train_dataloader))  # why the iter() ?
    print(
        f"features type is {type(train_features)}"
    )  # a tensor containing all the batch images (64, 1, 28, 28) == (B, C, H, W)
    print(
        f"Each feature is of type {type(train_features[0])}"
    )  # a tensor containing only one image (1, 28, 28)
    print(f"labels type is {type(train_labels)}")  # idem (label-array tensor)
    print(
        f"each label is of type {type(train_labels[0])}"
    )  # idem (a single-element tensor)
    print(f"Feature batch shape {train_features.size()}")
    print(f"Labels batch shape {train_labels.size()}")

    img = train_features[
        0
    ].squeeze()  # what the fuck is a squeeze ? is it for the tensor shape (redundant dimensions)?
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


if __name__ == "__main__":
    main()
