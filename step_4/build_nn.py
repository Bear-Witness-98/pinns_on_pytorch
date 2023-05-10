import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# for some strange reason, layers in the PyTorch jargon is equal to Module
# why do you ask? No fucking idea. Just to make things worse to understand codewise
# btw, a whole NN is also a module. A Module may be a concatenation of Layers and/or Modules
# themselves. (Maybe that generality is why the call them Modules??)
# Each available layer is subclass from nn.Module, so yes, the name should be because of that.

# in the CS231n, they distinguish between 3 ways of creating a model with PyTorch, namely:
# 1) Barebones 2) nn.Module API, 3) nn.Sequential API. I will try to determine de differences
# in this tutorial I think the 2 later are more or less reasonable, but the first is yet to
# be understood here.

# There is something I am missing from all the possibilities to connect things with PyTorch. I should
# re-check the Assignment 2's PyTorch notebook for more explanation (or inference from my side, as cs231n is
# not actually really good at explaining things).


class MyNeuralNetwork(nn.Module):
    # here I init the layers that will be used for the model. The weights or other
    # layer things are initialized
    def __init__(self):
        super().__init__()  # calls the constructor of the super class (nn.Module). This is needed for some reason.
        self.flatten = (
            nn.Flatten()
        )  # defines its own layer called "flatten" from the torch.nn method nn.Flatten()
        # The sequentiality here should make sense dimensionwise, and that is defined by hand calculations.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )  # creates its own layer (module) called linear_relu_stack from the torch.nn method nn.Sequential,
        # concatenating other nn.Module layers.

    # Here I define how the layers previously initialized are interconnected. Here is simple, and
    # in the init all these layers may have been in a single Sequential module, but I could have things
    # like:
    #     O -> - O
    #   /         \
    #  O           O
    #   \         /
    #    O -> - O
    # Or even more complex ramifications and thus, more complex layer-weight sharing. It is a blessing
    # that pytorch then cares of all of this stuff in the background
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = MyNeuralNetwork().to(device)
    # apparently both torch.tensor and torch.nn.Modules can be moved to other devices
    # I suppose that when "moving" a module, in reality, you move all the tensor parameters
    # that it has to the other device
    print(model)

    # To use the model, we simply pass it the input data, DONT CALL THE forward METHOD DIRECTLY, as
    # under the hood operations are not correctly performed (maybe if you know what you are doing you can?)
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(
        logits
    )  # notice how we can use (certain?) Layers (nn.Module) directly.
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    # let's break this down to see what the model does under the hood (more or less)
    input_image = torch.rand(3, 28, 28)  # three 28x28 images
    print(input_image.size())

    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())

    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    hidden1 = nn.ReLU()(
        hidden1
    )  # Why are some layers init in a variable, and others don't?
    # I think that the layers that are actually instantiated
    # as variables are done so because I want to keep their state
    # on the code a.k.a. their weights as is, and not regenerate them
    # each time. On some layers this is not necesarry, as they do not have
    # associated weights (for example, the ReLU layer)

    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10),
    )
    input_image = torch.rand(3, 28, 28)
    logits = seq_modules(input_image)

    softmax = nn.Softmax(dim=1)  # sum over the dimension 1 should be equal to 1
    pred_probab = softmax(logits)

    # when I subclass nn.Module to make a NN, it aoutomatically tracks all the parameters
    # in all nn.Module s inside my nn.Module. It also makes all the parameters accessible
    # giving my model the methods: `parameters()` and `named_parameters()`

    print(f"Model structure: {model}")

    for name, param in model.named_parameters():
        # I think the param is a tensor, keeping the parameters of the layer. It only shows the
        # first two "columns" to make it readable here in the console.
        # remember that columns is in a general sense, as these may be general n-dimensional tensors
        print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")

    return


if __name__ == "__main__":
    main()
