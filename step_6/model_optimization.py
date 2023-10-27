import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# the pipeline is:
# the data
# construct (obtain) the Net
# get the loss function
# get the optimizer (and hyperparameters)
# train, modify and repeat!


class NeuralNetwork(nn.Module):
    def __init__(self):
        # Why the god damn hell is this called this way??
        # this isn't even in the torch tutorial!
        super(
            NeuralNetwork, self
        ).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Full optimization loop implementation.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
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


def main():
    training_data = datasets.FashionMNIST(
        root="../step_2/data", train=True, download=False, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="../step_2/data", train=False, download=False, transform=ToTensor()
    )

    # I thing the dataloader should be later in the code. After defining the
    # batch_size parameter in the code.
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # use cpu as training and testing tensor are un cpu (and fixing it now in
    # a tidy way is pretty time consuming)
    # one more reason the DataLoaders should be used later on.
    model = NeuralNetwork().to("cpu")
    print(model)

    # define training hyperparameters:
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    ## there may be much more hyperparameters here

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n----------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    print("Done!")

    return


# There are lots of things that should be customizable:
# data transformations (to make the changes I want to)
# net architecture (kind of easy to see)
# net Layers (I should be able to create a layer from numpy operations and its derivatives)
# loss functions (to use in different contexts)
# optimizer (in case I want to do bat-crazy shit)

# All this changes should be doable, as to mantain all the code in just
# one framework (PyTorch). But some processing may be better done in other libraries, languages, etc.
# (For example, if getting data through queries, maybe the encoding is better done in SQL, but
# it is very problem dependant).


if __name__ == "__main__":
    main()
