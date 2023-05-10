import torch
import torchvision.models as models


def main():
    # how are models prepared for varying size input?

    # take a model, downlaod it and save it
    model = models.vgg16(weights="IMAGENET1K_V1")
    torch.save(model.state_dict(), "model_weights.pth")

    # to load a mode, I need to create an instance of the same model first
    # a.k.a., the same architecture

    model = models.vgg16()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    # model should be changed to eval mode (the line above) to set batchnorm and
    # batchnormalization layers to evaluation mode. This means that the models in general
    # can be in two modes, evaluation and training.

    # when loading the model wights, we need to instantiate the odel class first, as we need
    # first to have the structure before the weights.

    # we might want to also save the structure of the model, this is done by saving the
    # model instead of just the model.state_dict()

    torch.save(model, "model.pth")

    # to load the model

    model1 = torch.load("model.pth")
    print(model1)
    # this relies on python's pickle module for serializing the model. Thus, it relies on
    # the actual class definition to be available when loading the model

    return


if __name__ == "__main__":
    main()
