import torch


def main():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    # we need to compute the gradient of the loss with respect to the
    # parameters w and b

    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    # computing gradients
    loss.backward()  # I think this computes the relevant derivatives (For all the computational graph it is in?)
    print(w.grad)
    print(b.grad)

    # we can only get the grad `property` for the lead nodes of the computational graph, which have the
    # `requires_grad` property set to TRUE. For all other nodes in the graph, the gradients are unavailable.

    # we can only perform gradient calculations using `backward` once on a given graph for performance reasons
    # If we need to do several `backward` calls on the same graph, we need to pass `retrain_graph=True` to the
    # `backward` call

    # we can disable the gradient computation and tracking history if we want to with
    # the following wrapper:

    z = torch.matmul(x, w) + b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)

    # another way to do so is to use the `detach()` method on the tensor
    z = torch.matmul(x, w) + b
    z_det = z.detach()
    print(z_det.requires_grad)

    # All this things performed over the z tensor, could be done over a
    # nn.Model object? Such as a whole NN? Is this passed through to all
    # its tensors?

    # disable this is useful to speed up inference, and to freeze only some
    # parameters of the NN

    # More con computational graphs.

    # Let's check the gradients, and when to zero them.
    x = torch.ones(5, requires_grad=False)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    loss.backward()

    print(f"The gradient of x is: {x.grad}")
    print(f"The gradient of w is: {w.grad}")
    print(f"The gradient of w is: {b.grad}")

    # this explodes if you do not set retain_graph=True in the first backward
    if False:
        loss.backward()

    x = torch.ones(5, requires_grad=False)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    loss.backward(retain_graph=True)
    # it is a little bit unclear to me what happens with the state of the graph
    # when performing some of these operations. In projects or whatever dive into it.

    print(f"The gradients now are:")
    print(f"The gradient of x is: {x.grad}")
    print(f"The gradient of w is: {w.grad}")
    print(f"The gradient of w is: {b.grad}")

    loss.backward(retain_graph=True)

    print(f"Now the gradients should've doubled")
    print(f"The gradient of x is: {x.grad}")
    print(f"The gradient of w is: {w.grad}")
    print(f"The gradient of w is: {b.grad}")

    z = torch.matmul(x, w) + b

    print(f"Now the gradients should stay the same")
    print(f"The gradient of x is: {x.grad}")
    print(f"The gradient of w is: {w.grad}")
    print(f"The gradient of w is: {b.grad}")

    # x.grad.zero_() cannot be done as x has require_grad=False
    w.grad.zero_()
    b.grad.zero_()

    print(f"Now all the grads should be zeroed out")
    print(f"The gradient of x is: {x.grad}")
    print(f"The gradient of w is: {w.grad}")
    print(f"The gradient of w is: {b.grad}")

    return


if __name__ == "__main__":
    main()
