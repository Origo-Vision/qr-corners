import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_loss(x: torch.Tensor) -> torch.Tensor:
    # (codes, tags)
    assert x.shape[1] == 4

    # Get the reference/mean for each one of the codes.
    reference = x.mean(dim=1)

    # Calculate the Lpull, based on mean square distance between the reference for each code and the code's tags.
    # Intuition: Within a code, the assigned tags shall be as close as possible ("pulled together").
    Lpull = ((x - reference.view(-1, 1)) ** 2).mean()

    # Calculate the Lpush, based on the absolute distances between the reference values.
    # Intuition: The individual codes should have values assigned far from each other ("pushed apart").
    ns, ms = zip(*itertools.combinations(range(len(reference)), 2))
    Lpush = torch.clip(1.0 - (reference[ns,] - reference[ms,]).abs(), 0.0, 1.0).mean()

    return Lpull + Lpush


def main() -> None:
    params = torch.rand(5, 4, requires_grad=True)

    print(f"initial parameters=\n{F.sigmoid(params)}")

    epochs = 3000
    lr = 1
    for epoch in range(epochs):
        loss = group_loss(F.sigmoid(params))

        params.grad = None
        loss.backward()
        params.data -= params.grad * lr

        print(f"Epoch={epoch+1:3d}/{epochs:3d}. Loss={loss.item():.8f}")

    print(f"final parameters=\n{F.sigmoid(params)}")


if __name__ == "__main__":
    torch.manual_seed(1598)
    main()
