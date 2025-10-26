import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """
    Set the random seed.

    Parameters:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(mod: nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Parameters:
        mod: The model.

    Returns:
        The parameter count.
    """
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def find_device(force_cpu: bool) -> torch.device:
    """
    Find the best the device for the execution.

    Parameters:
        force_cpu: If true, the device will be forced to CPU.

    Returns:
        The device.
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
