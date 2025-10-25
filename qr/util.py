import random

import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Set the random seed.

    Parameters:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)