from typing import Callable

import torch
from torch.nn import functional as F


def get_torch_function(str: str) -> Callable:
    """Get torch function by name"""
    if hasattr(torch, str):
        return getattr(torch, str)
    elif hasattr(F, str):
        return getattr(F, str)
    elif hasattr(torch.special, str):
        return getattr(torch.special, str)
    else:
        raise ValueError(f"Unknown torch function: {str}")
