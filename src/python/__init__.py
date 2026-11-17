"""Simple package extending PyTorch with the very important "add_pi" operator"""

import torch

import add_pi._native


def add_pi(tensor: torch.Tensor) -> torch.Tensor:
    """
    Highly complicated operator that takes an input tensor ... and adds Pi to it

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to be Pi'd

    Returns
    -------
    torch.Tensor
        The Pi'd tensor
    """
    return torch.ops.add_pi.add_pi(tensor)
