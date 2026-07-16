from __future__ import annotations
import numpy as np
import torch
from aurafit.constants.labels import LEFT_ARM, RIGHT_ARM, UPPER

def get_clothes_mask(old_label: torch.Tensor) -> torch.Tensor:
    clothes = torch.FloatTensor((old_label.cpu().numpy() == UPPER).astype(np.int))
    return clothes


def changearm(old_label: torch.Tensor) -> torch.Tensor:
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == LEFT_ARM).astype(np.int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == RIGHT_ARM).astype(np.int))
    label = label * (1 - arm1) + arm1 * UPPER
    label = label * (1 - arm2) + arm2 * UPPER
    return label

__all__ = [
    "get_clothes_mask",
    "changearm",
]