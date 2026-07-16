from __future__ import annotations
from typing import Sequence, Union
import cv2
import numpy as np
import torch

def gen_noise(shape: Union[Sequence[int], "np._ShapeLike"]) -> torch.Tensor:
    noise = np.zeros(shape, dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise

__all__ = [
    "gen_noise",
]