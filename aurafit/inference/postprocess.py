from __future__ import annotations
import torch

def remove_overlap(seg_out: torch.Tensor, warped_cm: torch.Tensor) -> torch.Tensor:
    assert len(warped_cm.shape) == 4
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

__all__ = [
    "remove_overlap",
]