from __future__ import annotations
import argparse
import torch

def make_grid(N: int, iH: int, iW: int, opt: argparse.Namespace) -> torch.Tensor:
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    if opt.cuda:
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
    return grid

__all__ = [
    "make_grid",
]