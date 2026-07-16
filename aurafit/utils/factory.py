from __future__ import annotations
import argparse
from typing import Any, Type, TypeVar
import torch

T = TypeVar("T")

def create_network(cls: Type[T], opt: argparse.Namespace) -> T:
    net: Any = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net

__all__ = [
    "create_network",
]