from __future__ import annotations
import os
from collections import OrderedDict
from typing import Any
import torch
import torch.nn as nn

def save_checkpoint(model: nn.Module, save_path: str, opt: Any) -> None:
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    if opt.cuda:
        model.cuda()


def load_checkpoint(model: nn.Module, checkpoint_path: str, opt: Any) -> None:
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    if opt.cuda:
        model.cuda()


def load_checkpoint_G(model: nn.Module, checkpoint_path: str, opt: Any) -> None:
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda:
        model.cuda()


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_G",
]