from __future__ import annotations
import os
from typing import Any, Callable, Iterable
from tensorboardX import SummaryWriter
from tqdm import tqdm

def create_summary_writer(opt: Any) -> SummaryWriter:
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    return board


def training_step_range(opt: Any, extra_steps: int = 0) -> range:
    return range(opt.load_step, opt.keep_step + extra_steps)


def run_training_loop(
    opt: Any,
    step_fn: Callable[[int], None],
    extra_steps: int = 0,
    use_tqdm: bool = True,
) -> None:
    step_range: Iterable[int] = training_step_range(opt, extra_steps=extra_steps)
    if use_tqdm:
        step_range = tqdm(step_range)
    for step in step_range:
        step_fn(step)


def is_display_step(step: int, opt: Any) -> bool:
    return (step + 1) % opt.display_count == 0


def is_save_step(step: int, opt: Any) -> bool:
    return (step + 1) % opt.save_count == 0

__all__ = [
    "create_summary_writer",
    "training_step_range",
    "run_training_loop",
    "is_display_step",
    "is_save_step",
]