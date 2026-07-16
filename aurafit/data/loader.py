from __future__ import annotations
import argparse
from typing import Any, Iterator
import torch
import torch.utils.data

class CPDataLoader(object):
    def __init__(self, opt: argparse.Namespace, dataset: torch.utils.data.Dataset) -> None:
        super(CPDataLoader, self).__init__()
        if opt.shuffle:
            train_sampler: Any = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )
        self.dataset = dataset
        self.data_iter: Iterator[Any] = self.data_loader.__iter__()

    def next_batch(self) -> Any:
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

__all__ = [
    "CPDataLoader",
]