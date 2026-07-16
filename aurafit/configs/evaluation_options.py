from __future__ import annotations
import argparse

def add_evaluation_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--evaluation", default="LPIPS")
    parser.add_argument("--predict_dir", default="./result/bg_ver1/output/")
    parser.add_argument("--ground_truth_dir", default="./data/zalando-hd-resize/test/image")
    parser.add_argument("--resolution", type=int, default=1024)

    return parser

__all__ = [
    "add_evaluation_options",
]