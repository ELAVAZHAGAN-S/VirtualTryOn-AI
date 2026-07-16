from __future__ import annotations
import argparse
from aurafit.configs.evaluation_options import add_evaluation_options
from aurafit.evaluation.runner import run

def get_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_evaluation_options(parser)
    opt = parser.parse_args()
    return opt


def main() -> None:
    opt = get_opt()
    run(opt)

if __name__ == '__main__':
    main()