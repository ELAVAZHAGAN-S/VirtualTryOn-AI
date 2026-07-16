from __future__ import annotations
import argparse
from aurafit.configs.preprocessing_options import add_parse_agnostic_options
from aurafit.preprocessing.parse_agnostic import run

def get_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_parse_agnostic_options(parser)
    opt = parser.parse_args()
    return opt

def main() -> None:
    opt = get_opt()
    run(opt)

if __name__ == "__main__":
    main()