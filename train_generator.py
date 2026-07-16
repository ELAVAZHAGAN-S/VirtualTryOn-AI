from __future__ import annotations
import argparse
from aurafit.configs.generator_options import add_generator_train_options
from aurafit.training.generator_engine import run

def get_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_generator_train_options(parser)
    opt = parser.parse_args()
    return opt


def main() -> None:
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)
    run(opt)

if __name__ == "__main__":
    main()