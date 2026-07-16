from __future__ import annotations
import argparse
import os
from aurafit.configs.generator_options import add_generator_test_options
from aurafit.inference.generator_runner import run

def get_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_generator_test_options(parser)
    opt = parser.parse_args()
    return opt

def main() -> None:
    opt = get_opt()
    print(opt)
    print("Start to test %s!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    run(opt)

if __name__ == "__main__":
    main()