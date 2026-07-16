from __future__ import annotations
import argparse
import os
from aurafit.configs.condition_options import add_condition_train_options
from aurafit.training.condition_engine import run

def get_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_condition_train_options(parser)
    opt = parser.parse_args()
    return opt

def main() -> None:
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    run(opt)

if __name__ == "__main__":
    main()