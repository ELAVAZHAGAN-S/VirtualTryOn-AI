from __future__ import annotations
import argparse
import os
from aurafit.configs.preprocessing_options import add_norm_const_options
from aurafit.preprocessing.norm_const import run

def get_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_norm_const_options(parser)
    opt = parser.parse_args()
    return opt


def main() -> None:
    opt = get_opt()
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    run(opt)

if __name__ == "__main__":
    main()