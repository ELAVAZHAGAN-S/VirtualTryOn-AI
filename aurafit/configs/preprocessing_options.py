from __future__ import annotations
import argparse
from aurafit.configs.base_options import add_base_options

def add_norm_const_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_base_options(
        parser,
        gpu_ids="",
        workers=4,
        batch_size=8,
        batch_size_flags=("-b", "--batch-size"),
        include_cuda=False,
        dataroot="./data",
        datamode="train",
        data_list="train_pairs_zalando.txt",
        fine_width=192,
        fine_height=256,
        tensorboard_dir="tensorboard",
        checkpoint_dir="checkpoints",
        tocg_checkpoint="",
        tocg_checkpoint_required=False,
        tocg_checkpoint_help="tocg checkpoint",
        tensorboard_count=100,
        semantic_nc=13,
        semantic_nc_help=None,
        include_output_nc=True,
        output_nc=13,
        warp_feature="T1",
        out_layer="relu",
        clothmask_composition="warp_grad",
        include_upsample=False,
        include_occlusion=False,
    )

    parser.add_argument("--name", default="test")

    parser.add_argument("--D_checkpoint", type=str, default="", help="tocg checkpoint")

    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)

    parser.add_argument(
        "--Ddownx2",
        action="store_true",
        help="Downsample D's input to increase the receptive field",
    )
    parser.add_argument("--Ddropout", action="store_true", help="Apply dropout to D")
    parser.add_argument("--num_D", type=int, default=2, help="Generator ngf")
    parser.add_argument("--spectral", action="store_true", help="Apply spectral normalization to D")

    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="./data/zalando-hd-resize")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    return parser

def add_parse_agnostic_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data_path", type=str, help="dataset dir")
    parser.add_argument("--output_path", type=str, help="output dir")
    return parser

__all__ = [
    "add_norm_const_options",
    "add_parse_agnostic_options",
]