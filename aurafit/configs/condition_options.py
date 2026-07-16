from __future__ import annotations
import argparse
from aurafit.configs.base_options import add_base_options

def _add_condition_discriminator_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--Ddownx2",
        action="store_true",
        help="Downsample D's input to increase the receptive field",
    )
    parser.add_argument("--Ddropout", action="store_true", help="Apply dropout to D")
    parser.add_argument("--num_D", type=int, default=2, help="Generator ngf")
    parser.add_argument("--spectral", action="store_true", help="Apply spectral normalization to D")
    return parser

def add_condition_train_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_base_options(
        parser,
        gpu_ids="",
        workers=4,
        batch_size=8,
        batch_size_flags=("-b", "--batch-size"),
        include_cuda=True,
        cuda_default=False,
        dataroot="./data/",
        datamode="train",
        data_list="train_pairs.txt",
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
        include_upsample=True,
        upsample="bilinear",
        include_occlusion=True,
        occlusion_help="Occlusion handling",
    )

    parser.add_argument("--name", default="test")
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)

    _add_condition_discriminator_options(parser)
    parser.add_argument("--G_D_seperate", action="store_true")
    parser.add_argument("--no_GAN_loss", action="store_true")
    parser.add_argument("--lasttvonly", action="store_true")
    parser.add_argument("--interflowloss", action="store_true", help="Intermediate flow loss")
    parser.add_argument(
        "--edgeawaretv",
        type=str,
        choices=["no_edge", "last_only", "weighted"],
        default="no_edge",
        help="Edge aware TV loss",
    )
    parser.add_argument("--add_lasttv", action="store_true")

    parser.add_argument("--no_test_visualize", action="store_true")
    parser.add_argument("--num_test_visualize", type=int, default=3)
    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    parser.add_argument("--G_lr", type=float, default=0.0002, help="Generator initial learning rate for adam")
    parser.add_argument("--D_lr", type=float, default=0.0002, help="Discriminator initial learning rate for adam")
    parser.add_argument("--CElamda", type=float, default=10, help="initial learning rate for adam")
    parser.add_argument("--GANlambda", type=float, default=1)
    parser.add_argument("--tvlambda", type=float, default=2)
    parser.add_argument("--val_count", type=int, default="1000")

    return parser


def add_condition_test_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_base_options(
        parser,
        gpu_ids="",
        workers=4,
        batch_size=8,
        batch_size_flags=("-b", "--batch-size"),
        include_cuda=False,
        dataroot="./data/zalando-hd-resize",
        datamode="test",
        data_list="test_pairs.txt",
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
        include_upsample=True,
        upsample="bilinear",
        include_occlusion=True,
        occlusion_help="Occlusion handling",
    )

    parser.add_argument("--datasetting", default="paired")
    parser.add_argument("--D_checkpoint", type=str, default="", help="D checkpoint")
    _add_condition_discriminator_options(parser)
    parser.add_argument("--norm_const", type=float, help="Normalizing constant for rejection sampling")

    return parser

__all__ = [
    "add_condition_train_options",
    "add_condition_test_options",
]