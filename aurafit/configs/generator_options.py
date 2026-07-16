from __future__ import annotations
import argparse
from aurafit.configs.base_options import add_base_options

def _add_spade_common_synthesis_options(
    parser: argparse.ArgumentParser,
    *,
    norm_G: str,
    ngf: int,
    init_type: str,
    init_variance: float,
    num_upsampling_layers_help: str,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--norm_G",
        type=str,
        default=norm_G,
        help="instance normalization or batch normalization",
    )
    parser.add_argument("--ngf", type=int, default=ngf, help="# of gen filters in first conv layer")
    parser.add_argument(
        "--init_type",
        type=str,
        default=init_type,
        help="network initialization [normal|xavier|kaiming|orthogonal]",
    )
    parser.add_argument(
        "--init_variance",
        type=float,
        default=init_variance,
        help="variance of the initialization distribution",
    )
    parser.add_argument(
        "--num_upsampling_layers",
        choices=["normal", "more", "most"],
        default="most",
        help=num_upsampling_layers_help,
    )
    return parser

def add_generator_train_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_base_options(
        parser,
        gpu_ids="0",
        workers=4,
        batch_size=8,
        batch_size_flags=("-b", "--batch_size"),
        include_cuda=True,
        cuda_default=False,
        dataroot="./data/",
        datamode="train",
        data_list="train_pairs.txt",
        fine_width=768,
        fine_height=1024,
        tensorboard_dir="tensorboard",
        checkpoint_dir="checkpoints",
        tocg_checkpoint=None,
        tocg_checkpoint_required=False,
        tocg_checkpoint_help="condition generator checkpoint",
        tensorboard_count=100,
        semantic_nc=13,
        semantic_nc_help="# of input label classes without unknown class",
        include_output_nc=False,
        warp_feature="T1",
        out_layer="relu",
        clothmask_composition="warp_grad",
        include_upsample=False,
        include_occlusion=True,
        occlusion_help=None,
    )

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--gen_checkpoint", type=str, default="", help="gen checkpoint")
    parser.add_argument("--dis_checkpoint", type=str, default="", help="dis checkpoint")

    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)

    parser.add_argument("--lpips_count", type=int, default=1000)
    parser.add_argument("--test_datasetting", default="paired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    parser.add_argument("--G_lr", type=float, default=0.0001, help="initial learning rate for adam")
    parser.add_argument("--D_lr", type=float, default=0.0004, help="initial learning rate for adam")
    parser.add_argument("--GMM_const", type=float, default=None, help="constraint for GMM module")

    parser.add_argument(
        "--gen_semantic_nc",
        type=int,
        default=7,
        help="# of input label classes without unknown class",
    )

    _add_spade_common_synthesis_options(
        parser,
        norm_G="spectralaliasinstance",
        ngf=64,
        init_type="xavier",
        init_variance=0.02,
        num_upsampling_layers_help=(
            "If 'more', adds upsampling layer between the two middle resnet blocks. "
            "If 'most', also add one more upsampling + resnet layer at the end of the generator"
        ),
    )
    parser.add_argument(
        "--norm_D",
        type=str,
        default="spectralinstance",
        help="instance normalization or batch normalization",
    )
    parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in first conv layer")

    parser.add_argument(
        "--no_ganFeat_loss",
        action="store_true",
        help="if specified, do *not* use discriminator feature matching loss",
    )
    parser.add_argument(
        "--no_vgg_loss",
        action="store_true",
        help="if specified, do *not* use VGG feature matching loss",
    )
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="weight for feature matching loss")
    parser.add_argument("--lambda_feat", type=float, default=10.0, help="weight for feature matching loss")
    parser.add_argument("--lambda_vgg", type=float, default=10.0, help="weight for vgg loss")

    parser.add_argument("--n_layers_D", type=int, default=3, help="# layers in each discriminator")
    parser.add_argument("--netD_subarch", type=str, default="n_layer", help="architecture of each discriminator")
    parser.add_argument(
        "--num_D",
        type=int,
        default=2,
        help="number of discriminators to be used in multiscale",
    )
    parser.add_argument("--GT", action="store_true")
    parser.add_argument("--num_test_visualize", type=int, default=3)
    return parser

def add_generator_test_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_base_options(
        parser,
        gpu_ids="",
        workers=4,
        batch_size=1,
        batch_size_flags=("-b", "--batch-size"),
        include_cuda=True,
        cuda_default=False,
        dataroot="./data/zalando-hd-resize",
        datamode="test",
        data_list="test_pairs.txt",
        fine_width=768,
        fine_height=1024,
        tensorboard_dir="./data/zalando-hd-resize/tensorboard",
        checkpoint_dir="checkpoints",
        tocg_checkpoint="./eval_models/weights/v0.1/mtviton.pth",
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

    parser.add_argument("--test_name", type=str, default="test", help="test name")

    parser.add_argument("--output_dir", type=str, default="./Output")

    parser.add_argument("--datasetting", default="unpaired")

    parser.add_argument(
        "--gen_checkpoint",
        type=str,
        default="./eval_models/weights/v0.1/gen.pth",
        help="G checkpoint",
    )

    parser.add_argument(
        "--gen_semantic_nc",
        type=int,
        default=7,
        help="# of input label classes without unknown class",
    )

    _add_spade_common_synthesis_options(
        parser,
        norm_G="spectralaliasinstance",
        ngf=64,
        init_type="xavier",
        init_variance=0.02,
        num_upsampling_layers_help="normal: 256, more: 512",
    )

    return parser

__all__ = [
    "add_generator_train_options",
    "add_generator_test_options",
]