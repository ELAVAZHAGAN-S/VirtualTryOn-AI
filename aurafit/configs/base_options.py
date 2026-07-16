from __future__ import annotations
import argparse
from typing import Optional, Sequence

def add_base_options(
    parser: argparse.ArgumentParser,
    *,
    gpu_ids: str = "",
    workers: int = 4,
    batch_size: int = 8,
    batch_size_flags: Sequence[str] = ("-b", "--batch-size"),
    fp16: bool = False,
    include_cuda: bool = True,
    cuda_default: bool = False,
    dataroot: str = "./data/",
    datamode: str = "train",
    data_list: str = "train_pairs.txt",
    fine_width: int = 192,
    fine_height: int = 256,
    tensorboard_dir: str = "tensorboard",
    checkpoint_dir: str = "checkpoints",
    tocg_checkpoint: Optional[str] = "",
    tocg_checkpoint_required: bool = False,
    tocg_checkpoint_help: str = "tocg checkpoint",
    tensorboard_count: int = 100,
    shuffle: bool = False,
    semantic_nc: int = 13,
    semantic_nc_help: Optional[str] = None,
    include_output_nc: bool = True,
    output_nc: int = 13,
    warp_feature: str = "T1",
    out_layer: str = "relu",
    clothmask_composition: str = "warp_grad",
    include_upsample: bool = True,
    upsample: str = "bilinear",
    include_occlusion: bool = True,
    occlusion_help: Optional[str] = "Occlusion handling",
) -> argparse.ArgumentParser:
    parser.add_argument("--gpu_ids", default=gpu_ids)
    parser.add_argument("-j", "--workers", type=int, default=workers)
    parser.add_argument(*batch_size_flags, type=int, default=batch_size)
    parser.add_argument("--fp16", action="store_true", help="use amp")

    if include_cuda:
        parser.add_argument("--cuda", default=cuda_default, help="cuda or cpu")

    parser.add_argument("--dataroot", default=dataroot)
    parser.add_argument("--datamode", default=datamode)
    parser.add_argument("--data_list", default=data_list)
    parser.add_argument("--fine_width", type=int, default=fine_width)
    parser.add_argument("--fine_height", type=int, default=fine_height)

    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=tensorboard_dir,
        help="save tensorboard infos",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=checkpoint_dir,
        help="save checkpoint infos",
    )
    if tocg_checkpoint_required:
        parser.add_argument(
            "--tocg_checkpoint",
            type=str,
            help=tocg_checkpoint_help,
        )
    else:
        parser.add_argument(
            "--tocg_checkpoint",
            type=str,
            default=tocg_checkpoint,
            help=tocg_checkpoint_help,
        )
    parser.add_argument("--tensorboard_count", type=int, default=tensorboard_count)

    parser.add_argument("--shuffle", action="store_true", help="shuffle input data")

    if semantic_nc_help is not None:
        parser.add_argument("--semantic_nc", type=int, default=semantic_nc, help=semantic_nc_help)
    else:
        parser.add_argument("--semantic_nc", type=int, default=semantic_nc)

    if include_output_nc:
        parser.add_argument("--output_nc", type=int, default=output_nc)

    parser.add_argument("--warp_feature", choices=["encoder", "T1"], default=warp_feature)
    parser.add_argument("--out_layer", choices=["relu", "conv"], default=out_layer)
    parser.add_argument(
        "--clothmask_composition",
        type=str,
        choices=["no_composition", "detach", "warp_grad"],
        default=clothmask_composition,
    )

    if include_upsample:
        parser.add_argument(
            "--upsample",
            type=str,
            default=upsample,
            choices=["nearest", "bilinear"],
        )

    if include_occlusion:
        if occlusion_help is not None:
            parser.add_argument("--occlusion", action="store_true", help=occlusion_help)
        else:
            parser.add_argument("--occlusion", action="store_true")

    return parser


__all__ = [
    "add_base_options",
]