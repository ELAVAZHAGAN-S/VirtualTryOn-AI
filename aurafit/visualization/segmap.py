from __future__ import annotations
import os
from typing import Iterable, List, Union
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def ndim_tensor2im(
    image_tensor: torch.Tensor,
    imtype: type = np.uint8,
    batch: int = 0,
) -> np.ndarray:
    image_numpy = image_tensor[batch].cpu().float().numpy()
    result = np.argmax(image_numpy, axis=0)
    return result.astype(imtype)

def visualize_segmap(
    input: torch.Tensor,
    multi_channel: bool = True,
    tensor_out: bool = True,
    batch: int = 0,
) -> Union[torch.Tensor, Image.Image]:
    palette = [
        0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51,
        254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85,
        85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220,
        0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0
    ]
    input = input.detach()
    if multi_channel:
        input = ndim_tensor2im(input, batch=batch)
    else:
        input = input[batch][0].cpu()
        input = np.asarray(input)
        input = input.astype(np.uint8)
    input = Image.fromarray(input, 'P')
    input.putpalette(palette)

    if tensor_out:
        trans = transforms.ToTensor()
        return trans(input.convert('RGB'))
    return input

def pred_to_onehot(prediction: torch.Tensor) -> torch.Tensor:
    size = prediction.shape
    prediction_max = torch.argmax(prediction, dim=1)
    oneHot_size = (size[0], 13, size[2], size[3])
    pred_onehot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    pred_onehot = pred_onehot.scatter_(1, prediction_max.unsqueeze(1).data.long(), 1.0)
    return pred_onehot


def save_images(
    img_tensors: Iterable[torch.Tensor],
    img_names: Iterable[str],
    save_dir: str,
) -> None:
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')

__all__: List[str] = [
    "ndim_tensor2im",
    "visualize_segmap",
    "pred_to_onehot",
    "save_images",
]