from __future__ import annotations
import torch
from aurafit.visualization.segmap import pred_to_onehot

def iou_metric(y_pred_batch: torch.Tensor, y_true_batch: torch.Tensor) -> torch.Tensor:
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        y_pred = y_pred > 0.5

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)

        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou

def cal_miou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    size = prediction.shape
    target = target.cpu()
    prediction = pred_to_onehot(prediction.detach().cpu())
    list = [1, 2, 3, 4, 5, 6, 7, 8]
    union = 0
    intersection = 0
    for b in range(size[0]):
        for c in list:
            intersection += torch.logical_and(target[b, c], prediction[b, c]).sum()
            union += torch.logical_or(target[b, c], prediction[b, c]).sum()
    return intersection.item() / union.item()

__all__ = [
    "iou_metric",
    "cal_miou",
]