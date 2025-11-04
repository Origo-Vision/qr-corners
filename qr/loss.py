from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self: DiceLoss, eps: float = 1e-6) -> None:
        self.eps = eps

    def forward(
        self: DiceLoss, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        num = 2.0 * (pred * target).sum()
        den = pred.sum() + target.sum() + self.eps

        return 1. - num / den


class MixedLoss(nn.Module):
    def __init__(
        self: MixedLoss, left: nn.Module, right: nn.Module, left_coeff: float = 0.5
    ) -> None:
        self.left = left
        self.right = right
        self.left_coeff = left_coeff
        self.right_coeff = 1.0 - left_coeff

    def forward(
        self: MixedLoss, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.left_coeff * self.left(
            pred, target
        ) + self.right_coeff * self.right(pred, target)
