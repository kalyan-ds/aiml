"""
Retinal AI: Multi-Class Focal Loss for ICDR Severity Imbalance
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Formula:
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  gamma = 2.0
  alpha_t = N_total / (5 * N_c)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss to mitigate extreme class imbalance between
    Grade 0 (majority ~70%) and severe NPDR / PDR (<5%).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (num_classes,) tensor of class weights
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw unnormalized logits
            targets: (B,) class indices 0..C-1
        """
        # Cross entropy loss without reduction to get log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        p_t = torch.exp(-ce_loss)  # (B,)

        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
