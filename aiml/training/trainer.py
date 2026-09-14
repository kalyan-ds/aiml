"""
Retinal AI: Training Pipeline & Optimizer
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Pipeline Protocol:
  - ResNet-50 + 16-D Clinical Projection -> 2080-D Fusion
  - Loss: MultiClassFocalLoss (gamma = 2.0, inverse class frequency weights)
  - Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-2)
  - Scheduler: CosineAnnealingLR
  - Master Seed: 42
"""

import random
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.training.focal_loss import MultiClassFocalLoss
from aiml.src.calibration.manager import CalibrationManager
from aiml.training.checkpoint_builder import create_checkpoint, save_checkpoint


def set_deterministic_seed(seed: int = 42):
    """
    Sets global deterministic seeds across PyTorch, NumPy, and Python random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RetinalTrainer:
    """
    Coordinates training, validation, and calibration for RetinalFusionModel.
    """

    def __init__(
        self,
        model: RetinalFusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        num_epochs: int = 20,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        set_deterministic_seed(seed)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        # Loss function with focal gamma=2.0 and class weighting
        self.criterion = MultiClassFocalLoss(gamma=2.0, alpha=class_weights)

        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0

        for images, clinical_feats, labels in self.train_loader:
            images = images.to(self.device)
            clinical_feats = clinical_feats.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images, clinical_feats)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        self.scheduler.step()
        return total_loss / max(1, batches)

    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, clinical_feats, labels in self.val_loader:
                images = images.to(self.device)
                clinical_feats = clinical_feats.to(self.device)

                logits = self.model(images, clinical_feats)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(labels.numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)

        # Referable risk: P2 + P3 + P4
        val_risks = np.sum(all_probs[:, 2:], axis=1)
        val_ref_labels = (all_labels >= 2).astype(int)

        # Perform threshold sweep
        sweep_result = CalibrationManager.calibrate_threshold(
            val_risks=val_risks,
            val_labels_referable=val_ref_labels,
            target_sensitivity=0.900,
        )

        return {
            "all_probs": all_probs,
            "all_labels": all_labels,
            "optimal_threshold": sweep_result["optimal_threshold"],
            "sensitivity": sweep_result["achieved_sensitivity"],
            "specificity": sweep_result["achieved_specificity"],
        }
