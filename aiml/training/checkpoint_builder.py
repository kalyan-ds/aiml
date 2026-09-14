"""
Retinal AI: Model Checkpoint Builder & Serializer
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Enforces standard metadata schema for full reproducibility and auditability.
"""

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.src.preprocessing.image_transforms import FundusPreprocessor
from aiml.src.features.contract_validator import FeatureContractValidator
from aiml.src.inference.predictor import ICDR_LABELS


def create_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    calibrated_threshold: float = 0.17,
    metrics: Optional[Dict[str, Any]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Assembles a fully specified audit-ready model checkpoint package.
    """
    checkpoint = {
        "model_version": RetinalFusionModel.ARCHITECTURE_VERSION,
        "feature_contract_version": FeatureContractValidator.FEATURE_CONTRACT_VERSION,
        "preprocessing_version": FundusPreprocessor.PREPROCESSING_VERSION,
        "calibrated_threshold": float(calibrated_threshold),
        "seed": int(seed),
        "epoch": int(epoch),
        "class_mapping": ICDR_LABELS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
        "dataset_metadata": dataset_metadata or {
            "source": "APTOS-2019-Blindness-Detection / Messidor-2",
            "split_strategy": "Patient-level Stratified 70/15/15",
            "image_size": [512, 512],
            "channels": "RGB",
        },
        "normalization_config": {
            "image_mean": FundusPreprocessor.IMAGENET_MEAN,
            "image_std": FundusPreprocessor.IMAGENET_STD,
            "clinical_dim": FeatureContractValidator.FEATURE_DIM,
            "fusion_dim": RetinalFusionModel.FUSION_DIM,
        },
    }
    return checkpoint


def save_checkpoint(
    checkpoint: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Saves the checkpoint dictionary to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(checkpoint, output_path)
    return os.path.abspath(output_path)
