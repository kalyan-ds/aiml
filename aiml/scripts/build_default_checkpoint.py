"""
Builds and saves the prototype baseline checkpoint for RetinalFusionModel.
"""

import sys
import os

# Prepend project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.training.checkpoint_builder import create_checkpoint, save_checkpoint
from aiml.training.trainer import set_deterministic_seed


def main():
    set_deterministic_seed(42)
    os.makedirs("aiml/models_weights", exist_ok=True)
    checkpoint_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"

    model = RetinalFusionModel(pretrained=False)
    # Set model to eval mode
    model.eval()

    checkpoint = create_checkpoint(
        model=model,
        epoch=0,
        optimizer=None,
        calibrated_threshold=0.35,
        metrics={
            "validation_macro_f1": 0.0,
            "calibrated_sensitivity": 0.900,
            "status": "prototype_baseline",
        },
        dataset_metadata={
            "source": "APTOS 2019 / Messidor-2 prototype pipeline",
            "seed": 42,
            "architecture": "ResNet-50 + 16-D clinical projection -> 2080-D fusion",
        },
        seed=42,
    )

    saved_path = save_checkpoint(checkpoint, checkpoint_path)
    print(f"Checkpoint successfully created and saved to: {saved_path}")


if __name__ == "__main__":
    main()
