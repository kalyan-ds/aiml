"""
Retinal AI: Real Retinal AI Predictor & Inference Engine
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Backend Compatible Adapter:
  Class: RealRetinalAIPredictor
  Method: predict(image, clinical_features=None, lens_diopter=None)
"""

import os
import time
import uuid
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.src.preprocessing.image_transforms import FundusPreprocessor
from aiml.src.features.contract_validator import FeatureContractValidator
from aiml.src.explainability.gradcam import GradCAMGenerator
from aiml.src.calibration.manager import CalibrationManager

ICDR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}


class RealRetinalAIPredictor:
    """
    Production-grade AI inference engine adapter implementing the exact
    BaseInferenceService contract expected by the FastAPI backend.
    """

    MODEL_VERSION = "resnet50_fusion_v1.0.0"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        heatmap_output_dir: str = "outputs/heatmaps",
        default_threshold: float = 0.35,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.heatmap_output_dir = heatmap_output_dir
        os.makedirs(self.heatmap_output_dir, exist_ok=True)

        # Initialize subcomponents
        self.preprocessor = FundusPreprocessor()
        self.calibration_manager = CalibrationManager(frozen_threshold=default_threshold)

        # Initialize Model
        self.model = RetinalFusionModel(pretrained=False)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            # Running with initialized weights
            self.model.eval()

        self.model.to(self.device)

        # Initialize Grad-CAM Generator
        self.gradcam = GradCAMGenerator(self.model)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads state dict and operational metadata from model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "calibrated_threshold" in checkpoint:
                self.calibration_manager.frozen_threshold = float(checkpoint["calibrated_threshold"])
            if "model_version" in checkpoint:
                self.MODEL_VERSION = str(checkpoint["model_version"])
            if "clinical_normalization_parameters" in checkpoint:
                norm_params = checkpoint["clinical_normalization_parameters"]
                if isinstance(norm_params, dict) and "mean" in norm_params and "std" in norm_params:
                    FeatureContractValidator.set_normalization_parameters(
                        mean=norm_params["mean"],
                        std=norm_params["std"],
                    )
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(
        self,
        image: Union[bytes, Image.Image, np.ndarray],
        clinical_features: Optional[Union[List[float], Dict[str, float]]] = None,
        lens_diopter: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal screening inference.

        Args:
            image: Raw fundus image input.
            clinical_features: Canonical 16-D clinical feature vector or dictionary.
            lens_diopter: Optional condensing lens power (e.g. 20D/28D) for metadata tracking.

        Returns:
            Structured JSON-compatible dictionary strictly matching the result contract.
        """
        start_time = time.perf_counter()

        # Phase 15: CNN-only fallback check
        if clinical_features is None:
            return {
                "status": "error",
                "gradeable": False,
                "error_code": "ERR_MATLAB_UNAVAILABLE",
                "message": (
                    "Clinical feature vector is missing. The 2080-D hybrid architecture "
                    "requires canonical 16-D clinical features from MATLAB/CV. "
                    "Independent CNN-only checkpoint is not active."
                ),
                "processing_mode": "failed",
                "icdr_grade": None,
                "icdr_label": None,
                "probabilities": None,
                "referable_risk": None,
                "threshold": self.calibration_manager.frozen_threshold,
                "is_referable": None,
                "confidence": None,
                "clinical_features": None,
                "explainability": {
                    "cam_status": "skipped",
                    "cam_overlay_path": None,
                },
                "inference_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_version": self.MODEL_VERSION,
            }

        # Step 1: Validate clinical features
        try:
            ordered_features, feature_dict = FeatureContractValidator.validate_and_serialize(
                clinical_features
            )
            norm_features = FeatureContractValidator.normalize_vector(ordered_features)
            clinical_tensor = torch.tensor(
                [norm_features], dtype=torch.float32, device=self.device
            )
        except Exception as e:
            return {
                "status": "error",
                "gradeable": False,
                "error_code": "ERR_INVALID_CLINICAL_FEATURES",
                "message": f"Clinical feature validation failed: {str(e)}",
                "processing_mode": "failed",
                "icdr_grade": None,
                "icdr_label": None,
                "probabilities": None,
                "referable_risk": None,
                "threshold": self.calibration_manager.frozen_threshold,
                "is_referable": None,
                "confidence": None,
                "clinical_features": None,
                "explainability": {
                    "cam_status": "skipped",
                    "cam_overlay_path": None,
                },
                "inference_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_version": self.MODEL_VERSION,
            }

        # Step 2: Validate and preprocess image
        try:
            image_tensor, pil_rgb = self.preprocessor(image)
            image_tensor = image_tensor.to(self.device)
        except Exception as e:
            return {
                "status": "error",
                "gradeable": False,
                "error_code": "ERR_INVALID_IMAGE",
                "message": f"Fundus image preprocessing failed: {str(e)}",
                "processing_mode": "failed",
                "icdr_grade": None,
                "icdr_label": None,
                "probabilities": None,
                "referable_risk": None,
                "threshold": self.calibration_manager.frozen_threshold,
                "is_referable": None,
                "confidence": None,
                "clinical_features": feature_dict,
                "explainability": {
                    "cam_status": "skipped",
                    "cam_overlay_path": None,
                },
                "inference_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_version": self.MODEL_VERSION,
            }

        # Step 3: Forward inference through RetinalFusionModel (2080-D fusion)
        try:
            # Use inference mode for speed, then enable grad for Grad-CAM
            with torch.no_grad():
                logits = self.model(image_tensor, clinical_tensor)  # (1, 5)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()  # (5,)

            # Parse probabilities
            prob_dict = {
                f"P{i}": round(float(probs[i]), 4) for i in range(5)
            }

            icdr_grade = int(np.argmax(probs))
            icdr_label = ICDR_LABELS.get(icdr_grade, "Unknown")

            # Referable Risk: P2 + P3 + P4
            referable_risk = self.calibration_manager.compute_referable_risk(prob_dict)
            is_referable, active_threshold = self.calibration_manager.evaluate_referable(referable_risk)

            # Confidence = max(P0...P4)
            confidence = round(float(np.max(probs)), 4)

        except Exception as e:
            return {
                "status": "error",
                "gradeable": False,
                "error_code": "ERR_INFERENCE_FAILED",
                "message": f"Model forward pass failed: {str(e)}",
                "processing_mode": "failed",
                "icdr_grade": None,
                "icdr_label": None,
                "probabilities": None,
                "referable_risk": None,
                "threshold": self.calibration_manager.frozen_threshold,
                "is_referable": None,
                "confidence": None,
                "clinical_features": feature_dict,
                "explainability": {
                    "cam_status": "skipped",
                    "cam_overlay_path": None,
                },
                "inference_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_version": self.MODEL_VERSION,
            }

        # Step 4: Grad-CAM generation
        cam_status = "success"
        overlay_path = None
        try:
            cam_map = self.gradcam.generate(
                image_tensor=image_tensor,
                clinical_tensor=clinical_tensor,
                target_class=icdr_grade,
            )
            filename = f"cam_{uuid.uuid4().hex[:8]}.jpg"
            overlay_file = os.path.join(self.heatmap_output_dir, filename)
            overlay_path = self.gradcam.save_overlay(
                original_image=pil_rgb,
                cam_map=cam_map,
                output_path=overlay_file,
                alpha=0.45,
            )
        except Exception as e:
            cam_status = f"failed: {str(e)}"
            overlay_path = None

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "success",
            "gradeable": True,
            "error_code": None,
            "processing_mode": "hybrid",
            "icdr_grade": icdr_grade,
            "icdr_label": icdr_label,
            "probabilities": prob_dict,
            "referable_risk": round(referable_risk, 4),
            "threshold": round(active_threshold, 4),
            "is_referable": is_referable,
            "confidence": confidence,
            "clinical_features": feature_dict,
            "explainability": {
                "cam_status": cam_status,
                "cam_overlay_path": overlay_path,
            },
            "inference_latency_ms": round(latency_ms, 2),
            "model_version": self.MODEL_VERSION,
        }
