"""
Retinal AI: Backend Integration Contract & Schema Compatibility Test
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening

Simulates backend caller invoking RealRetinalAIPredictor.predict(...)
and validates the complete returned dictionary against the backend BaseInferenceService contract.
"""

import os
import math
import pytest
import numpy as np
from PIL import Image

from aiml.src.inference.predictor import RealRetinalAIPredictor, ICDR_LABELS
from aiml.src.features.contract_validator import CANONICAL_FEATURE_KEYS


@pytest.fixture
def trained_predictor():
    checkpoint_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    predictor = RealRetinalAIPredictor(checkpoint_path=checkpoint_path)
    return predictor


@pytest.fixture
def backend_clinical_input():
    # 16 finite numeric values simulating MATLAB opaque array
    return [
        6.0, 4.0, 10.0, 2.0,     # MAs
        25.0, 10.0, 40.0, 15.0,  # Hemorrhages
        14.2, 13.8, 14.5, 13.9,  # Vessel densities
        22.0,                    # Total MAs
        90.0,                    # Total Hemorrhage area
        8.0,                     # Total exudates count
        210.0,                   # Min exudate distance to fovea
    ]


def test_backend_interface_signature(trained_predictor, backend_clinical_input):
    """
    Verifies that predict() accepts exactly (image, clinical_features, lens_diopter)
    without raising TypeError or requiring missing arguments.
    """
    img = Image.new("RGB", (512, 512), (180, 70, 30))
    result = trained_predictor.predict(
        image=img,
        clinical_features=backend_clinical_input,
        lens_diopter=20.0,
    )
    assert isinstance(result, dict)


def test_backend_response_schema_completeness(trained_predictor, backend_clinical_input):
    """
    Validates that every single required field in the backend contract is present,
    has the correct type, and contains non-null values for successful inference.
    """
    img = Image.new("RGB", (512, 512), (190, 80, 25))
    result = trained_predictor.predict(
        image=img,
        clinical_features=backend_clinical_input,
        lens_diopter=20.0,
    )

    required_fields = [
        "status",
        "gradeable",
        "error_code",
        "processing_mode",
        "icdr_grade",
        "icdr_label",
        "probabilities",
        "referable_risk",
        "threshold",
        "is_referable",
        "confidence",
        "clinical_features",
        "explainability",
        "inference_latency_ms",
        "model_version",
    ]

    for field in required_fields:
        assert field in result, f"Missing required backend field: {field}"

    # Status checks
    assert result["status"] == "success"
    assert result["gradeable"] is True
    assert result["error_code"] is None
    assert result["processing_mode"] == "hybrid"

    # ICDR Grade & Label
    assert isinstance(result["icdr_grade"], int)
    assert 0 <= result["icdr_grade"] <= 4
    assert result["icdr_label"] == ICDR_LABELS[result["icdr_grade"]]

    # Probabilities
    probs = result["probabilities"]
    assert isinstance(probs, dict)
    for p_key in ["P0", "P1", "P2", "P3", "P4"]:
        assert p_key in probs
        assert isinstance(probs[p_key], float)
        assert 0.0 <= probs[p_key] <= 1.0
    prob_sum = sum(probs.values())
    assert math.isclose(prob_sum, 1.0, rel_tol=1e-2)

    # Referable Risk = P2 + P3 + P4
    expected_risk = probs["P2"] + probs["P3"] + probs["P4"]
    assert math.isclose(result["referable_risk"], expected_risk, rel_tol=1e-3)

    # Calibrated Threshold
    assert isinstance(result["threshold"], float)
    assert 0.0 < result["threshold"] < 1.0

    # Decision flag
    assert result["is_referable"] == (result["referable_risk"] >= result["threshold"])

    # Confidence = max(P0...P4)
    expected_conf = max(probs.values())
    assert math.isclose(result["confidence"], expected_conf, rel_tol=1e-3)

    # Clinical features returned as dictionary
    assert isinstance(result["clinical_features"], dict)
    assert len(result["clinical_features"]) == 16
    for k in CANONICAL_FEATURE_KEYS:
        assert k in result["clinical_features"]

    # Explainability
    assert result["explainability"]["cam_status"] == "success"
    cam_path = result["explainability"]["cam_overlay_path"]
    assert cam_path is not None
    assert os.path.exists(cam_path)

    # Latency and model version
    assert isinstance(result["inference_latency_ms"], float)
    assert result["inference_latency_ms"] > 0
    assert result["model_version"] == "resnet50_fusion_v1.0.0"


def test_backend_graceful_missing_features_error(trained_predictor):
    """
    Verifies that passing None for clinical_features returns a structured error
    compatible with the backend error schema without crashing.
    """
    img = Image.new("RGB", (512, 512), (180, 70, 30))
    result = trained_predictor.predict(image=img, clinical_features=None)

    assert result["status"] == "error"
    assert result["gradeable"] is False
    assert result["error_code"] == "ERR_MATLAB_UNAVAILABLE"
    assert result["probabilities"] is None
    assert result["referable_risk"] is None
    assert result["is_referable"] is None
