"""
Retinal AI: Comprehensive Test Suite (Contract, Architecture, Preprocessing, Inference, Explainability)
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Tests covering all Phase 16 and Phase 17 requirements:
  1. Valid image input (PIL, numpy, bytes)
  2. Invalid image handling
  3. RGB preprocessing (3 channels, 512x512)
  4. ImageNet normalization
  5. Feature vector length exactly 16
  6. Feature finiteness (NaN/Inf rejection)
  7. Feature ordering & dict-to-vector mapping
  8. 16 -> 32 clinical projection
  9. 2048-D CNN embedding
  10. 2080-D multimodal fusion
  11. 5 output logits
  12. Softmax probability sum ≈ 1.0
  13. Referable risk calculation (P2 + P3 + P4)
  14. Calibrated threshold decision
  15. Confidence = max(P0...P4)
  16. Checkpoint loading and version metadata
  17. Grad-CAM generation (layer4[2] hook, JET overlay, alpha 0.45)
  18. Missing clinical features rejection (ERR_MATLAB_UNAVAILABLE)
  19. Malformed clinical features rejection (ERR_INVALID_CLINICAL_FEATURES)
  20. Backend-compatible dictionary contract verification
"""

import os
import io
import math
import pytest
import numpy as np
from PIL import Image
import torch

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.src.preprocessing.image_transforms import FundusPreprocessor
from aiml.src.features.contract_validator import (
    FeatureContractValidator,
    CANONICAL_FEATURE_KEYS,
    MAX_IMAGE_DIAGONAL_PX,
    DEFAULT_ZERO_EXUDATE_DISTANCE_SENTINEL,
)
from aiml.src.explainability.gradcam import GradCAMGenerator
from aiml.src.calibration.manager import CalibrationManager
from aiml.src.inference.predictor import RealRetinalAIPredictor, ICDR_LABELS
from aiml.evaluation.metrics import compute_comprehensive_metrics


@pytest.fixture
def sample_pil_image():
    """Generates a synthetic 600x600 RGB fundus-like PIL image."""
    arr = np.zeros((600, 600, 3), dtype=np.uint8)
    arr[:, :, 0] = 180  # Strong red channel
    arr[:, :, 1] = 60   # Green channel
    arr[:, :, 2] = 20   # Blue channel
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """Encodes sample image into JPEG bytes."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_numpy_image():
    """Generates a synthetic numpy uint8 image (550, 550, 3)."""
    return np.random.randint(50, 200, size=(550, 550, 3), dtype=np.uint8)


@pytest.fixture
def canonical_16d_list():
    """Returns valid 16-element clinical vector."""
    return [
        8.0,   # F[0]  ma_count_st
        3.0,   # F[1]  ma_count_sn
        14.0,  # F[2]  ma_count_it
        1.0,   # F[3]  ma_count_in
        12.5,  # F[4]  hem_area_st
        0.0,   # F[5]  hem_area_sn
        48.2,  # F[6]  hem_area_it
        4.0,   # F[7]  hem_area_in
        14.2,  # F[8]  vessel_density_st
        11.8,  # F[9]  vessel_density_sn
        16.5,  # F[10] vessel_density_it
        13.1,  # F[11] vessel_density_in
        26.0,  # F[12] total_microaneurysms
        64.7,  # F[13] total_hemorrhage_area_px
        12.0,  # F[14] total_exudates_count
        215.4, # F[15] min_exudate_distance_to_fovea_px
    ]


@pytest.fixture
def canonical_16d_dict(canonical_16d_list):
    """Returns valid dictionary mapping feature keys to values."""
    return {k: canonical_16d_list[i] for i, k in enumerate(CANONICAL_FEATURE_KEYS)}


# ============================================================
# 1. Preprocessing Tests
# ============================================================

def test_preprocessing_valid_pil_and_dimensions(sample_pil_image):
    preprocessor = FundusPreprocessor()
    tensor, pil_rgb = preprocessor(sample_pil_image)

    # Must be 1 x 3 x 512 x 512
    assert tensor.shape == (1, 3, 512, 512), f"Expected (1, 3, 512, 512), got {tensor.shape}"
    assert pil_rgb.size == (512, 512), f"Expected (512, 512), got {pil_rgb.size}"
    assert pil_rgb.mode == "RGB"


def test_preprocessing_bytes_and_numpy(sample_image_bytes, sample_numpy_image):
    preprocessor = FundusPreprocessor()
    t_bytes, _ = preprocessor(sample_image_bytes)
    t_numpy, _ = preprocessor(sample_numpy_image)

    assert t_bytes.shape == (1, 3, 512, 512)
    assert t_numpy.shape == (1, 3, 512, 512)


def test_preprocessing_imagenet_normalization():
    preprocessor = FundusPreprocessor()
    # Create pure white image (255, 255, 255) -> ToTensor is 1.0
    white_img = Image.new("RGB", (512, 512), (255, 255, 255))
    tensor, _ = preprocessor(white_img)

    # (1.0 - mean) / std
    expected_r = (1.0 - 0.485) / 0.229
    expected_g = (1.0 - 0.456) / 0.224
    expected_b = (1.0 - 0.406) / 0.225

    assert math.isclose(tensor[0, 0, 0, 0].item(), expected_r, rel_tol=1e-3)
    assert math.isclose(tensor[0, 1, 0, 0].item(), expected_g, rel_tol=1e-3)
    assert math.isclose(tensor[0, 2, 0, 0].item(), expected_b, rel_tol=1e-3)


def test_preprocessing_invalid_image_rejection():
    preprocessor = FundusPreprocessor()
    with pytest.raises(ValueError):
        preprocessor(b"this is not an image file")


# ============================================================
# 2. Feature Contract Tests
# ============================================================

def test_feature_contract_length_16(canonical_16d_list):
    ordered, feat_dict = FeatureContractValidator.validate_and_serialize(canonical_16d_list)
    assert len(ordered) == 16
    assert len(feat_dict) == 16


def test_feature_contract_rejects_wrong_length():
    with pytest.raises(ValueError, match="must have exactly 16 elements"):
        FeatureContractValidator.validate_and_serialize([1.0] * 15)

    with pytest.raises(ValueError, match="must have exactly 16 elements"):
        FeatureContractValidator.validate_and_serialize([1.0] * 17)


def test_feature_contract_rejects_non_finite(canonical_16d_list):
    # NaN check
    nan_list = list(canonical_16d_list)
    nan_list[5] = float("nan")
    with pytest.raises(ValueError, match="Invalid non-finite value"):
        FeatureContractValidator.validate_and_serialize(nan_list)

    # Inf check
    inf_list = list(canonical_16d_list)
    inf_list[12] = float("inf")
    with pytest.raises(ValueError, match="Invalid non-finite value"):
        FeatureContractValidator.validate_and_serialize(inf_list)


def test_feature_contract_dict_mapping(canonical_16d_dict, canonical_16d_list):
    ordered, feat_dict = FeatureContractValidator.validate_and_serialize(canonical_16d_dict)
    assert ordered == canonical_16d_list
    assert feat_dict == canonical_16d_dict


def test_feature_contract_zero_exudates_handling():
    # If exudates == 0, distance 0.0 should safely default to sentinel
    features = [0.0] * 16
    ordered, feat_dict = FeatureContractValidator.validate_and_serialize(features)
    assert ordered[14] == 0.0
    assert ordered[15] == DEFAULT_ZERO_EXUDATE_DISTANCE_SENTINEL


# ============================================================
# 3. Model Architecture & Dimensionality Tests
# ============================================================

def test_model_latent_dimensions():
    model = RetinalFusionModel(pretrained=False)
    model.eval()

    dummy_image = torch.randn(2, 3, 512, 512)
    dummy_clinical = torch.randn(2, 16)

    # 1. Check CNN embedding extraction
    _, v_cnn = model.extract_cnn_features(dummy_image)
    assert v_cnn.shape == (2, 2048), f"Expected CNN embedding (2, 2048), got {v_cnn.shape}"

    # 2. Check Clinical branch projection (16 -> 32)
    v_clin = model.clinical_branch(dummy_clinical)
    assert v_clin.shape == (2, 32), f"Expected clinical embedding (2, 32), got {v_clin.shape}"

    # 3. Check Forward pass logits (2080 -> 5)
    logits = model(dummy_image, dummy_clinical)
    assert logits.shape == (2, 5), f"Expected output logits (2, 5), got {logits.shape}"


def test_model_rejects_non_16_clinical():
    model = RetinalFusionModel(pretrained=False)
    dummy_image = torch.randn(1, 3, 512, 512)
    invalid_clinical = torch.randn(1, 15)

    with pytest.raises(ValueError, match="dimension mismatch"):
        model(dummy_image, invalid_clinical)


# ============================================================
# 4. Calibration & Referable Logic Tests
# ============================================================

def test_referable_risk_calculation():
    prob_dict = {
        "P0": 0.10,
        "P1": 0.15,
        "P2": 0.40,
        "P3": 0.25,
        "P4": 0.10,
    }
    risk = CalibrationManager.compute_referable_risk(prob_dict)
    # P2 + P3 + P4 = 0.40 + 0.25 + 0.10 = 0.75
    assert math.isclose(risk, 0.75, rel_tol=1e-4)


def test_threshold_decision_logic():
    calib = CalibrationManager(frozen_threshold=0.35)
    is_ref_high, t1 = calib.evaluate_referable(0.50)
    assert is_ref_high is True
    assert t1 == 0.35

    is_ref_low, t2 = calib.evaluate_referable(0.20)
    assert is_ref_low is False
    assert t2 == 0.35


# ============================================================
# 5. Grad-CAM Explainability Tests
# ============================================================

def test_gradcam_hook_and_generation(sample_pil_image, canonical_16d_list):
    model = RetinalFusionModel(pretrained=False)
    gradcam = GradCAMGenerator(model)

    preprocessor = FundusPreprocessor()
    img_t, pil_rgb = preprocessor(sample_pil_image)
    clin_t = FeatureContractValidator.to_tensor(canonical_16d_list)

    cam_map = gradcam.generate(img_t, clin_t, target_class=2)

    assert cam_map.shape == (512, 512)
    assert cam_map.min() >= 0.0
    assert cam_map.max() <= 1.0

    overlay = gradcam.create_overlay(pil_rgb, cam_map, alpha=0.45)
    assert overlay.size == (512, 512)
    assert overlay.mode == "RGB"


# ============================================================
# 6. RealRetinalAIPredictor Integration Tests
# ============================================================

def test_real_predictor_full_pipeline(sample_pil_image, canonical_16d_list):
    ckpt_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    predictor = RealRetinalAIPredictor(checkpoint_path=ckpt_path)

    result = predictor.predict(
        image=sample_pil_image,
        clinical_features=canonical_16d_list,
        lens_diopter=20.0,
    )

    # Verify result contract schema
    assert result["status"] == "success"
    assert result["gradeable"] is True
    assert result["error_code"] is None
    assert result["processing_mode"] == "hybrid"
    assert result["icdr_grade"] in [0, 1, 2, 3, 4]
    assert result["icdr_label"] in ICDR_LABELS.values()

    # Probabilities
    probs = result["probabilities"]
    assert len(probs) == 5
    prob_sum = sum(probs.values())
    assert math.isclose(prob_sum, 1.0, rel_tol=1e-2)

    # Referable risk
    ref_risk = result["referable_risk"]
    expected_risk = probs["P2"] + probs["P3"] + probs["P4"]
    assert math.isclose(ref_risk, expected_risk, rel_tol=1e-3)

    # Threshold & referral
    threshold = result["threshold"]
    assert result["is_referable"] == (ref_risk >= threshold)

    # Confidence = max probability
    assert result["confidence"] == max(probs.values())

    # Features
    assert len(result["clinical_features"]) == 16

    # Explainability
    assert result["explainability"]["cam_status"] == "success"
    assert result["explainability"]["cam_overlay_path"] is not None
    assert os.path.exists(result["explainability"]["cam_overlay_path"])

    # Latency & Version
    assert result["inference_latency_ms"] > 0
    assert result["model_version"] == "resnet50_fusion_v1.0.0"


def test_real_predictor_missing_clinical_rejection(sample_pil_image):
    ckpt_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    predictor = RealRetinalAIPredictor(checkpoint_path=ckpt_path)

    # Missing clinical features (clinical_features=None)
    result = predictor.predict(image=sample_pil_image, clinical_features=None)

    assert result["status"] == "error"
    assert result["gradeable"] is False
    assert result["error_code"] == "ERR_MATLAB_UNAVAILABLE"
    assert "CNN-only checkpoint is not active" in result["message"]


def test_real_predictor_malformed_clinical_rejection(sample_pil_image):
    ckpt_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    predictor = RealRetinalAIPredictor(checkpoint_path=ckpt_path)

    # Malformed features (only 3 elements)
    result = predictor.predict(image=sample_pil_image, clinical_features=[1.0, 2.0, 3.0])

    assert result["status"] == "error"
    assert result["gradeable"] is False
    assert result["error_code"] == "ERR_INVALID_CLINICAL_FEATURES"


def test_real_predictor_invalid_image_rejection(canonical_16d_list):
    ckpt_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    predictor = RealRetinalAIPredictor(checkpoint_path=ckpt_path)

    result = predictor.predict(image=b"corrupted", clinical_features=canonical_16d_list)

    assert result["status"] == "error"
    assert result["gradeable"] is False
    assert result["error_code"] == "ERR_INVALID_IMAGE"


# ============================================================
# 7. Comprehensive Evaluation Metrics Tests
# ============================================================

def test_evaluation_metrics_reporting():
    y_true = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    y_probs = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.05, 0.05, 0.8, 0.05, 0.05],
        [0.02, 0.03, 0.05, 0.8, 0.1],
        [0.01, 0.01, 0.03, 0.05, 0.9],
        [0.75, 0.15, 0.05, 0.03, 0.02],
        [0.05, 0.1, 0.7, 0.1, 0.05],
        [0.02, 0.02, 0.06, 0.1, 0.8],
    ])

    report = compute_comprehensive_metrics(y_true, y_probs, threshold=0.35, cohort_name="test_cohort")

    assert report["overall_accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["referable_triage"]["sensitivity"] == 1.0
    assert report["referable_triage"]["specificity"] == 1.0
    assert len(report["confusion_matrix"]) == 5
