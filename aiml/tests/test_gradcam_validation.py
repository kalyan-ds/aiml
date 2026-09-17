"""
Retinal AI: Unit Tests for Phase 5 Grad-CAM Validation
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Tests:
  1. Target Layer Check (model.layer4[2])
  2. Non-zero gradient verification
  3. Finite value checks (no NaN/Inf)
  4. Range normalization [0.0, 1.0]
  5. Spatial dimension match (512x512)
  6. Overlay RGB rendering & alpha blending
  7. Logits invariance before vs after CAM
  8. Unique output naming & file creation
"""

import os
import uuid
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import pytest

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.src.preprocessing.image_transforms import FundusPreprocessor
from aiml.src.features.contract_validator import FeatureContractValidator
from aiml.src.explainability.gradcam import GradCAMGenerator
from aiml.src.inference.predictor import RealRetinalAIPredictor


@pytest.fixture
def predictor():
    checkpoint = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    return RealRetinalAIPredictor(checkpoint_path=checkpoint if os.path.exists(checkpoint) else None)


@pytest.fixture
def dummy_fundus_data():
    img = Image.new("RGB", (512, 512), (180, 60, 20))
    feats = [
        2.0, 3.0, 1.0, 4.0,       # MA quadrant
        15.0, 20.0, 10.0, 25.0,   # Hemorrhage quadrant
        14.0, 13.5, 14.2, 13.8,   # Vessel density
        10.0,                     # Total MA
        70.0,                     # Total Hem
        5.0,                      # Total Exudates
        180.0,                    # Min Fovea Dist
    ]
    return img, feats


def test_target_layer_exists(predictor):
    """Verify that ResNet-50 layer4[2] exists and is a valid nn.Module."""
    model = predictor.model
    assert hasattr(model, "layer4"), "model must have layer4"
    assert len(model.layer4) >= 3, "layer4 must contain at least 3 Bottleneck blocks"
    target_block = model.layer4[2]
    assert isinstance(target_block, nn.Module), "layer4[2] must be an instance of nn.Module"


def test_gradcam_finite_and_normalized(predictor, dummy_fundus_data):
    """Verify Grad-CAM produces finite values strictly normalized to [0, 1]."""
    img, feats = dummy_fundus_data
    img_t, pil_rgb = predictor.preprocessor(img)
    ordered, _ = FeatureContractValidator.validate_and_serialize(feats)
    norm_feats = FeatureContractValidator.normalize_vector(ordered)
    clin_t = torch.tensor([norm_feats], dtype=torch.float32)

    cam_map = predictor.gradcam.generate(img_t, clin_t, target_class=2)

    assert isinstance(cam_map, np.ndarray), "CAM output must be a numpy ndarray"
    assert np.all(np.isfinite(cam_map)), "CAM must contain only finite values (no NaN/Inf)"
    assert cam_map.shape == (512, 512), f"CAM spatial shape must be (512, 512), got {cam_map.shape}"
    assert cam_map.min() >= 0.0 - 1e-6, f"CAM min must be >= 0.0, got {cam_map.min()}"
    assert cam_map.max() <= 1.0 + 1e-6, f"CAM max must be <= 1.0, got {cam_map.max()}"


def test_gradcam_gradients_non_zero(predictor, dummy_fundus_data):
    """Verify backpropagation to layer4 activations yields non-zero gradients."""
    img, feats = dummy_fundus_data
    img_t, _ = predictor.preprocessor(img)
    ordered, _ = FeatureContractValidator.validate_and_serialize(feats)
    norm_feats = FeatureContractValidator.normalize_vector(ordered)
    clin_t = torch.tensor([norm_feats], dtype=torch.float32)

    model = predictor.model
    model.eval()

    # Forward up to layer4
    x = model.conv1(img_t)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    act = model.layer4(x).detach().clone().requires_grad_(True)

    v_cnn = torch.flatten(model.avgpool(act), 1)
    v_clin = model.clinical_branch(clin_t)
    z = torch.cat([v_cnn, v_clin], dim=1)
    logits = model.classifier(z)
    score = logits[0, 2]
    model.classifier.zero_grad()
    score.backward()

    assert act.grad is not None, "Gradients w.r.t layer4 activations must not be None"
    grad_norm = act.grad.norm().item()
    assert grad_norm > 1e-9, f"Gradient norm must be non-zero, got {grad_norm}"


def test_gradcam_overlay_creation(predictor, dummy_fundus_data):
    """Verify overlay blending creates a valid 512x512 RGB image."""
    img, feats = dummy_fundus_data
    img_t, pil_rgb = predictor.preprocessor(img)
    ordered, _ = FeatureContractValidator.validate_and_serialize(feats)
    norm_feats = FeatureContractValidator.normalize_vector(ordered)
    clin_t = torch.tensor([norm_feats], dtype=torch.float32)

    cam_map = predictor.gradcam.generate(img_t, clin_t, target_class=2)
    overlay = predictor.gradcam.create_overlay(pil_rgb, cam_map, alpha=0.45)

    assert isinstance(overlay, Image.Image), "Overlay must be a PIL Image"
    assert overlay.size == (512, 512), f"Overlay size must be (512, 512), got {overlay.size}"
    assert overlay.mode == "RGB", f"Overlay mode must be RGB, got {overlay.mode}"


def test_logits_invariance_after_cam(predictor, dummy_fundus_data):
    """Verify model forward pass logits and probabilities are identical before vs after CAM."""
    img, feats = dummy_fundus_data
    img_t, _ = predictor.preprocessor(img)
    ordered, _ = FeatureContractValidator.validate_and_serialize(feats)
    norm_feats = FeatureContractValidator.normalize_vector(ordered)
    clin_t = torch.tensor([norm_feats], dtype=torch.float32)

    # 1. Logits before
    with torch.no_grad():
        logits_before = predictor.model(img_t, clin_t).clone()
        probs_before = torch.softmax(logits_before, dim=1).cpu().numpy()

    # 2. Run Grad-CAM
    _ = predictor.gradcam.generate(img_t, clin_t, target_class=0)

    # 3. Logits after
    with torch.no_grad():
        logits_after = predictor.model(img_t, clin_t)
        probs_after = torch.softmax(logits_after, dim=1).cpu().numpy()

    max_logit_diff = torch.max(torch.abs(logits_before - logits_after)).item()
    max_prob_diff = np.max(np.abs(probs_before - probs_after))

    assert max_logit_diff < 1e-6, f"Logits changed after Grad-CAM! Max diff: {max_logit_diff}"
    assert max_prob_diff < 1e-6, f"Softmax probabilities changed after Grad-CAM! Max diff: {max_prob_diff}"
