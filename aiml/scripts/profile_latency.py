"""
Retinal AI: Real Inference Latency Benchmark
Profiles actual runtime latency breakdown on host hardware.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import numpy as np
from PIL import Image
import torch

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.src.preprocessing.image_transforms import FundusPreprocessor
from aiml.src.features.contract_validator import FeatureContractValidator
from aiml.src.explainability.gradcam import GradCAMGenerator


def profile():
    print("=== Retinal AI Latency Benchmark ===")
    preprocessor = FundusPreprocessor()
    model = RetinalFusionModel(pretrained=False)
    model.eval()
    gradcam = GradCAMGenerator(model)

    raw_img = Image.new("RGB", (600, 600), (180, 70, 30))
    raw_feats = [1.0] * 16

    # Warmup runs (5 runs)
    print("Running 5 warmup iterations...")
    for _ in range(5):
        img_t, pil_rgb = preprocessor(raw_img)
        clin_t = FeatureContractValidator.to_tensor(raw_feats)
        with torch.no_grad():
            _ = model(img_t, clin_t)
        _ = gradcam.generate(img_t, clin_t, target_class=2)

    print("Running 20 benchmark iterations...")
    t_preprocess = []
    t_resnet_backbone = []
    t_clinical_proj = []
    t_fusion_head = []
    t_gradcam = []
    t_end_to_end = []

    for _ in range(20):
        t0 = time.perf_counter()

        # 1. Preprocessing
        t_p0 = time.perf_counter()
        img_t, pil_rgb = preprocessor(raw_img)
        clin_t = FeatureContractValidator.to_tensor(raw_feats)
        t_p1 = time.perf_counter()
        t_preprocess.append((t_p1 - t_p0) * 1000)

        # 2. ResNet CNN Backbone
        t_c0 = time.perf_counter()
        with torch.no_grad():
            _, v_cnn = model.extract_cnn_features(img_t)
        t_c1 = time.perf_counter()
        t_resnet_backbone.append((t_c1 - t_c0) * 1000)

        # 3. Clinical Projection
        t_cl0 = time.perf_counter()
        with torch.no_grad():
            v_clin = model.clinical_branch(clin_t)
        t_cl1 = time.perf_counter()
        t_clinical_proj.append((t_cl1 - t_cl0) * 1000)

        # 4. Fusion & Head Classification
        t_f0 = time.perf_counter()
        with torch.no_grad():
            z = torch.cat([v_cnn, v_clin], dim=1)
            logits = model.classifier(z)
        t_f1 = time.perf_counter()
        t_fusion_head.append((t_f1 - t_f0) * 1000)

        # 5. Grad-CAM
        t_g0 = time.perf_counter()
        cam_map = gradcam.generate(img_t, clin_t, target_class=2)
        _ = gradcam.create_overlay(pil_rgb, cam_map)
        t_g1 = time.perf_counter()
        t_gradcam.append((t_g1 - t_g0) * 1000)

        t_end = time.perf_counter()
        t_end_to_end.append((t_end - t0) * 1000)

    print("\n--- Benchmark Results (Host CPU: 20 iterations) ---")
    print(f"Preprocessing (RGB -> 512x512 -> Norm) : {np.mean(t_preprocess):.2f} +/- {np.std(t_preprocess):.2f} ms")
    print(f"ResNet-50 CNN Backbone (GAP -> 2048-D) : {np.mean(t_resnet_backbone):.2f} +/- {np.std(t_resnet_backbone):.2f} ms")
    print(f"Clinical Projection (16-D -> 32-D)     : {np.mean(t_clinical_proj):.2f} +/- {np.std(t_clinical_proj):.2f} ms")
    print(f"Multimodal Fusion & Head (2080 -> 5)   : {np.mean(t_fusion_head):.2f} +/- {np.std(t_fusion_head):.2f} ms")
    print(f"Grad-CAM Heatmap + Overlay Blend       : {np.mean(t_gradcam):.2f} +/- {np.std(t_gradcam):.2f} ms")
    print(f"Total End-to-End Latency               : {np.mean(t_end_to_end):.2f} +/- {np.std(t_end_to_end):.2f} ms")


if __name__ == "__main__":
    profile()
