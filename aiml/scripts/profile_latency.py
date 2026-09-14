"""
Retinal AI: Real Inference Latency Benchmark
Profiles actual runtime latency breakdown on host hardware across all 7 required dimensions.
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
    checkpoint_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    model = RetinalFusionModel(pretrained=False)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gradcam = GradCAMGenerator(model)

    raw_img = Image.new("RGB", (512, 512), (180, 70, 30))
    raw_feats = [
        6.0, 4.0, 10.0, 2.0,
        25.0, 10.0, 40.0, 15.0,
        14.2, 13.8, 14.5, 13.9,
        22.0, 90.0, 8.0, 210.0,
    ]

    # Warmup runs (5 runs)
    print("Running 5 warmup iterations...")
    for _ in range(5):
        img_t, pil_rgb = preprocessor(raw_img)
        clin_t = FeatureContractValidator.to_tensor(raw_feats)
        with torch.no_grad():
            _ = model(img_t, clin_t)
        _ = gradcam.generate(img_t, clin_t, target_class=2)

    print("Running 25 benchmark iterations...")
    t_preprocess = []
    t_resnet_backbone = []
    t_clinical_proj = []
    t_fusion_head = []
    t_gradcam = []
    t_without_gradcam = []
    t_with_gradcam = []

    for _ in range(25):
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

        # 3. Clinical Branch Projection
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
            probs = torch.softmax(logits, dim=1)
        t_f1 = time.perf_counter()
        t_fusion_head.append((t_f1 - t_f0) * 1000)

        t_without_g = time.perf_counter()
        t_without_gradcam.append((t_without_g - t0) * 1000)

        # 5. Grad-CAM
        t_g0 = time.perf_counter()
        cam_map = gradcam.generate(img_t, clin_t, target_class=2)
        _ = gradcam.create_overlay(pil_rgb, cam_map)
        t_g1 = time.perf_counter()
        t_gradcam.append((t_g1 - t_g0) * 1000)

        t_end = time.perf_counter()
        t_with_gradcam.append((t_end - t0) * 1000)

    print("\n--- Benchmark Results (Host CPU: 25 iterations) ---")
    print(f"1. Preprocessing (RGB -> 512x512 -> Norm) : {np.mean(t_preprocess):.2f} +/- {np.std(t_preprocess):.2f} ms")
    print(f"2. CNN Backbone (ResNet-50 GAP -> 2048-D) : {np.mean(t_resnet_backbone):.2f} +/- {np.std(t_resnet_backbone):.2f} ms")
    print(f"3. Clinical Branch (16-D -> 32-D)         : {np.mean(t_clinical_proj):.2f} +/- {np.std(t_clinical_proj):.2f} ms")
    print(f"4. Fusion & Head (2080-D -> 5 logits)     : {np.mean(t_fusion_head):.2f} +/- {np.std(t_fusion_head):.2f} ms")
    print(f"5. Grad-CAM (Hook + JET LUT Overlay)      : {np.mean(t_gradcam):.2f} +/- {np.std(t_gradcam):.2f} ms")
    print(f"6. Total WITHOUT Grad-CAM                 : {np.mean(t_without_gradcam):.2f} +/- {np.std(t_without_gradcam):.2f} ms")
    print(f"7. Total WITH Grad-CAM                    : {np.mean(t_with_gradcam):.2f} +/- {np.std(t_with_gradcam):.2f} ms")


if __name__ == "__main__":
    profile()
