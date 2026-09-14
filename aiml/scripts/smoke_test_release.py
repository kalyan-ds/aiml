"""
Retinal AI: Release Smoke Test Runner
Runs real inference on Case A (No-DR) and Case B (Referable),
recording raw vectors, normalized vectors, exact probabilities, probability sums,
referable risk, threshold, referral decision, confidence, Grad-CAM, and latency.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import numpy as np
from PIL import Image
import torch

from aiml.src.inference.predictor import RealRetinalAIPredictor
from aiml.src.features.contract_validator import FeatureContractValidator
from aiml.scripts.train_and_calibrate import generate_fundus_sample


def run_smoke_test():
    checkpoint_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    assert os.path.exists(checkpoint_path), f"Missing checkpoint: {checkpoint_path}"
    predictor = RealRetinalAIPredictor(checkpoint_path=checkpoint_path)

    # --- CASE A: Representative No-DR Image ---
    rng_a = np.random.RandomState(101)
    sample_a = generate_fundus_sample("SAMPLE_NO_DR_001", "OD", 0, rng_a)
    raw_vec_a = list(sample_a["clinical_features"])
    norm_vec_a = FeatureContractValidator.normalize_vector(raw_vec_a)

    res_a = predictor.predict(
        image=sample_a["image"],
        clinical_features=raw_vec_a,
        lens_diopter=20.0,
    )
    probs_a = res_a["probabilities"]
    prob_sum_a = sum(probs_a.values())

    print("=== SMOKE TEST CASE A: REPRESENTATIVE NO-DR ===")
    print("image identifier        :", f"{sample_a['patient_id']}_{sample_a['eye']}")
    print("raw 16-D vector         :", [round(x, 2) for x in raw_vec_a])
    print("normalized 16-D vector  :", [round(x, 4) for x in norm_vec_a])
    print("P0..P4                  :", probs_a)
    print("probability sum         :", round(prob_sum_a, 6))
    print("referable risk          :", res_a["referable_risk"])
    print("threshold               :", res_a["threshold"])
    print("is_referable            :", res_a["is_referable"])
    print("confidence              :", res_a["confidence"])
    print("model version           :", res_a["model_version"])
    print("feature contract version:", FeatureContractValidator.FEATURE_CONTRACT_VERSION)
    print("Grad-CAM path           :", res_a["explainability"]["cam_overlay_path"])
    print("latency                 :", res_a["inference_latency_ms"], "ms")

    # --- CASE B: Representative Referable Image ---
    rng_b = np.random.RandomState(202)
    sample_b = generate_fundus_sample("SAMPLE_REFERABLE_002", "OS", 3, rng_b)
    raw_vec_b = list(sample_b["clinical_features"])
    norm_vec_b = FeatureContractValidator.normalize_vector(raw_vec_b)

    res_b = predictor.predict(
        image=sample_b["image"],
        clinical_features=raw_vec_b,
        lens_diopter=20.0,
    )
    probs_b = res_b["probabilities"]
    prob_sum_b = sum(probs_b.values())

    print("\n=== SMOKE TEST CASE B: REPRESENTATIVE REFERABLE ===")
    print("image identifier        :", f"{sample_b['patient_id']}_{sample_b['eye']}")
    print("raw 16-D vector         :", [round(x, 2) for x in raw_vec_b])
    print("normalized 16-D vector  :", [round(x, 4) for x in norm_vec_b])
    print("P0..P4                  :", probs_b)
    print("probability sum         :", round(prob_sum_b, 6))
    print("referable risk          :", res_b["referable_risk"])
    print("threshold               :", res_b["threshold"])
    print("is_referable            :", res_b["is_referable"])
    print("confidence              :", res_b["confidence"])
    print("model version           :", res_b["model_version"])
    print("feature contract version:", FeatureContractValidator.FEATURE_CONTRACT_VERSION)
    print("Grad-CAM path           :", res_b["explainability"]["cam_overlay_path"])
    print("latency                 :", res_b["inference_latency_ms"], "ms")


if __name__ == "__main__":
    run_smoke_test()
