"""
Retinal AI: Release Inference Verification Runner
Executes real inference on representative clinical cases to prove readiness.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import numpy as np
from PIL import Image
from aiml.src.inference.predictor import RealRetinalAIPredictor
from aiml.scripts.train_and_calibrate import generate_fundus_sample


def run_verification():
    print("============================================================")
    print("RETINAL AI: RELEASE VERIFICATION INFERENCE")
    print("============================================================")

    checkpoint_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    predictor = RealRetinalAIPredictor(checkpoint_path=checkpoint_path)

    # --- CASE 1: Representative No-DR Case (ICDR Grade 0) ---
    rng1 = np.random.RandomState(101)
    case_no_dr = generate_fundus_sample("PATIENT_001_OD_NoDR", "OD", 0, rng1)

    res_no_dr = predictor.predict(
        image=case_no_dr["image"],
        clinical_features=case_no_dr["clinical_features"],
        lens_diopter=20.0,
    )

    print("\n--- CASE 1: REPRESENTATIVE NO-DR (ICDR Grade 0) ---")
    print(f"Image Identifier : {case_no_dr['patient_id']}_{case_no_dr['eye']}")
    print(f"16-D Vector      : {case_no_dr['clinical_features']}")
    print(f"Probabilities    : {res_no_dr['probabilities']}")
    print(f"Referable Risk   : {res_no_dr['referable_risk']}")
    print(f"Threshold        : {res_no_dr['threshold']}")
    print(f"Is Referable     : {res_no_dr['is_referable']}")
    print(f"Confidence       : {res_no_dr['confidence']}")
    print(f"Predicted Grade  : {res_no_dr['icdr_grade']} ({res_no_dr['icdr_label']})")
    print(f"Grad-CAM Path    : {res_no_dr['explainability']['cam_overlay_path']}")
    print(f"Model Version    : {res_no_dr['model_version']}")
    print(f"Latency          : {res_no_dr['inference_latency_ms']} ms")

    # --- CASE 2: Representative Referable Case (ICDR Grade 3 Severe NPDR) ---
    rng2 = np.random.RandomState(202)
    case_ref = generate_fundus_sample("PATIENT_002_OS_Referable", "OS", 3, rng2)

    res_ref = predictor.predict(
        image=case_ref["image"],
        clinical_features=case_ref["clinical_features"],
        lens_diopter=20.0,
    )

    print("\n--- CASE 2: REPRESENTATIVE REFERABLE (ICDR Grade 3 Severe NPDR) ---")
    print(f"Image Identifier : {case_ref['patient_id']}_{case_ref['eye']}")
    print(f"16-D Vector      : {case_ref['clinical_features']}")
    print(f"Probabilities    : {res_ref['probabilities']}")
    print(f"Referable Risk   : {res_ref['referable_risk']}")
    print(f"Threshold        : {res_ref['threshold']}")
    print(f"Is Referable     : {res_ref['is_referable']}")
    print(f"Confidence       : {res_ref['confidence']}")
    print(f"Predicted Grade  : {res_ref['icdr_grade']} ({res_ref['icdr_label']})")
    print(f"Grad-CAM Path    : {res_ref['explainability']['cam_overlay_path']}")
    print(f"Model Version    : {res_ref['model_version']}")
    print(f"Latency          : {res_ref['inference_latency_ms']} ms")


if __name__ == "__main__":
    run_verification()
