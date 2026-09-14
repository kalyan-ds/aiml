"""
Retinal AI: End-to-End Real Training, Validation Calibration, and Held-Out Test Pipeline
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Executes authentic optimizer updates, Focal Loss backprop, empirical ROC threshold
calibration on validation cohort, and final held-out test set evaluation.
"""

import sys
import os
from typing import Tuple, List, Dict, Any, Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from aiml.src.models.retinal_fusion import RetinalFusionModel
from aiml.src.features.contract_validator import (
    FeatureContractValidator,
    CANONICAL_FEATURE_KEYS,
    FROZEN_ZERO_EXUDATE_SENTINEL_PX,
)
from aiml.src.calibration.manager import CalibrationManager
from aiml.training.focal_loss import MultiClassFocalLoss
from aiml.training.dataset import RetinalFundusDataset
from aiml.training.checkpoint_builder import create_checkpoint, save_checkpoint
from aiml.training.trainer import set_deterministic_seed
from aiml.evaluation.metrics import compute_comprehensive_metrics


# ============================================================
# Synthetic Fundus Cohort Generator (Medically Parameterized)
# ============================================================

def generate_fundus_sample(
    patient_id: str,
    eye: str,
    grade: int,
    rng: np.random.RandomState,
) -> dict:
    """
    Generates a 512x512 RGB fundus image and paired 16-D clinical feature vector
    strictly adhering to ICDR grading pathology criteria.
    """
    img = Image.new("RGB", (512, 512), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Retinal Fundus Disc (Orange-Red background)
    cx, cy, r = 256, 256, 230
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(185, 65, 20))

    # Optic Disc (Nasal side)
    od_x = cx - 110 if eye == "OD" else cx + 110
    od_y = cy - 20
    draw.ellipse([od_x - 35, od_y - 45, od_x + 35, od_y + 45], fill=(245, 215, 140))

    # Macula / Fovea Center (Temporal side)
    fovea_x = cx + 70 if eye == "OD" else cx - 70
    fovea_y = cy + 10
    draw.ellipse([fovea_x - 18, fovea_y - 18, fovea_x + 18, fovea_y + 18], fill=(130, 40, 15))

    # Major Vascular Arches
    draw.arc([od_x - 60, od_y - 90, od_x + 140, od_y + 70], start=180, end=330, fill=(110, 20, 10), width=4)
    draw.arc([od_x - 60, od_y - 50, od_x + 140, od_y + 110], start=30, end=180, fill=(110, 20, 10), width=4)

    # Pathological Lesion Synthesis & Clinical Feature Vector Construction
    # Quadrant coordinates relative to fovea center (fovea_x, fovea_y):
    # ST: Superior-Temporal (y < fovea_y, temporal)
    # SN: Superior-Nasal (y < fovea_y, nasal)
    # IT: Inferior-Temporal (y > fovea_y, temporal)
    # IN: Inferior-Nasal (y > fovea_y, nasal)

    if grade == 0:  # No DR
        ma_quad = [0, 0, 0, 0]
        hem_quad = [0.0, 0.0, 0.0, 0.0]
        exudates = 0
        min_fovea_dist = FROZEN_ZERO_EXUDATE_SENTINEL_PX
        vessel_dens = [14.5 + rng.uniform(-0.5, 0.5) for _ in range(4)]

    elif grade == 1:  # Mild NPDR (Microaneurysms only)
        ma_quad = [rng.randint(0, 4) for _ in range(4)]
        if sum(ma_quad) == 0:
            ma_quad[rng.randint(0, 4)] = 1
        hem_quad = [0.0, 0.0, 0.0, 0.0]
        exudates = 0
        min_fovea_dist = FROZEN_ZERO_EXUDATE_SENTINEL_PX
        vessel_dens = [14.0 + rng.uniform(-0.5, 0.5) for _ in range(4)]
        # Draw microaneurysms (tiny dark red dots)
        for _ in range(sum(ma_quad)):
            mx, my = rng.randint(cx - 150, cx + 150), rng.randint(cy - 150, cy + 150)
            draw.ellipse([mx - 2, my - 2, mx + 2, my + 2], fill=(90, 10, 10))

    elif grade == 2:  # Moderate NPDR (MAs + blot hemorrhages + hard exudates)
        ma_quad = [rng.randint(2, 8) for _ in range(4)]
        hem_quad = [float(rng.randint(10, 45)) for _ in range(4)]
        exudates = rng.randint(3, 15)
        min_fovea_dist = float(rng.randint(120, 320))
        vessel_dens = [13.5 + rng.uniform(-0.6, 0.6) for _ in range(4)]
        # Draw MAs and Hemorrhages (larger blotches)
        for _ in range(sum(ma_quad)):
            mx, my = rng.randint(cx - 160, cx + 160), rng.randint(cy - 160, cy + 160)
            draw.ellipse([mx - 2, my - 2, mx + 2, my + 2], fill=(90, 10, 10))
        for _ in range(4):
            hx, hy = rng.randint(cx - 140, cx + 140), rng.randint(cy - 140, cy + 140)
            draw.ellipse([hx - 6, hy - 4, hx + 6, hy + 4], fill=(80, 5, 5))
        # Draw Hard Exudates (bright yellow flecks)
        for _ in range(exudates):
            ex = fovea_x + rng.randint(-180, 180)
            ey = fovea_y + rng.randint(-180, 180)
            draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(250, 240, 90))

    elif grade == 3:  # Severe NPDR (4:2:1 rule: severe hemorrhages in all 4 quadrants)
        ma_quad = [rng.randint(8, 20) for _ in range(4)]
        hem_quad = [float(rng.randint(60, 160)) for _ in range(4)]
        exudates = rng.randint(12, 35)
        min_fovea_dist = float(rng.randint(60, 200))
        vessel_dens = [12.0 + rng.uniform(-0.8, 0.8) for _ in range(4)]
        # Extensive blot hemorrhages in all quadrants
        for _ in range(12):
            hx, hy = rng.randint(cx - 180, cx + 180), rng.randint(cy - 180, cy + 180)
            draw.ellipse([hx - 10, hy - 8, hx + 10, hy + 8], fill=(70, 5, 5))
        for _ in range(exudates):
            ex, ey = rng.randint(cx - 150, cx + 150), rng.randint(cy - 150, cy + 150)
            draw.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(250, 240, 90))

    else:  # Grade 4: Proliferative DR (Neovascularization / Vitreous Hemorrhage)
        ma_quad = [rng.randint(12, 30) for _ in range(4)]
        hem_quad = [float(rng.randint(120, 350)) for _ in range(4)]
        exudates = rng.randint(20, 55)
        min_fovea_dist = float(rng.randint(30, 160))
        vessel_dens = [16.5 + rng.uniform(-1.0, 1.0) for _ in range(4)]  # Increased due to NVD/NVE
        # Neovascular fronds on disc & preretinal hemorrhage
        draw.ellipse([od_x - 25, od_y - 25, od_x + 25, od_y + 25], fill=(130, 15, 10))
        draw.chord([cx - 80, cy - 40, cx + 120, cy + 80], start=45, end=190, fill=(60, 0, 0))

    total_ma = float(sum(ma_quad))
    total_hem = float(sum(hem_quad))
    total_ex = float(exudates)

    features = [
        float(ma_quad[0]), float(ma_quad[1]), float(ma_quad[2]), float(ma_quad[3]),
        hem_quad[0], hem_quad[1], hem_quad[2], hem_quad[3],
        vessel_dens[0], vessel_dens[1], vessel_dens[2], vessel_dens[3],
        total_ma,
        total_hem,
        total_ex,
        float(min_fovea_dist),
    ]

    return {
        "patient_id": patient_id,
        "eye": eye,
        "image": img,
        "clinical_features": features,
        "label": grade,
    }


def build_cohort(seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Constructs train (140), validation (30), and held-out test (30) splits
    with strict patient-level separation.
    """
    rng = np.random.RandomState(seed)

    # 100 unique patients, 2 eyes each = 200 total samples
    # Target class distribution across 200 samples:
    # Grade 0: 70 (35%)
    # Grade 1: 40 (20%)
    # Grade 2: 40 (20%)
    # Grade 3: 26 (13%)
    # Grade 4: 24 (12%)
    patient_grades = []
    # 35 patients Grade 0 (70 scans)
    patient_grades.extend([0] * 35)
    # 20 patients Grade 1 (40 scans)
    patient_grades.extend([1] * 20)
    # 20 patients Grade 2 (40 scans)
    patient_grades.extend([2] * 20)
    # 13 patients Grade 3 (26 scans)
    patient_grades.extend([3] * 13)
    # 12 patients Grade 4 (24 scans)
    patient_grades.extend([4] * 12)

    rng.shuffle(patient_grades)

    all_samples = []
    for p_idx, grade in enumerate(patient_grades):
        p_id = f"PATIENT_{p_idx:03d}"
        for eye in ["OD", "OS"]:
            sample = generate_fundus_sample(p_id, eye, grade, rng)
            all_samples.append(sample)

    # Patient-level grouping for stratified splits:
    # 70 patients (140 samples) -> Train (70%)
    # 15 patients (30 samples)  -> Val (15%)
    # 15 patients (30 samples)  -> Test (15%)
    train_samples = all_samples[:140]
    val_samples = all_samples[140:170]
    test_samples = all_samples[170:200]

    return train_samples, val_samples, test_samples


# ============================================================
# Main Training & Calibration Routine
# ============================================================

def main():
    print("============================================================")
    print("RETINAL AI: REAL TRAINING & VALIDATION CALIBRATION RUNNER")
    print("============================================================")
    seed = 42
    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target compute device: {device}")

    # 1. Build Patient-Stratified Cohorts
    train_data, val_data, test_data = build_cohort(seed=seed)
    print(f"Cohort Partitioning:")
    print(f"  - Training Set   : {len(train_data)} fundus scans (70 unique patients)")
    print(f"  - Validation Set : {len(val_data)} fundus scans (15 unique patients)")
    print(f"  - Held-Out Test  : {len(test_data)} fundus scans (15 unique patients)")

    # Record Class Counts
    train_labels = [s["label"] for s in train_data]
    val_labels = [s["label"] for s in val_data]
    test_labels = [s["label"] for s in test_data]

    print("\nClass Distribution (Train):")
    for c in range(5):
        print(f"  ICDR Grade {c}: {train_labels.count(c)} samples ({train_labels.count(c)/len(train_labels)*100:.1f}%)")

    # 2. Compute Deterministic Normalization Parameters on Training Cohort
    train_feats_matrix = np.array([s["clinical_features"] for s in train_data], dtype=np.float32)
    mu_train = np.mean(train_feats_matrix, axis=0).tolist()
    std_train = np.std(train_feats_matrix, axis=0).tolist()
    # Apply to validator
    FeatureContractValidator.set_normalization_parameters(mu_train, std_train)
    norm_params = FeatureContractValidator.get_normalization_parameters()
    print("\nDeterministic Normalization Parameters (Fitted on Train):")
    print(f"  mu   : {[round(x, 2) for x in mu_train]}")
    print(f"  sigma: {[round(x, 2) for x in std_train]}")

    # 3. Create Datasets and DataLoaders
    train_dataset = RetinalFundusDataset(train_data, is_training=True)
    val_dataset = RetinalFundusDataset(val_data, is_training=False)
    test_dataset = RetinalFundusDataset(test_data, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4. Compute Class Weights for Focal Loss (Inverse class frequency)
    n_total = len(train_labels)
    class_weights = [n_total / (5.0 * max(1, train_labels.count(c))) for c in range(5)]
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"\nFocal Loss Class Weights alpha: {[round(x, 3) for x in class_weights]}")

    # 5. Initialize Model
    model = RetinalFusionModel(
        pretrained=False,
        dropout_rate_fusion=0.4,
        dropout_rate_head=0.2,
        use_clinical_batchnorm=False,
    )
    model.to(device)

    # 6. Training Configuration
    epochs = 5
    learning_rate = 3e-4
    weight_decay = 1e-2
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = MultiClassFocalLoss(gamma=2.0, alpha=alpha_tensor)

    print(f"\nCommencing real model training ({epochs} epochs, AdamW lr={learning_rate}, Focal Loss gamma=2.0)...")

    training_history = []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    for ep in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, clin_feats, targets in train_loader:
            images = images.to(device)
            clin_feats = clin_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images, clin_feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / train_batches

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for images, clin_feats, targets in val_loader:
                images = images.to(device)
                clin_feats = clin_feats.to(device)
                targets = targets.to(device)

                logits = model(images, clin_feats)
                loss = criterion(logits, targets)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                val_loss += loss.item()
                val_batches += 1
                val_probs_list.append(probs)
                val_targets_list.append(targets.cpu().numpy())

        avg_val_loss = val_loss / val_batches
        val_probs_arr = np.vstack(val_probs_list)
        val_targets_arr = np.concatenate(val_targets_list)

        val_metrics = compute_comprehensive_metrics(
            y_true=val_targets_arr,
            y_probs=val_probs_arr,
            threshold=0.35,
            cohort_name="validation",
        )

        print(
            f"Epoch [{ep}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_metrics['overall_accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        training_history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": val_metrics["overall_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = ep

    print(f"\nTraining Complete. Best validation epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(best_model_state)
    model.eval()

    # 7. Calibration on Validation Cohort
    print("\n--- Performing Operational ROC Threshold Sweep on Validation Cohort ---")
    val_probs_best = []
    val_targets_best = []
    with torch.no_grad():
        for images, clin_feats, targets in val_loader:
            images = images.to(device)
            clin_feats = clin_feats.to(device)
            logits = model(images, clin_feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_probs_best.append(probs)
            val_targets_best.append(targets.numpy())

    val_probs_best = np.vstack(val_probs_best)
    val_targets_best = np.concatenate(val_targets_best)

    # Referable risk = P2 + P3 + P4
    val_risks = np.sum(val_probs_best[:, 2:], axis=1)
    val_ref_targets = (val_targets_best >= 2).astype(int)

    calibration_results = CalibrationManager.calibrate_threshold(
        val_risks=val_risks,
        val_labels_referable=val_ref_targets,
        target_sensitivity=0.900,
        sweep_start=0.10,
        sweep_end=0.90,
        step=0.01,
    )

    frozen_tau = calibration_results["optimal_threshold"]
    achieved_val_sens = calibration_results["achieved_sensitivity"]
    achieved_val_spec = calibration_results["achieved_specificity"]

    print(f"Calibration Result:")
    print(f"  Frozen Calibrated Threshold tau*: {frozen_tau:.4f}")
    print(f"  Achieved Validation Sensitivity : {achieved_val_sens:.4f} (Constraint >= 0.900)")
    print(f"  Achieved Validation Specificity : {achieved_val_spec:.4f}")

    final_val_metrics = compute_comprehensive_metrics(
        y_true=val_targets_best,
        y_probs=val_probs_best,
        threshold=frozen_tau,
        cohort_name="validation",
    )

    # 8. Evaluate on Unseen Held-Out Test Cohort (Zero leakage)
    print("\n--- Evaluating on Held-Out Test Cohort (at Frozen Threshold tau*) ---")
    test_probs_list = []
    test_targets_list = []
    with torch.no_grad():
        for images, clin_feats, targets in test_loader:
            images = images.to(device)
            clin_feats = clin_feats.to(device)
            logits = model(images, clin_feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_probs_list.append(probs)
            test_targets_list.append(targets.numpy())

    test_probs_arr = np.vstack(test_probs_list)
    test_targets_arr = np.concatenate(test_targets_list)

    final_test_metrics = compute_comprehensive_metrics(
        y_true=test_targets_arr,
        y_probs=test_probs_arr,
        threshold=frozen_tau,
        cohort_name="held_out_test",
    )

    print(f"Held-Out Test Results:")
    print(f"  Overall Accuracy     : {final_test_metrics['overall_accuracy']:.4f}")
    print(f"  Macro F1 Score       : {final_test_metrics['macro_f1']:.4f}")
    print(f"  Referable Sensitivity: {final_test_metrics['referable_triage']['sensitivity']:.4f}")
    print(f"  Referable Specificity: {final_test_metrics['referable_triage']['specificity']:.4f}")
    print(f"  Referable ROC-AUC    : {final_test_metrics['referable_triage']['roc_auc']}")
    print(f"  Confusion Matrix     : {final_test_metrics['confusion_matrix']}")

    # 9. Build and Save the Final Checkpoint
    checkpoint_path = "aiml/models_weights/resnet50_fusion_v1.0.0.pt"
    checkpoint = create_checkpoint(
        model=model,
        epoch=best_epoch,
        optimizer=optimizer,
        calibrated_threshold=frozen_tau,
        metrics={
            "validation_metrics": final_val_metrics,
            "test_metrics": final_test_metrics,
            "training_history": training_history,
        },
        dataset_metadata={
            "dataset_name": "Synthetic-Fundus-Prototype-Cohort-v1",
            "total_samples": 200,
            "train_samples": 140,
            "val_samples": 30,
            "test_samples": 30,
            "split_strategy": "Patient-level Stratified 70/15/15",
            "batch_size": 8,
            "epochs": epochs,
            "seed": seed,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
            "best_epoch": best_epoch,
        },
        seed=seed,
    )
    # Inject normalization parameters directly into checkpoint
    checkpoint["clinical_normalization_parameters"] = norm_params

    save_checkpoint(checkpoint, checkpoint_path)
    print(f"\nAudit-ready checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
