"""
Retinal AI: Comprehensive Evaluation Protocol (Multi-Class & Referable Triage)
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Metrics Reported:
  1. Overall Multi-Class Accuracy
  2. Macro F1 Score
  3. Per-Class Precision & Recall (ICDR Grades 0..4)
  4. 5x5 Confusion Matrix
  5. Multi-Class One-vs-Rest ROC-AUC
  6. Referable DR Sensitivity (TPR) at operating threshold tau*
  7. Referable DR Specificity (TNR) at operating threshold tau*
  8. Referable DR Binary ROC-AUC
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

from aiml.src.inference.predictor import ICDR_LABELS


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.17,
    cohort_name: str = "held_out_test",
) -> Dict[str, Any]:
    """
    Evaluates multi-class and binary referable performance against ground truth.

    Args:
        y_true: Ground truth ICDR labels (N,) with integers 0..4
        y_probs: Predicted probability distributions (N, 5)
        threshold: Calibrated operating threshold tau* for referable risk
        cohort_name: Identification string ('validation', 'held_out_test', etc.)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = np.argmax(y_probs, axis=1)

    # 1. Multi-class metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    per_class_metrics = {}
    for c in range(5):
        per_class_metrics[f"grade_{c}_{ICDR_LABELS[c]}"] = {
            "precision": round(float(prec[c]), 4),
            "recall": round(float(rec[c]), 4),
            "f1_score": round(float(f1[c]), 4),
            "support": int(support[c]),
        }

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    # 3. Multi-class OvR ROC-AUC
    try:
        # Check if all classes present in y_true
        present_classes = np.unique(y_true)
        if len(present_classes) == 5:
            ovr_roc_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr"))
        else:
            ovr_roc_auc = None
    except Exception:
        ovr_roc_auc = None

    # 4. Binary Referable DR Metrics (Referable = Grades 2, 3, 4)
    # Ground truth: 1 if y_true >= 2 else 0
    ref_true = (y_true >= 2).astype(int)
    # Predicted risk = P2 + P3 + P4
    ref_risk = np.sum(y_probs[:, 2:], axis=1)
    ref_pred = (ref_risk >= threshold).astype(int)

    tp = int(np.sum((ref_pred == 1) & (ref_true == 1)))
    fn = int(np.sum((ref_pred == 0) & (ref_true == 1)))
    tn = int(np.sum((ref_pred == 0) & (ref_true == 0)))
    fp = int(np.sum((ref_pred == 1) & (ref_true == 0)))

    ref_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ref_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        if len(np.unique(ref_true)) == 2:
            ref_roc_auc = float(roc_auc_score(ref_true, ref_risk))
        else:
            ref_roc_auc = None
    except Exception:
        ref_roc_auc = None

    return {
        "cohort": cohort_name,
        "total_samples": len(y_true),
        "overall_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "ovr_roc_auc": round(ovr_roc_auc, 4) if ovr_roc_auc is not None else None,
        "referable_triage": {
            "applied_threshold": round(float(threshold), 4),
            "sensitivity": round(float(ref_sensitivity), 4),
            "specificity": round(float(ref_specificity), 4),
            "roc_auc": round(float(ref_roc_auc), 4) if ref_roc_auc is not None else None,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }
