"""
Retinal AI: Calibration & Operational Threshold Manager
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Referable DR Definition:
  - ICDR Levels 2, 3, 4 (Moderate NPDR, Severe NPDR, Proliferative DR)
  - referable_risk = P2 + P3 + P4

Threshold Selection Protocol:
  1. Sweep tau in [0.10, 0.90] with step 0.01 on validation set.
  2. For each tau, compute:
     - Sensitivity (TPR) = TP / (TP + FN)
     - Specificity (TNR) = TN / (TN + FP)
  3. Select operating point:
     tau* = argmax_{tau} { Specificity(tau) | Sensitivity(tau) >= 0.900 }
  4. Freeze tau* in checkpoint metadata.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


class CalibrationManager:
    """
    Manages probability calibration, ROC threshold sweeps, and operational point selection.
    """

    DEFAULT_CALIBRATED_THRESHOLD = 0.17  # Frozen empirical threshold from validation ROC sweep
    DEFAULT_THRESHOLD = 0.17

    def __init__(self, frozen_threshold: Optional[float] = None):
        self.frozen_threshold = frozen_threshold if frozen_threshold is not None else self.DEFAULT_THRESHOLD

    @staticmethod
    def compute_referable_risk(probabilities: Dict[str, float]) -> float:
        """
        Calculates referable DR risk score: Score_ref = P2 + P3 + P4.
        """
        p2 = probabilities.get("P2", 0.0)
        p3 = probabilities.get("P3", 0.0)
        p4 = probabilities.get("P4", 0.0)
        return float(np.clip(p2 + p3 + p4, 0.0, 1.0))

    def evaluate_referable(self, referable_risk: float) -> Tuple[bool, float]:
        """
        Evaluates binary referral decision against frozen threshold:
        is_referable = referable_risk >= frozen_threshold
        Returns: (is_referable, threshold)
        """
        is_referable = bool(referable_risk >= self.frozen_threshold)
        return is_referable, self.frozen_threshold

    @classmethod
    def calibrate_threshold(
        cls,
        val_risks: np.ndarray,
        val_labels_referable: np.ndarray,
        target_sensitivity: float = 0.900,
        sweep_start: float = 0.10,
        sweep_end: float = 0.90,
        step: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Performs empirical ROC sweep to find optimal clinical operating threshold.
        Args:
            val_risks: 1D array of referable risk scores (P2+P3+P4) for validation cohort.
            val_labels_referable: 1D array of binary ground-truth labels (1 if ICDR >= 2 else 0).
            target_sensitivity: Minimum clinical sensitivity constraint (0.900 = 90.0%).
            sweep_start, sweep_end, step: Sweep parameters.
        Returns:
            Dictionary containing optimal threshold, metrics, and full sweep records.
        """
        thresholds = np.arange(sweep_start, sweep_end + step / 2.0, step)
        best_threshold = None
        best_specificity = -1.0
        achieved_sensitivity = 0.0

        sweep_results = []

        total_positives = np.sum(val_labels_referable == 1)
        total_negatives = np.sum(val_labels_referable == 0)

        if total_positives == 0 or total_negatives == 0:
            raise ValueError("Validation set must contain both referable and non-referable cases.")

        for tau in thresholds:
            preds = (val_risks >= tau).astype(int)
            tp = np.sum((preds == 1) & (val_labels_referable == 1))
            fn = np.sum((preds == 0) & (val_labels_referable == 1))
            tn = np.sum((preds == 0) & (val_labels_referable == 0))
            fp = np.sum((preds == 1) & (val_labels_referable == 0))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            sweep_results.append({
                "threshold": round(float(tau), 4),
                "sensitivity": round(float(sens), 4),
                "specificity": round(float(spec), 4),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            })

            # Check clinical constraint: sensitivity >= target_sensitivity
            if sens >= target_sensitivity:
                if spec > best_specificity:
                    best_specificity = spec
                    best_threshold = tau
                    achieved_sensitivity = sens

        # Fallback if no threshold meets target sensitivity (e.g. model needs more training)
        if best_threshold is None:
            # Pick threshold that maximizes sensitivity
            max_sens_entry = max(sweep_results, key=lambda x: x["sensitivity"])
            best_threshold = max_sens_entry["threshold"]
            achieved_sensitivity = max_sens_entry["sensitivity"]
            best_specificity = max_sens_entry["specificity"]

        return {
            "optimal_threshold": round(float(best_threshold), 4),
            "achieved_sensitivity": round(float(achieved_sensitivity), 4),
            "achieved_specificity": round(float(best_specificity), 4),
            "target_sensitivity": target_sensitivity,
            "sweep_records": sweep_results,
        }
