"""
Retinal AI: 16-D Clinical Feature Contract Validator & Normalizer
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Canonical 16-Element Feature Vector Ordering:
  F[0]  = ma_count_st                     (float32, [0, inf), count)
  F[1]  = ma_count_sn                     (float32, [0, inf), count)
  F[2]  = ma_count_it                     (float32, [0, inf), count)
  F[3]  = ma_count_in                     (float32, [0, inf), count)
  F[4]  = hem_area_st                     (float32, [0, inf) px^2)
  F[5]  = hem_area_sn                     (float32, [0, inf) px^2)
  F[6]  = hem_area_it                     (float32, [0, inf) px^2)
  F[7]  = hem_area_in                     (float32, [0, inf) px^2)
  F[8]  = vessel_density_st               (float32, [0.0, 100.0] %)
  F[9]  = vessel_density_sn               (float32, [0.0, 100.0] %)
  F[10] = vessel_density_it               (float32, [0.0, 100.0] %)
  F[11] = vessel_density_in               (float32, [0.0, 100.0] %)
  F[12] = total_microaneurysms            (float32, [0, inf), count)
  F[13] = total_hemorrhage_area_px        (float32, [0, inf) px^2)
  F[14] = total_exudates_count            (float32, [0, inf), count)
  F[15] = min_exudate_distance_to_fovea_px(float32, [0, inf) px)
"""

import math
from typing import List, Dict, Union, Any, Tuple, Optional
import numpy as np
import torch


CANONICAL_FEATURE_KEYS = [
    "ma_count_st",
    "ma_count_sn",
    "ma_count_it",
    "ma_count_in",
    "hem_area_st",
    "hem_area_sn",
    "hem_area_it",
    "hem_area_in",
    "vessel_density_st",
    "vessel_density_sn",
    "vessel_density_it",
    "vessel_density_in",
    "total_microaneurysms",
    "total_hemorrhage_area_px",
    "total_exudates_count",
    "min_exudate_distance_to_fovea_px",
]

# Max diagonal of 512x512 fundus image: sqrt(512^2 + 512^2) ≈ 724.08 px
MAX_IMAGE_DIAGONAL_PX = 724.08
# Safe sentinel representation for distance when zero exudates exist (max distance)
DEFAULT_ZERO_EXUDATE_DISTANCE_SENTINEL = 724.08


class FeatureContractValidator:
    """
    Validates, serializes, and transforms the 16-D clinical feature vector
    in strict adherence to the project contract.
    """

    FEATURE_CONTRACT_VERSION = "feature_v1.0"
    FEATURE_DIM = 16

    @classmethod
    def validate_and_serialize(
        cls,
        features: Union[List[float], Dict[str, float], np.ndarray, torch.Tensor],
        handle_zero_exudates: bool = True,
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Validates the input clinical features and produces both:
          1. Canonical ordered List[float] of length 16
          2. Standardized Dict[str, float] keyed by feature names

        Raises:
            ValueError: If dimensionality is incorrect, values are non-finite (NaN/Inf),
                        or negative where prohibited.
            TypeError: If input is not a supported sequence or dictionary.
        """
        if features is None:
            raise ValueError("Clinical features cannot be None for hybrid multimodal inference.")

        # Convert dict to ordered list if necessary
        if isinstance(features, dict):
            missing_keys = [k for k in CANONICAL_FEATURE_KEYS if k not in features]
            if missing_keys:
                raise ValueError(
                    f"Clinical feature dictionary is missing required keys: {missing_keys}"
                )
            ordered_vals = [float(features[k]) for k in CANONICAL_FEATURE_KEYS]
        elif isinstance(features, (list, tuple)):
            if len(features) != cls.FEATURE_DIM:
                raise ValueError(
                    f"Clinical feature vector must have exactly {cls.FEATURE_DIM} elements. "
                    f"Received {len(features)} elements."
                )
            try:
                ordered_vals = [float(x) for x in features]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Clinical features must be numeric: {e}")
        elif isinstance(features, (np.ndarray, torch.Tensor)):
            flat = features.cpu().numpy().flatten() if isinstance(features, torch.Tensor) else features.flatten()
            if flat.shape[0] != cls.FEATURE_DIM:
                raise ValueError(
                    f"Clinical feature array must have exactly {cls.FEATURE_DIM} elements. "
                    f"Received shape {flat.shape}."
                )
            ordered_vals = [float(x) for x in flat]
        else:
            raise TypeError(
                f"Unsupported clinical feature type: {type(features)}. "
                f"Expected List[float], Dict[str, float], or 1D ndarray/tensor."
            )

        # Validate finiteness
        for i, val in enumerate(ordered_vals):
            if math.isnan(val) or math.isinf(val):
                key = CANONICAL_FEATURE_KEYS[i]
                raise ValueError(
                    f"Invalid non-finite value at index {i} ({key}): {val}. "
                    f"Features must be finite numbers."
                )

        # Validate non-negativity
        for i, val in enumerate(ordered_vals):
            if val < 0.0:
                key = CANONICAL_FEATURE_KEYS[i]
                raise ValueError(
                    f"Negative value not permitted at index {i} ({key}): {val}."
                )

        # Handle zero-exudate edge case in F[15]
        # F[14] is total_exudates_count, F[15] is min_exudate_distance_to_fovea_px
        total_exudates = ordered_vals[14]
        if handle_zero_exudates and total_exudates == 0.0:
            # When no exudates exist, distance to fovea should not be 0.0 (which mimics foveal center involvement)
            # Default to image diagonal boundary if 0.0 was supplied
            if ordered_vals[15] == 0.0:
                ordered_vals[15] = DEFAULT_ZERO_EXUDATE_DISTANCE_SENTINEL

        # Build clean dictionary
        feature_dict = {
            k: float(ordered_vals[i]) for i, k in enumerate(CANONICAL_FEATURE_KEYS)
        }

        return ordered_vals, feature_dict

    @classmethod
    def to_tensor(
        cls,
        features: Union[List[float], Dict[str, float], np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Validates and converts clinical features into a PyTorch batch tensor (1, 16).
        """
        ordered_vals, _ = cls.validate_and_serialize(features)
        return torch.tensor([ordered_vals], dtype=torch.float32)
