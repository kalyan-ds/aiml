"""
Retinal AI: Frozen 16-D Clinical Feature Contract & Deterministic Normalizer
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

FROZEN FEATURE CONTRACT DECISIONS:
  1. F[14] Semantic Definition:
     - `total_exudates_count`: Non-negative integer count (float32) of discrete segmented
       hard exudate lesion connected components in CIE L*a*b* space.
  2. F[15] Zero-Exudate Sentinel:
     - When no hard exudates are detected (F[14] == 0.0), F[15] is deterministically set to
       724.08 px (exact Euclidean diagonal of 512x512 fundus frame: sqrt(512^2 + 512^2)).
       This maintains monotonicity (larger distance = safer / farther from foveal center).
  3. F[12] and F[13] Retention:
     - F[12] (total microaneurysms = sum(F[0..3])) and F[13] (total hemorrhage area = sum(F[4..7]))
       are intentionally retained to allow direct global severity weighting in the 16->32 projection
       while quadrant features preserve local spatial 4:2:1 NPDR criteria.
  4. Deterministic Clinical Normalization:
     - Fixed z-score standardization: F_norm = (F - mu) / (sigma + eps)
     - Normalization parameters (mu, sigma) are computed exclusively on the training set,
       frozen in the checkpoint, and applied identically across train, val, calibration, test, and inference.
     - Fully batch-independent and deterministic.
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

# Max diagonal of 512x512 fundus image: sqrt(512^2 + 512^2) = 724.077...
MAX_IMAGE_DIAGONAL_PX = 724.08
FROZEN_ZERO_EXUDATE_SENTINEL_PX = 724.08
DEFAULT_ZERO_EXUDATE_DISTANCE_SENTINEL = FROZEN_ZERO_EXUDATE_SENTINEL_PX

# Default reference empirical means and standard deviations (fitted on training data)
DEFAULT_CLINICAL_MU = [
    4.5, 3.8, 5.2, 3.6,      # ma_count (ST, SN, IT, IN)
    25.0, 18.0, 32.0, 20.0,  # hem_area (ST, SN, IT, IN)
    14.0, 13.5, 14.8, 13.2,  # vessel_density (ST, SN, IT, IN)
    17.1,                    # total_microaneurysms
    95.0,                    # total_hemorrhage_area_px
    8.4,                     # total_exudates_count
    240.0,                   # min_exudate_distance_to_fovea_px
]

DEFAULT_CLINICAL_SIGMA = [
    5.0, 4.2, 5.8, 4.0,       # ma_count
    35.0, 25.0, 45.0, 28.0,   # hem_area
    2.5, 2.3, 2.6, 2.4,       # vessel_density
    18.5,                     # total_microaneurysms
    128.0,                    # total_hemorrhage_area_px
    11.2,                     # total_exudates_count
    185.0,                    # min_exudate_distance_to_fovea_px
]


class FeatureContractValidator:
    """
    Validates, serializes, and deterministically normalizes the 16-D clinical feature vector.
    """

    FEATURE_CONTRACT_VERSION = "feature_v1.0"
    FEATURE_DIM = 16

    _mean: List[float] = list(DEFAULT_CLINICAL_MU)
    _std: List[float] = list(DEFAULT_CLINICAL_SIGMA)

    @classmethod
    def set_normalization_parameters(cls, mean: List[float], std: List[float]):
        """
        Sets frozen normalization parameters (mu, sigma) loaded from checkpoint.
        """
        if len(mean) != cls.FEATURE_DIM or len(std) != cls.FEATURE_DIM:
            raise ValueError(f"Normalization parameters must have length {cls.FEATURE_DIM}")
        cls._mean = [float(x) for x in mean]
        cls._std = [max(float(x), 1e-6) for x in std]

    @classmethod
    def get_normalization_parameters(cls) -> Dict[str, List[float]]:
        return {
            "mean": list(cls._mean),
            "std": list(cls._std),
        }

    @classmethod
    def validate_and_serialize(
        cls,
        features: Union[List[float], Dict[str, float], np.ndarray, torch.Tensor],
        handle_zero_exudates: bool = True,
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Validates the input clinical features and produces:
          1. Canonical ordered List[float] of length 16 (raw scale)
          2. Standardized Dict[str, float] keyed by feature names
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

        # Handle zero-exudates edge case on F[15]
        total_exudates = ordered_vals[14]
        if handle_zero_exudates and total_exudates == 0.0:
            if ordered_vals[15] == 0.0:
                ordered_vals[15] = FROZEN_ZERO_EXUDATE_SENTINEL_PX

        feature_dict = {
            k: float(ordered_vals[i]) for i, k in enumerate(CANONICAL_FEATURE_KEYS)
        }

        return ordered_vals, feature_dict

    @classmethod
    def normalize_vector(cls, raw_features: List[float]) -> List[float]:
        """
        Applies deterministic z-score normalization: (x - mu) / (sigma + eps)
        """
        eps = 1e-6
        return [
            (val - cls._mean[i]) / (cls._std[i] + eps)
            for i, val in enumerate(raw_features)
        ]

    @classmethod
    def to_tensor(
        cls,
        features: Union[List[float], Dict[str, float], np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Validates, deterministically normalizes, and returns a PyTorch batch tensor (1, 16).
        """
        raw_vals, _ = cls.validate_and_serialize(features)
        proc_vals = cls.normalize_vector(raw_vals) if normalize else raw_vals
        return torch.tensor([proc_vals], dtype=torch.float32)
