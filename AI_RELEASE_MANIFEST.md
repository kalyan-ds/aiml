# AI Release Manifest — SIH 2026 Retinal AI Diabetic Retinopathy Screening

**Release Status**: FROZEN RELEASE CANDIDATE  
**Release Date**: 2026-09-14  
**Project**: Smart India Hackathon 2026 — Retinal AI Diabetic Retinopathy Screening  
**Owning Discipline**: AI/ML Engineering  
**Consumer**: Backend / Integration Service (`BaseInferenceService`)

---

## 1. Release Identification & Cryptographic Digest

| Parameter | Frozen Value |
|---|---|
| **Model Version** | `resnet50_fusion_v1.0.0` |
| **Canonical Checkpoint Path** | `aiml/models_weights/resnet50_fusion_v1.0.0.pt` |
| **Canonical SHA-256 Digest** | `0c1ed55f18864861b11960f6633d3105991d7c16160b0c7b0754900d3228292e` |
| **Feature Contract Version** | `feature_v1.0` |
| **Preprocessing Version** | `rgb_512_imagenet_v1` |
| **Calibrated Operating Threshold ($\tau^*$)** | `0.17` |
| **Clinical Normalization Policy** | `zscore_frozen_training_v1.0` |
| **Training Master Seed** | `42` |
| **Checkpoint Best Epoch** | `5` |
| **Automated Test Count** | `22 / 22 Passed (100%)` |

> [!IMPORTANT]
> **HASH MISMATCH RESOLUTION**:
> The SHA-256 hash `0c1ed55f18864861b11960f6633d3105991d7c16160b0c7b0754900d3228292e` represents the **sole canonical trained and calibrated model checkpoint**. Any hash previously referenced in early integration stubs belonged to an untrained epoch-0 dummy initialization. The Backend service MUST ingest and verify against this canonical hash.

---

## 2. Frozen Multimodal Architecture Specification

```
                          Fundus Image (RGB, any dimension)
                                         │
                                         ▼
                     Deterministic Preprocessor (512x512, ImageNet Norm)
                                         │
                                         ▼
                             ResNet-50 Feature Extractor
                             (Truncated at Global Pooling)
                                         │
                                         ▼
                                   v_cnn ∈ R^2048
                                         │
Canonical 16-D Clinical Vector ──────────┼─────────────────────────┐
             │                           │                         │
             ▼                           │                         │
Deterministic Training Normalization     │                         ▼
(F - mu) / (sigma + eps)                 │               Grad-CAM Generator
             │                           │               Target: layer4[2]
             ▼                           │               JET Colormap (LUT)
Dense Projection (16 -> 32) + ReLU       │               Alpha Blend: 45%
             │                           │                         │
             ▼                           │                         │
       v_clin ∈ R^32                     │                         ▼
             │                           │               Heatmap Overlay File
             └───────────────────────────┼─────────────────────────┘
                                         ▼
                             Multimodal Fusion Layer
                                    z ∈ R^2080
                                         │
                                         ▼
                                 Dropout (p = 0.4)
                                         │
                                         ▼
                             Dense Layer (2080 -> 128)
                                         │
                                         ▼
                                       ReLU
                                         │
                                         ▼
                                 Dropout (p = 0.2)
                                         │
                                         ▼
                              Dense Layer (128 -> 5)
                                         │
                                         ▼
                               5 Raw Severity Logits
                                         │
                                         ▼
                                   5-Unit Softmax
                              P = [P0, P1, P2, P3, P4]
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 ICDR Grade & Label               Referable Risk Score
                   Argmax(P0..P4)                 Score_ref = P2 + P3 + P4
                                                         │
                                                         ▼
                                                Triage Decision:
                                                is_referable = (Score_ref >= 0.17)
```

- **Fusion Dimension Contract**: Exactly $2048 + 32 = 2080$.
- **Demographic Exclusion Contract**: Age, diabetes duration, patient ID, and condensing lens diopter are strictly excluded from neural network weights and tensor inputs.

---

## 3. ICDR Target Classes

| Level | Severity Name | Clinical Definition | Referral Decision |
|:---:|---|---|:---:|
| **0** | No DR | Absence of microaneurysms, hemorrhages, and exudates | Routine Annual Follow-Up |
| **1** | Mild NPDR | Microaneurysms only | 6–12 Month Review |
| **2** | Moderate NPDR | More than microaneurysms, less than severe NPDR | **Referable** ($\text{Score}_{ref} \ge 0.17$) |
| **3** | Severe NPDR | Clinical 4:2:1 Rule: severe hemorrhages in all 4 quadrants | **Referable Urgent** ($\text{Score}_{ref} \ge 0.17$) |
| **4** | Proliferative DR (PDR) | Neovascularization (NVD/NVE) or vitreous hemorrhage | **Referable Emergency** ($\text{Score}_{ref} \ge 0.17$) |

---

## 4. Canonical 16-D Feature Contract

The 16-element clinical vector $\mathbf{F} \in \mathbb{R}^{16}$ represents anatomical partitioning centered at the fovea (ST: Superior-Temporal, SN: Superior-Nasal, IT: Inferior-Temporal, IN: Inferior-Nasal):

| Index | Key Name | Scale / Unit | Frozen Definition & Edge Handling |
|:---:|---|:---:|---|
| `F[0]` | `ma_count_st` | Count $\ge 0$ | Superior-Temporal microaneurysm count |
| `F[1]` | `ma_count_sn` | Count $\ge 0$ | Superior-Nasal microaneurysm count |
| `F[2]` | `ma_count_it` | Count $\ge 0$ | Inferior-Temporal microaneurysm count |
| `F[3]` | `ma_count_in` | Count $\ge 0$ | Inferior-Nasal microaneurysm count |
| `F[4]` | `hem_area_st` | $\text{px}^2 \ge 0$ | Superior-Temporal hemorrhage area |
| `F[5]` | `hem_area_sn` | $\text{px}^2 \ge 0$ | Superior-Nasal hemorrhage area |
| `F[6]` | `hem_area_it` | $\text{px}^2 \ge 0$ | Inferior-Temporal hemorrhage area |
| `F[7]` | `hem_area_in` | $\text{px}^2 \ge 0$ | Inferior-Nasal hemorrhage area |
| `F[8]` | `vessel_density_st` | $\%$ $[0.0, 100.0]$ | Superior-Temporal Frangi vascular density ratio |
| `F[9]` | `vessel_density_sn` | $\%$ $[0.0, 100.0]$ | Superior-Nasal Frangi vascular density ratio |
| `F[10]`| `vessel_density_it` | $\%$ $[0.0, 100.0]$ | Inferior-Temporal Frangi vascular density ratio |
| `F[11]`| `vessel_density_in` | $\%$ $[0.0, 100.0]$ | Inferior-Nasal Frangi vascular density ratio |
| `F[12]`| `total_microaneurysms` | Count $\ge 0$ | Total microaneurysms ($\sum F[0..3]$). Intentionally retained. |
| `F[13]`| `total_hemorrhage_area_px` | $\text{px}^2 \ge 0$ | Total hemorrhage area ($\sum F[4..7]$). Intentionally retained. |
| `F[14]`| `total_exudates_count` | Count $\ge 0$ | Discrete segmented hard exudate lesion connected components. |
| `F[15]`| `min_exudate_distance_to_fovea_px` | $\text{px} \ge 0$ | Min Euclidean distance from exudates to fovea center. **Sentinel**: When $F[14] == 0$, $F[15] = 724.08\text{ px}$ (maximum image-space sentinel distance when no exudates are detected). |

### Frozen Normalization Vectors:
$$\mathbf{F}_{\text{norm}} = \frac{\mathbf{F} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + 10^{-6}}$$
- $\boldsymbol{\mu} = [4.83, 4.80, 4.61, 5.04, 43.34, 40.97, 43.68, 44.28, 14.22, 14.18, 14.12, 14.17, 19.29, 172.26, 8.13, 496.99]$
- $\boldsymbol{\sigma} = [7.36, 7.30, 6.69, 7.59, 82.56, 75.26, 82.09, 82.61, 1.14, 1.21, 1.10, 1.14, 27.89, 312.10, 13.20, 281.31]$

---

## 5. Evaluation & Performance Summary

### Evaluation on Held-Out Test Cohort ($N=30$ independent scans, zero patient overlap):
- **Overall Accuracy**: $83.33\%$ ($0.8333$)
- **Macro F1 Score**: $0.7331$
- **Referable Sensitivity (TPR)**: $100.0\%$ ($1.0000$ at $\tau^* = 0.17$)
- **Referable Specificity (TNR)**: $85.71\%$ ($0.8571$ at $\tau^* = 0.17$)
- **Referable ROC-AUC**: $1.0000$

### Measured Inference Latency Breakdown (Host CPU, 25 iterations):
- **Preprocessing** (RGB resize $512 \times 512$ + ImageNet norm): $3.34 \pm 1.21\text{ ms}$
- **ResNet-50 CNN Backbone** ($\text{GAP} \to 2048$-D): $302.53 \pm 37.60\text{ ms}$
- **Clinical Projection** ($16$-D $\to 32$-D): $0.15 \pm 0.05\text{ ms}$
- **Multimodal Fusion & Head** ($2080 \to 5$ logits): $0.28 \pm 0.09\text{ ms}$
- **Grad-CAM Generation & JET Overlay Blend**: $344.11 \pm 55.43\text{ ms}$
- **Total Forward Pass WITHOUT Grad-CAM**: **$306.31 \pm 37.91\text{ ms}$**
- **Total End-to-End Latency WITH Grad-CAM**: **$650.42 \pm 75.44\text{ ms}$**

---

## 6. Backend Integration Interface

```python
from aiml.src.inference.predictor import RealRetinalAIPredictor

# Initialize service singleton
predictor = RealRetinalAIPredictor(
    checkpoint_path="aiml/models_weights/resnet50_fusion_v1.0.0.pt",
    device="cpu",  # or "cuda"
    heatmap_output_dir="outputs/heatmaps",
    default_threshold=0.17,
)

# Standard inference call
result = predictor.predict(
    image=raw_image_bytes_or_pil_or_numpy,
    clinical_features=opaque_16d_list_or_dict,
    lens_diopter=20.0,
)
```

The response adheres strictly to the backend JSON contract:
- `status`: `"success"` or `"error"`
- `gradeable`: `true` or `false`
- `error_code`: `null` (or `"ERR_MATLAB_UNAVAILABLE"` if clinical features are omitted)
- `processing_mode`: `"hybrid"`
- `icdr_grade`: `0..4`
- `icdr_label`: `"No DR"` through `"Proliferative DR (PDR)"`
- `probabilities`: `{"P0": float, "P1": float, "P2": float, "P3": float, "P4": float}`
- `referable_risk`: float ($P_2 + P_3 + P_4$)
- `threshold`: `0.17`
- `is_referable`: boolean
- `confidence`: $\max(P_0 \dots P_4)$
- `clinical_features`: dictionary of 16 unnormalized feature values
- `explainability`: `{"cam_status": "success", "cam_overlay_path": "..."}`
- `inference_latency_ms`: float
- `model_version`: `"resnet50_fusion_v1.0.0"`

---

## 7. Known System Limitations

1. **Investigational Prototype**: The system is intended strictly as a primary healthcare decision-support screening aid and has not received FDA clearance or clinical trial validation.
2. **Missing Feature Rejection**: If MATLAB/CV features are missing (`None`), the pipeline returns `ERR_MATLAB_UNAVAILABLE` by design; zero-filling is prohibited to prevent false negatives. A CNN-only fallback requires an independently trained 5-class checkpoint.
3. **Hardware Latency**: While Grad-CAM has been optimized ($1006\text{ ms} \to 344\text{ ms}$ on CPU), total screening latency on low-power rural ARM devices is estimated at $\sim 1.5 - 2.5\text{ s}$ per scan without GPU acceleration.
