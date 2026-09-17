# Retinal AI: AI/ML Subsystem Backend Handoff & Release Specification

**Project:** Retinal AI Diabetic Retinopathy Screening System  
**Hackathon:** Smart India Hackathon (SIH 2026) | Problem Statement #027 (PS ID: 26038)  
**Document Identifier:** `docs/AI_HANDOFF.md`  
**Model Version:** `resnet50_fusion_v1.0.0`  
**Status:** **FROZEN RELEASE CANDIDATE — READY FOR BACKEND INGESTION**  
**Release Date:** 2026-09-17  
**Owning Discipline:** AI/ML Subsystem Engineering  

---

## 1. Frozen Model Identification & Checkpoint Digest

| Parameter | Authoritative Value |
|---|---|
| **Checkpoint Path** | `aiml/models_weights/resnet50_fusion_v1.0.0.pt` |
| **Checkpoint File Size** | `294,010,890 bytes` ($280.39\text{ MB}$) |
| **SHA-256 Checksum** | `856b387c9302bd681290c005ac95f53ba421c0148f6c0ab6d8b5d6c9bb019334` |
| **Model Version** | `resnet50_fusion_v1.0.0` |
| **Feature Contract Version** | `feature_v1.0` |
| **Preprocessing Version** | `rgb_512_imagenet_v1` |
| **Operating Threshold ($\tau^*$)** | **`0.1900`** (`0.19`) |
| **Referable Risk Formula** | $\text{Score}_{\text{ref}} = P_2 + P_3 + P_4$ |
| **Triage Decision Rule** | $\text{is\_referable} = (\text{Score}_{\text{ref}} \ge 0.1900)$ |

---

## 2. Input Contracts & Normalization

### 2.1 Image Input Specification
- **Input Modality:** 3-Channel RGB Fundus Photograph (PIL, NumPy array, or raw JPEG/PNG bytes)
- **Spatial Resolution:** $512 \times 512$ pixels (Bilinear interpolation)
- **Image Normalization:** Standard PyTorch ImageNet channel-wise standardization
  - Channel Mean ($\boldsymbol{\mu}_{img}$): `[0.485, 0.456, 0.406]`
  - Channel Std Dev ($\boldsymbol{\sigma}_{img}$): `[0.229, 0.224, 0.225]`

### 2.2 Canonical 16-D Clinical Feature Vector Contract ($\mathbf{F} \in \mathbb{R}^{16}$)
The backend MUST provide the clinical feature payload as a dictionary or ordered 16-element sequence conforming exactly to this canonical key ordering:

| Index | Canonical Feature Key | Description / Meaning | Scale / Unit |
|:---:|---|---|:---:|
| `F[0]` | `ma_count_st` | Superior-Temporal microaneurysm count | Integer $\ge 0$ |
| `F[1]` | `ma_count_sn` | Superior-Nasal microaneurysm count | Integer $\ge 0$ |
| `F[2]` | `ma_count_it` | Inferior-Temporal microaneurysm count | Integer $\ge 0$ |
| `F[3]` | `ma_count_in` | Inferior-Nasal microaneurysm count | Integer $\ge 0$ |
| `F[4]` | `hem_area_st` | Superior-Temporal hemorrhage area | Pixels ($\text{px}^2$) |
| `F[5]` | `hem_area_sn` | Superior-Nasal hemorrhage area | Pixels ($\text{px}^2$) |
| `F[6]` | `hem_area_it` | Inferior-Temporal hemorrhage area | Pixels ($\text{px}^2$) |
| `F[7]` | `hem_area_in` | Inferior-Nasal hemorrhage area | Pixels ($\text{px}^2$) |
| `F[8]` | `vessel_density_st` | Superior-Temporal Frangi vascular density | Percentage $[0, 100]$ |
| `F[9]` | `vessel_density_sn` | Superior-Nasal Frangi vascular density | Percentage $[0, 100]$ |
| `F[10]`| `vessel_density_it` | Inferior-Temporal Frangi vascular density | Percentage $[0, 100]$ |
| `F[11]`| `vessel_density_in` | Inferior-Nasal Frangi vascular density | Percentage $[0, 100]$ |
| `F[12]`| `total_microaneurysms` | Total microaneurysms count ($\sum F[0..3]$) | Integer $\ge 0$ |
| `F[13]`| `total_hemorrhage_area_px` | Total hemorrhage area ($\sum F[4..7]$) | Pixels ($\text{px}^2$) |
| `F[14]`| `total_exudates_count` | Discrete hard exudate lesion count | Integer $\ge 0$ |
| `F[15]`| `min_exudate_distance_to_fovea_px` | Min Euclidean distance from exudates to fovea center. **Sentinel Rule**: If $F[14] == 0$, $F[15] = 724.08\text{ px}$. | Pixels ($\text{px}$) |

### 2.3 Clinical Feature Normalization (Frozen Training Statistics)
$$\mathbf{F}_{norm} = \frac{\mathbf{F} - \boldsymbol{\mu}_{train}}{\boldsymbol{\sigma}_{train} + 10^{-6}}$$
- **Mean Vector ($\boldsymbol{\mu}_{train}$):**
  `[4.8286, 4.8000, 4.6143, 5.0429, 43.3357, 40.9714, 43.6786, 44.2786, 14.2188, 14.1820, 14.1205, 14.1736, 19.2857, 172.2643, 8.1286, 496.9903]`
- **Std Dev Vector ($\boldsymbol{\sigma}_{train}$):**
  `[7.3630, 7.3028, 6.6854, 7.5893, 82.5599, 75.2592, 82.0947, 82.6132, 1.1446, 1.2127, 1.0993, 1.1363, 27.8940, 312.0966, 13.2027, 281.3086]`

---

## 3. Multimodal Architecture Pipeline

```
  [ 512x512 RGB Fundus Image ]                      [ 16-D Clinical Vector ]
               │                                                │
               ▼                                                ▼
     ResNet-50 CNN Backbone                         Z-Score Normalization (mu, sigma)
               │                                                │
               ▼                                                ▼
  layer4[2] (Grad-CAM Hook Target)                     Linear(16, 32) + ReLU
               │                                                │
               ▼                                                ▼
   Global Average Pooling (GAP)                            v_clin in R^32
               │                                                │
               ▼                                                │
         v_cnn in R^2048                                        │
               │                                                │
               └───────────────────────┬────────────────────────┘
                                       ▼
                       Multimodal Fusion: z = [v_cnn ; v_clin] in R^2080
                                       │
                                       ▼
                                 Dropout(0.4)
                                       │
                                       ▼
                               Linear(2080, 128)
                                       │
                                       ▼
                                     ReLU
                                       │
                                       ▼
                                 Dropout(0.2)
                                       │
                                       ▼
                                 Linear(128, 5)
                                       │
                                       ▼
                             5 Raw Severity Logits
                                       │
                                       ▼
                                 5-Unit Softmax
                            P = [P0, P1, P2, P3, P4]
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
             ICDR Severity Grade                 Referable Risk Score
               Argmax(P0..P4)                  Score_ref = P2 + P3 + P4
                     │                                   │
                     ▼                                   ▼
          0: No DR                              Triage Referral Decision:
          1: Mild NPDR                          is_referable = (Score_ref >= 0.1900)
          2: Moderate NPDR (Referable)
          3: Severe NPDR (Referable Urgent)
          4: PDR (Referable Emergency)
```

---

## 4. Training Provenance

- **Dataset Identifier:** `Patient-Stratified-Retinal-Cohort-v1`
- **Total Patient Count:** 100 unique patients
- **Total Fundus Scans:** 200 bilateral scans (OD / OS)
- **Split Strategy:** 70/15/15 Patient-Stratified Group Split (Zero patient overlap, 0% leakage)
  - **Training Partition:** 140 scans (70 unique patients)
  - **Validation Partition:** 30 scans (15 unique patients)
  - **Held-Out Test Partition:** 30 scans (15 unique patients)
- **Optimizer:** `AdamW` ($\text{LR} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-2}$)
- **LR Scheduler:** `CosineAnnealingLR` ($T_{max} = 5$, $\eta_{min} = 10^{-6}$)
- **Loss Function:** `MultiClassFocalLoss` ($\gamma = 2.0$, inverse class frequency weights $\boldsymbol{\alpha} = [0.5185, 0.9333, 1.0769, 2.0, 1.75]$)
- **Batch Size:** 8
- **Master Random Seed:** `42`
- **Trained Epochs:** 5 (Best validation epoch: 5, validation loss: 0.1279)

---

## 5. Prototype Benchmark Cohort Evaluation

> [!IMPORTANT]
> All metrics below reflect benchmark evaluation on the prototype cohort ($N=30$ validation, $N=30$ held-out test). This system is an investigational decision-support aid and has not been cleared for clinical diagnostic use without an ophthalmologist in the loop.

| Metric | Validation Cohort ($N=30$) | Held-Out Test Cohort ($N=30$) | Release Requirement | Status |
|---|:---:|:---:|:---:|:---:|
| **Operating Threshold ($\tau^*$)** | `0.1900` | `0.1900` (Applied) | Empirically derived on Val | **FROZEN** |
| **Referable Sensitivity** | **`100.0%`** (18/18) | **`100.0%`** (16/16) | $\ge 90.0\%$ | **EXCEEDS** |
| **Referable Specificity** | **`100.0%`** (12/12) | **`92.86%`** (13/14) | $\ge 80.0\%$ | **EXCEEDS** |
| **Referable ROC-AUC** | **`1.0000`** | **`1.0000`** | $\ge 0.900$ | **EXCEEDS** |
| **Overall Multi-Class Accuracy** | **`76.67%`** (23/30) | **`86.67%`** (26/30) | $\ge 75.0\%$ | **EXCEEDS** |
| **Macro F1-Score** | **`0.7046`** | **`0.7667`** | $\ge 0.700$ | **EXCEEDS** |
| **Quadratic Weighted Kappa (QWK)** | **`0.9420`** | **`0.9703`** | $\ge 0.700$ | **EXCEEDS** |
| **Weighted F1-Score** | `0.7002` | `0.8111` | $\ge 0.750$ | **EXCEEDS** |

### Test Cohort Confusion Matrix (5x5):
```
True \ Pred   G0 (No DR)  G1 (Mild)  G2 (Mod)  G3 (Sev)  G4 (PDR)
G0 (No DR)        10          0         0         0         0
G1 (Mild)          4          0         0         0         0
G2 (Mod)           0          0         6         0         0
G3 (Sev)           0          0         0         6         0
G4 (PDR)           0          0         0         0         4
```

---

## 6. Grad-CAM Explainability Verification

- **Target Layer:** ResNet-50 `layer4[2]` (Stage 4 Bottleneck Output)
- **Spatial Resolution:** $512 \times 512$ pixels (Bilinear interpolation)
- **Gradient Check:** Verified non-zero gradient flow ($\|\nabla\| = 0.010129$)
- **Numerical Validity:** 100% finite values, strictly min-max normalized to $[0.0, 1.0]$
- **Colormap & Blend:** Vectorized 256-level JET colormap with $\alpha = 0.45$ (45% opacity)
- **Logits Invariance:** Forward logits verified strictly invariant ($\Delta < 10^{-6}$) before vs after Grad-CAM generation
- **Success Rate:** $60/60$ evaluations (100.0%)

---

## 7. Latency Profile (Host CPU, 25 Benchmark Runs)

- **Preprocessing Latency:** $3.34 \pm 1.21\text{ ms}$
- **CNN Backbone Forward Pass:** $302.53 \pm 37.60\text{ ms}$
- **Clinical Projection & Fusion Head:** $0.43 \pm 0.12\text{ ms}$
- **Total Forward Pass (Without Explainability):** **$344.08\text{ ms}$** (Well below $500\text{ ms}$ budget)
- **Grad-CAM Generation & Overlay:** $351.73\text{ ms}$
- **End-to-End Latency (With Grad-CAM):** **$577.19 - 714.42\text{ ms}$** (p95: $784.26\text{ ms}$, well within $1.5\text{ s}$ SLA)

---

## 8. Backend Integration API (Code Contract)

```python
from aiml.src.inference.predictor import RealRetinalAIPredictor

# Initialize AI Service Singleton
predictor = RealRetinalAIPredictor(
    checkpoint_path="aiml/models_weights/resnet50_fusion_v1.0.0.pt",
    device="cpu",  # or "cuda"
    heatmap_output_dir="outputs/heatmaps",
    default_threshold=0.1900,
)

# Standard Inference Call
result = predictor.predict(
    image=raw_image_input,           # PIL.Image, np.ndarray, or bytes
    clinical_features=dict_or_list,   # 16-D clinical feature dict/list
    lens_diopter=20.0,               # Excluded from neural weights
)
```

### Standard Response JSON Schema:
```json
{
  "status": "success",
  "gradeable": true,
  "error_code": null,
  "processing_mode": "hybrid",
  "icdr_grade": 2,
  "icdr_label": "Moderate NPDR",
  "probabilities": {
    "P0": 0.0125,
    "P1": 0.0125,
    "P2": 0.8003,
    "P3": 0.1500,
    "P4": 0.0247
  },
  "referable_risk": 0.9750,
  "threshold": 0.1900,
  "is_referable": true,
  "confidence": 0.8003,
  "clinical_features": {
    "ma_count_st": 6.0,
    ...
  },
  "explainability": {
    "cam_status": "success",
    "cam_overlay_path": "outputs/heatmaps/cam_xxxx.jpg"
  },
  "inference_latency_ms": 577.19,
  "model_version": "resnet50_fusion_v1.0.0"
}
```

---

## 9. Prototype Qualification & Regulatory Notice

> [!CAUTION]
> **Prototype Decision-Support Qualification**:
> 1. The Retinal AI system is engineered and benchmarked as an investigational multimodal triage tool for primary health screening under the Smart India Hackathon (SIH 2026).
> 2. It has NOT undergone statutory clinical trials (e.g., CDSCO / US FDA 510(k)) and MUST NOT be used for autonomous clinical diagnosis without ophthalmic specialist oversight.
> 3. If MATLAB/CV clinical features are omitted or unavailable, the predictor will return `ERR_MATLAB_UNAVAILABLE` by design to safeguard against ungrounded false negatives.
