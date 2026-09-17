# Retinal AI: Production AI Release Manifest

**Project:** Smart India Hackathon 2026 — Retinal AI Diabetic Retinopathy Screening  
**PS ID:** 26038 | **Model Version:** `resnet50_fusion_v1.0.0`  
**Release Date:** 2026-09-15 | **Status:** **FROZEN & VALIDATED**  
**Deterministic Random Seed:** `42`

---

## 1. Release Checkpoint Identification

- **Checkpoint Filename:** `resnet50_fusion_v1.0.0.pt`
- **Checkpoint Location:** [`aiml/models_weights/resnet50_fusion_v1.0.0.pt`](file:///d:/ai-ml/aiml/models_weights/resnet50_fusion_v1.0.0.pt)
- **SHA-256 Checksum (Exact File Bytes):**
  ```
  856b387c9302bd681290c005ac95f53ba421c0148f6c0ab6d8b5d6c9bb019334
  ```
- **Checkpoint File Size:** `294,010,890` bytes (`280.39` MB)
- **PyTorch Format:** Serialized state dictionary bundle with metadata header.

---

## 2. Frozen Multimodal Architecture Specification

```
                          [ 512x512x3 Fundus Image ]
                                       │
                                       ▼
                       ResNet-50 Backbone (Pretrained)
                                       │
                                       ▼
                         layer4[2] Feature Map (16x16x2048)  ◄── Grad-CAM Hook Target
                                       │
                                       ▼
                          Global Average Pooling (GAP)
                                       │
                                       ▼
                           v_cnn in R^2048 (Image Embedding)
                                       │
[ 16-D Clinical Vector ]               │
           │                           │
           ▼                           │
Deterministic Z-Score                  │
(mu_train, sigma_train)                │
           │                           │
           ▼                           │
Linear(16, 32) + ReLU                  │
           │                           │
           ▼                           │
v_clin in R^32 (Clinical Embedding)    │
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
         Multimodal Fusion Concatenation: z = [v_cnn ; v_clin] in R^2080
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
      5 Raw Logits (ICDR Severity Grades: P0, P1, P2, P3, P4)
                       │
                       ▼
             Softmax Normalization: sum(P_i) = 1.0
                       │
                       ▼
   Referable Risk Decision: Score_ref = P2 + P3 + P4 >= tau* (0.1900)
```

### Immutable Layer Dimensionality Contract:
1. **Clinical Feature Input Dimension:** Exactly $16$
2. **CNN Backbone GAP Dimension ($v_{\text{cnn}}$):** Exactly $2048$
3. **Clinical Feature Projection ($v_{\text{clin}}$):** Exactly $32$ (`Linear(16, 32)` + `ReLU`)
4. **Fused Multimodal Dimension ($z$):** Exactly $2080$ ($2048 + 32$)
5. **Intermediate Classification Latent:** Exactly $128$ (`Linear(2080, 128)` + `ReLU`)
6. **Final Classification Logits:** Exactly $5$ (`Linear(128, 5)`)

---

## 3. Preprocessing & Normalization Specifications

### Fundus Image Preprocessing:
- **Input Color Space:** Standard RGB
- **Input Spatial Dimension:** $512 \times 512$ pixels
- **Interpolation Mode:** Bilinear resampling
- **Tensor Range:** $[0.0, 1.0]$ float32
- **Image Normalization:** Standard ImageNet channel-wise normalization
  - Mean: $\mu = [0.485, 0.456, 0.406]$
  - Std Dev: $\sigma = [0.229, 0.224, 0.225]$

### Clinical Feature Normalization:
- **Feature Contract Version:** `v1.0.0_16D`
- **Ordering:** Frozen canonical 16-key index ordering (`FeatureContractValidator.CANONICAL_FEATURE_KEYS`)
- **Zero-Exudate Distance Sentinel:** $999.0\text{ px}$
- **Deterministic Z-Score Statistics (Fitted on Training Split, $N=140$):**
  - **Mean Vector ($\mu_{\text{train}}$):** `[4.8286, 4.8, 4.6143, 5.0429, 43.3357, 40.9714, 43.6786, 44.2786, 14.2188, 14.182, 14.1205, 14.1736, 19.2857, 172.2643, 8.1286, 496.9903]`
  - **Std Dev Vector ($\sigma_{\text{train}}$):** `[7.363, 7.3028, 6.6854, 7.5893, 82.5599, 75.2592, 82.0947, 82.6132, 1.1446, 1.2127, 1.0993, 1.1363, 27.894, 312.0966, 13.2027, 281.3086]`

---

## 4. Triage Calibration & Referable Operating Point

- **Referable Risk Definition:** $\text{Score}_{\text{ref}} = P_2 + P_3 + P_4$
- **Frozen Calibrated Threshold ($\tau^*$):** `0.1900`
- **Calibration Target:** $\ge 90\%$ Sensitivity for vision-threatening DR on validation cohort.
- **Achieved Sensitivity on Validation:** $100.0\%$
- **Decision Rule:**
  $$\text{Triage Action} = \begin{cases} \text{REFERABLE (Urgent/Emergency Ophthalmic Workup)}, & \text{Score}_{\text{ref}} \ge 0.1900 \\ \text{NON-REFERABLE (Routine Annual Screening)}, & \text{Score}_{\text{ref}} < 0.1900 \end{cases}$$

---

## 5. Training Configuration & Reproducibility Parameters

- **Optimizer:** `AdamW` (Initial $\text{LR} = 3 \times 10^{-4}$, Weight Decay $= 10^{-2}$)
- **LR Scheduler:** `CosineAnnealingLR` ($T_{\max} = 10$, $\eta_{\min} = 10^{-6}$)
- **Loss Function:** `MultiClassFocalLoss` ($\gamma = 2.0$, Inverse Class Frequency $\alpha$-weights)
- **Epochs Trained:** 10 (Best Validation Epoch: 5, Validation Loss: 0.1279)
- **Batch Size:** 8
- **Device:** CPU / CUDA compatible
- **Dataset Partition:** Patient-level grouped split (70% Train / 15% Val / 15% Held-Out Test)
  - Training scans: 140 (70 patients)
  - Validation scans: 30 (15 patients)
  - Held-out test scans: 30 (15 patients)

---

## 6. Evaluation Benchmark Summary (Held-Out Test Set, $N=30$)

| Benchmark Metric | Target Specification | Held-Out Test Score | Evaluation Result |
| :--- | :--- | :--- | :--- |
| **Referable DR Sensitivity** | $\ge 90.0\%$ | **100.0%** (16/16) | **EXCEEDS SPEC** |
| **Referable DR Specificity** | $\ge 80.0\%$ | **92.86%** (13/14) | **EXCEEDS SPEC** |
| **Referable DR ROC-AUC** | $\ge 0.900$ | **1.0000** | **EXCEEDS SPEC** |
| **Multi-Class Accuracy** | $\ge 75.0\%$ | **86.67%** (26/30) | **EXCEEDS SPEC** |
| **Quadratic Weighted Kappa (QWK)** | $\ge 0.700$ | **0.9703** | **EXCEEDS SPEC** |
| **Macro F1-Score** | $\ge 0.700$ | **0.7667** | **EXCEEDS SPEC** |
| **Weighted F1-Score** | $\ge 0.750$ | **0.8111** | **EXCEEDS SPEC** |
| **Grad-CAM Success Rate** | $100.0\%$ | **100.0%** (60/60) | **PASS** |
| **End-to-End CPU Latency** | $< 1500\text{ ms}$ | **577.19 ms** (p95: 784 ms) | **PASS** |

---

## 7. Verification Sign-Off

- **Checkpoint Integrity:** **VALID** (19/19 technical checks passed)
- **Model Reload Test:** **PASS** (Subprocess fresh load confirmed)
- **Inference Invariance:** **PASS** (Logit difference $< 10^{-6}$)
- **Grad-CAM Generation:** **PASS** (Target `layer4[2]`, non-zero gradients, finite values)
- **Test Suite Status:** **34/34 tests passing**

---
*Generated deterministically by `aiml/scripts/verify_release_checkpoint.py` | Smart India Hackathon 2026*
