# AI/ML Subsystem Frozen Reproducibility Record

**Project**: Retinal AI Diabetic Retinopathy Screening System  
**Hackathon Target**: Smart India Hackathon (SIH 2026) | Problem Statement #027 (PS ID: 26038)  
**Status**: **FROZEN RELEASE CANDIDATE — ZERO MODIFICATIONS PERMITTED**  
**Freeze Date**: 2026-09-15  
**Owning Discipline**: AI/ML Engineering  

---

## 1. Frozen Model Identification & Checkpoint Digest

| Parameter | Frozen Value | Verification Method |
|---|---|---|
| **Model Version** | `resnet50_fusion_v1.0.0` | Header metadata check |
| **Canonical Checkpoint Path** | `aiml/models_weights/resnet50_fusion_v1.0.0.pt` | File existence |
| **Canonical Checkpoint Size** | `294,010,890 bytes` ($280.39\text{ MB}$) | Byte count |
| **SHA-256 Digest** | `856b387c9302bd681290c005ac95f53ba421c0148f6c0ab6d8b5d6c9bb019334` | `hashlib.sha256` |
| **Feature Contract Version** | `feature_v1.0` | Schema validator |
| **Preprocessing Version** | `rgb_512_imagenet_v1` | Preprocessor identifier |
| **Calibrated Operating Threshold ($\tau^*$)** | **`0.19`** ($0.1900$) | Empirical validation ROC sweep |
| **Training Master Seed** | `42` | `torch.manual_seed(42)` |
| **Checkpoint Best Epoch** | `5` | Validation loss minimum ($0.1279$) |
| **Automated Test Count** | `29 / 29 Passed (100%)` | Pytest test suite |

---

## 2. Frozen Multimodal Architecture Contract

$$\mathbf{v}_{cnn} = \text{ResNet-50 GAP}(\mathbf{x}_{512 \times 512}) \in \mathbb{R}^{2048}$$
$$\mathbf{v}_{clin} = \text{ReLU}(\mathbf{W}_{clin} \mathbf{F}_{norm} + \mathbf{b}_{clin}) \in \mathbb{R}^{32} \quad (\mathbf{F}_{norm} \in \mathbb{R}^{16})$$
$$\mathbf{z} = [\mathbf{v}_{cnn} \,\|\, \mathbf{v}_{clin}] \in \mathbf{\mathbb{R}^{2080}} \quad \text{\textbf{(IMMUTABLE FUSION CONTRACT)}}$$
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \text{Dropout}_{0.4}(\mathbf{z}) + \mathbf{b}_1) \in \mathbb{R}^{128}$$
$$\mathbf{logits} = \mathbf{W}_2 \text{Dropout}_{0.2}(\mathbf{h}_1) + \mathbf{b}_2 \in \mathbb{R}^{5}$$
$$\mathbf{P} = \text{Softmax}(\mathbf{logits}) = [P_0, P_1, P_2, P_3, P_4]$$

---

## 3. Authoritative 16-D Feature Ordering Contract ($\mathbf{F} \in \mathbb{R}^{16}$)

| Index | Canonical Feature Key | Scale / Unit | Frozen Pathological Description |
|:---:|---|:---:|---|
| `F[0]` | `ma_count_st` | Count $\ge 0$ | Superior-Temporal microaneurysm count |
| `F[1]` | `ma_count_sn` | Count $\ge 0$ | Superior-Nasal microaneurysm count |
| `F[2]` | `ma_count_it` | Count $\ge 0$ | Inferior-Temporal microaneurysm count |
| `F[3]` | `ma_count_in` | Count $\ge 0$ | Inferior-Nasal microaneurysm count |
| `F[4]` | `hem_area_st` | $\text{px}^2 \ge 0$ | Superior-Temporal intraretinal hemorrhage area |
| `F[5]` | `hem_area_sn` | $\text{px}^2 \ge 0$ | Superior-Nasal intraretinal hemorrhage area |
| `F[6]` | `hem_area_it` | $\text{px}^2 \ge 0$ | Inferior-Temporal intraretinal hemorrhage area |
| `F[7]` | `hem_area_in` | $\text{px}^2 \ge 0$ | Inferior-Nasal intraretinal hemorrhage area |
| `F[8]` | `vessel_density_st` | $\%$ $[0, 100]$ | Superior-Temporal Frangi vascular density |
| `F[9]` | `vessel_density_sn` | $\%$ $[0, 100]$ | Superior-Nasal Frangi vascular density |
| `F[10]`| `vessel_density_it` | $\%$ $[0, 100]$ | Inferior-Temporal Frangi vascular density |
| `F[11]`| `vessel_density_in` | $\%$ $[0, 100]$ | Inferior-Nasal Frangi vascular density |
| `F[12]`| `total_microaneurysms` | Count $\ge 0$ | Total microaneurysms ($\sum F[0..3]$) |
| `F[13]`| `total_hemorrhage_area_px` | $\text{px}^2 \ge 0$ | Total hemorrhage area ($\sum F[4..7]$) |
| `F[14]`| `total_exudates_count` | Count $\ge 0$ | Discrete hard exudate lesion count (CIE $L^*a^*b^*$) |
| `F[15]`| `min_exudate_distance_to_fovea_px` | $\text{px} \ge 0$ | Min distance to fovea center. **Sentinel**: If $F[14] == 0$, $F[15] = 724.08\text{ px}$. |

---

## 4. Frozen Preprocessing & Normalization Statistics

### Image Standardization:
- Resolution: $512 \times 512$ pixels (Bilinear interpolation)
- Channels: 3-Channel RGB
- ImageNet Mean ($\boldsymbol{\mu}_{img}$): `[0.485, 0.456, 0.406]`
- ImageNet Std ($\boldsymbol{\sigma}_{img}$): `[0.229, 0.224, 0.225]`

### Clinical Feature Normalization:
$$\mathbf{F}_{norm} = \frac{\mathbf{F} - \boldsymbol{\mu}_{train}}{\boldsymbol{\sigma}_{train} + 10^{-6}}$$
- $\boldsymbol{\mu}_{train} = [4.83, 4.80, 4.61, 5.04, 43.34, 40.97, 43.68, 44.28, 14.22, 14.18, 14.12, 14.17, 19.29, 172.26, 8.13, 496.99]$
- $\boldsymbol{\sigma}_{train} = [7.36, 7.30, 6.69, 7.59, 82.56, 75.26, 82.09, 82.61, 1.14, 1.21, 1.10, 1.14, 27.89, 312.10, 13.20, 281.31]$

---

## 5. Dataset Partitioning & Zero-Leakage Guarantee

- **Dataset Identifier**: `Patient-Stratified-Retinal-Cohort-v1`
- **Total Unique Patients**: 100 patients ($200$ fundus scans, $2$ eyes per patient)
- **Training Partition (70%)**: 140 scans (70 unique patients)
- **Validation Partition (15%)**: 30 scans (15 unique patients)
- **Held-Out Test Partition (15%)**: 30 scans (15 unique patients)
- **Zero Patient Overlap**: Verified across all 3 partitions ($\text{Leakage} = 0$).

---

## 6. Training Hyperparameters

- **Optimizer**: `AdamW` ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-2}$)
- **Learning Rate Scheduler**: `CosineAnnealingLR` ($T_{max} = 5$, $\eta_{min} = 10^{-6}$)
- **Loss Objective**: `MultiClassFocalLoss` ($\gamma = 2.0$, $\boldsymbol{\alpha} = [0.5185, 0.9333, 1.0769, 2.0000, 1.7500]$)
- **Batch Size**: 8
- **Augmentation**: Rotation $[-180^\circ, 180^\circ]$, random flips ($p=0.5$), scaling $[0.9, 1.1]$, mild ColorJitter ($\pm 10\%$ brightness, $\pm 15\%$ contrast). Shearing prohibited.

---

## 7. Frozen Operational Operating Threshold ($\tau^*$)

$$\text{Score}_{ref} = P_2 + P_3 + P_4$$
$$\text{is\_referable} = \begin{cases} \text{true}, & \text{Score}_{ref} \ge 0.19 \\ \text{false}, & \text{Score}_{ref} < 0.19 \end{cases}$$

---

## 8. Frozen Evaluation Results Summary

| Cohort | Metric | Value |
|---|---|:---:|
| **Validation Cohort** ($N=30$) | Multi-Class Accuracy | **`76.67%`** |
| | Macro F1 Score | **`0.7046`** |
| | Referable Sensitivity (at $\tau^* = 0.19$) | **`100.00%`** |
| | Referable Specificity (at $\tau^* = 0.19$) | **`100.00%`** |
| | Referable ROC-AUC | **`1.0000`** |
| **Held-Out Test Cohort** ($N=30$) | Multi-Class Accuracy | **`86.67%`** |
| | Macro F1 Score | **`0.7667`** |
| | Weighted F1 Score | **`0.8111`** |
| | Quadratic Weighted Kappa (QWK) | **`0.9703`** |
| | Macro One-vs-Rest ROC-AUC | **`0.9708`** |
| | Referable Sensitivity (at $\tau^* = 0.19$) | **`100.00%`** |
| | Referable Specificity (at $\tau^* = 0.19$) | **`92.86%`** |
| | Referable ROC-AUC | **`1.0000`** |
| | False Negatives (Referable Cases Missed) | **`0`** |

---

## 9. Latency Profile (Host CPU, 25 Benchmark Iterations)

- **Total Forward Pass (Without Explainability)**: **$358.48 \pm 48.28\text{ ms}$**
- **Grad-CAM Generation & Overlay**: **$381.87 \pm 44.05\text{ ms}$**
- **Total End-to-End Latency (With Grad-CAM)**: **$740.36 \pm 71.53\text{ ms}$**
