# Retinal AI: Comprehensive Final Model Evaluation Report

**Project**: Retinal AI Diabetic Retinopathy Screening System  
**Hackathon Target**: Smart India Hackathon 2026 (PS ID: 26038)  
**Evaluation Protocol**: Untouched Held-Out Test Evaluation & Validation ROC Threshold Sweep  
**Model Architecture**: Multimodal ResNet-50 GAP (2048-D) + 16-D Clinical Projection (32-D) $\to$ 2080-D Fusion Head  
**Model Checkpoint**: `aiml/models_weights/resnet50_fusion_v1.0.0.pt`  
**Execution Timestamp**: 2026-09-17 10:58:22  

---

## 1. Dataset Partitioning & Cohort Summary

The evaluation was conducted on the patient-stratified cohort with **strict patient-level separation** (zero patient overlap between splits):

| Cohort | Patient Count | Fundus Scans | Class Distribution (Grades 0 / 1 / 2 / 3 / 4) | Role in Evaluation |
|---|:---:|:---:|:---:|---|
| **Training** | 70 | 140 | 54 / 30 / 26 / 14 / 16 | Parameter backprop & feature normalization ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) calculation |
| **Validation** | 15 | 30 | 10 / 6 / 6 / 4 / 4 | Model checkpoint selection & ROC threshold sweep ($\tau^*$) |
| **Held-Out Test** | 15 | 30 | 10 / 4 / 6 / 6 / 4 | **Completely locked untouched final evaluation** |
| **Total Cohort** | **100** | **200** | **74 / 40 / 38 / 24 / 24** | $100\%$ zero patient leakage verified |

---

## 2. Operational Threshold Calibration (Validation Cohort Only)

The operating point $\tau^*$ was calibrated strictly on the validation cohort ($	au \in [0.10, 0.90]$ with step $0.01$):
- **Clinical Sensitivity Constraint**: $\text{Sensitivity}(\tau) \ge 0.900$ ($90.0\%$)
- **Selection Optimization Rule**: $\tau^* = \arg\max_{\tau} \{\text{Specificity}(\tau) \mid \text{Sensitivity}(\tau) \ge 0.900\}$
- **Calibrated Operating Threshold ($\tau^*$)**: **`0.1900`**
- **Achieved Validation Sensitivity**: **`100.00%`**
- **Achieved Validation Specificity**: **`100.00%`**

---

## 3. Comprehensive Multiclass Evaluation (Held-Out Test Set)

Evaluated on the locked test cohort ($N=30$ independent fundus scans):

| Multiclass Metric | Achieved Test Value | Project Target | Compliance Status |
|---|:---:|:---:|:---:|
| **Overall Accuracy** | **`86.67%`** | $> 80.0\%$ | **PASS** |
| **Macro F1 Score** | **`0.7667`** | $> 0.700$ | **PASS** |
| **Weighted F1 Score** | **`0.8111`** | $> 0.750$ | **PASS** |
| **Quadratic Weighted Kappa (QWK)** | **`0.9703`** | $> 0.750$ | **PASS** |
| **Macro One-vs-Rest ROC-AUC** | **`0.9708`** | $> 0.850$ | **PASS** |

### Per-Class Detailed Breakdown:
| ICDR Grade | Severity Label | Support | Precision | Recall | F1 Score | OvR ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **0** | No DR | 10 | 0.7143 | 1.0000 | 0.8333 | 0.95 |
| **1** | Mild NPDR | 4 | 0.0000 | 0.0000 | 0.0000 | 0.9038 |
| **2** | Moderate NPDR | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0 |
| **3** | Severe NPDR | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0 |
| **4** | Proliferative DR | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0 |

### 5x5 Confusion Matrix:
```
                Predicted:
           0     1     2     3     4
True 0:   10     0     0     0     0
True 1:    4     0     0     0     0
True 2:    0     0     6     0     0
True 3:    0     0     0     6     0
True 4:    0     0     0     0     4
```

---

## 4. Referable DR Triage Performance

Referable DR is formulated as $\text{Score}_{ref} = P_2 + P_3 + P_4$ evaluated at the frozen validation threshold $\tau^* = 0.19$:

| Metric | Validation Cohort | Held-Out Test Cohort | Target Constraint | Status |
|---|:---:|:---:|:---:|:---:|
| **Referable Sensitivity (TPR)** | **`100.00%`** | **`100.00%`** | $\ge 90.0\%$ | **PASS** |
| **Referable Specificity (TNR)** | **`100.00%`** | **`92.86%`** | $\ge 80.0\%$ | **PASS** |
| **Referable ROC-AUC** | **`1.0000`** | **`1.0000`** | $> 0.900$ | **PASS** |
| **Positive Predictive Value (PPV)** | **`100.00%`** | **`94.12%`** | — | — |
| **Negative Predictive Value (NPV)** | **`100.00%`** | **`100.00%`** | — | — |
| **Referable F1 Score** | **`1.0000`** | **`0.9697`** | — | — |
| **False Negatives (Referable Missed)** | **`0`** | **`0`** | **0 Missed** | **PASS** |

---

## 5. Measured Host CPU Latency Profile (25 Benchmark Runs)

| Pipeline Sub-Phase | Mean Latency | Std Dev | 95th Percentile | Latency Budget |
|---|:---:|:---:|:---:|:---:|
| **1. Deterministic Preprocessing (512x512 + ImageNet Norm)** | `5.02 ms` | `±8.72 ms` | `4.15 ms` | $\le 10\text{ ms}$ |
| **2. ResNet-50 CNN Backbone (GAP $\to 2048$-D)** | `338.59 ms` | `±32.58 ms` | `391.96 ms` | $\le 450\text{ ms}$ |
| **3. Clinical Dense Branch ($16 \to 32$-D)** | `0.18 ms` | `±0.05 ms` | `0.25 ms` | $\le 5\text{ ms}$ |
| **4. Multimodal Fusion & Classification Head ($2080 \to 5$)** | `0.28 ms` | `±0.07 ms` | `0.43 ms` | $\le 5\text{ ms}$ |
| **5. Grad-CAM Localized Hook & JET Overlay Generation** | `370.34 ms` | `±53.32 ms` | `497.09 ms` | $\le 500\text{ ms}$ |
| **Total Forward Pass (Without Explainability)** | **`344.08 ms`** | `±35.6 ms` | `415.72 ms` | $\le 500\text{ ms}$ |
| **Total End-to-End Latency (With Grad-CAM Overlay)** | **`714.42 ms`** | `±70.42 ms` | `856.73 ms` | $\le 2500\text{ ms}$ |

---

## 6. Numerical & Architectural Contract Integrity

- **Probability Distribution Normalization**: $\sum_{i=0}^4 P_i = 1.0000 \pm 10^{-4}$ verified across $100\%$ of test samples.
- **Referable Risk Exact Formulation**: $\text{Score}_{ref} = P_2 + P_3 + P_4$ verified with zero delta.
- **Parameter Health**: **Zero NaN or Inf floating-point values** across all $644$ model weight tensors.
- **Deterministic Checkpoint Reload**: Model reloads from disk and replicates forward inference identically.
- **CPU Inference Support**: Validated fully functional on CPU without CUDA dependencies.

---

## 7. Known Failure Modes & Diagnostic Weaknesses

1. **Grade 0 vs Grade 1 Boundary Confusion**: In the test cohort, 4 Grade 1 (Mild NPDR) cases were classified as Grade 0 (No DR). This occurs because Mild NPDR presents with only isolated microaneurysms, which produce minimal visual perturbation on downsampled global CNN features. Crucially, **neither Grade 0 nor Grade 1 is referable**, meaning **zero clinical triage risk** resulted from this confusion.
2. **Missing Feature Dependency**: The 2080-D model is multimodal. If MATLAB/CV clinical features are missing (`None`), inference fails gracefully with `ERR_MATLAB_UNAVAILABLE`. A separate unifocal CNN checkpoint is required for degraded image-only fallback.
3. **Investigational Disclaimer**: This system is an engineering screening decision-support aid and has **not** received clinical certification. All diagnostic decisions require licensed clinician review.
