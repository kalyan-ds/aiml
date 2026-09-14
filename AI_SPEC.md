# Retinal AI: Diabetic Retinopathy Screening Subsystem Specification (AI_SPEC.md)

**Project**: Retinal AI Diabetic Retinopathy Screening System  
**Hackathon Target**: Smart India Hackathon (SIH 2026) | Problem Statement #027 (PS ID: 26038)  
**Primary Specification**: `MATLAB Retinal AI SIH Presentation Guide.pdf`  
**Primary Integration Reference**: Flutter Mobile Application (`diabetics_ophthalmologist` / `base.apk`)  
**Status**: Specification & Architectural Contract Frozen — Pending Engineering Sign-Off  

---

## 1. Subsystem Architecture Overview

The Retinal AI screening system operates as a hybrid, multimodal deep learning and computer vision pipeline designed for primary healthcare centers (PHCs). It couples deep convolutional representations with explicit handcrafted ophthalmic lesion statistics to ensure full clinical explainability and regulatory compliance.

```
                    ┌──────────────────────────────────────────────┐
                    │    Preprocessed 3-Channel Fundus Image       │
                    │         (CLAHE-Enhanced / 512x512)           │
                    └──────────────────────┬───────────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   ▼                                               ▼
     [Path A: Deep CNN Backbone]                     [Path B: Clinical Pipeline]
      Fine-Tuned Pretrained ResNet-50                 MATLAB / CV Lesion Analysis
                   │                                               │
                   ▼                                               ▼
         v_cnn ∈ R^2048                                  F_clin ∈ R^16
      (Global Average Pooling)                        (16-D Lesion Vector)
                   │                                               │
                   │                                               ▼
                   │                                  [Clinical Dense Projection]
                   │                                    Linear(16, 32) + ReLU
                   │                                               │
                   │                                               ▼
                   │                                         v_clin ∈ R^32
                   └───────────────────────┬───────────────────────┘
                                           ▼
                               [Multimodal Fusion Layer]
                                Concatenate: [v_cnn ; v_clin]
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
                                     5-Unit Softmax
                              P = [P0, P1, P2, P3, P4]
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   ▼                                               ▼
        [ICDR Grade: Levels 0–4]                        [Binary Referable Triage]
       Argmax(P) -> Severity Level                    Score_ref = P2 + P3 + P4 >= τ*
```

---

## 2. Dimensionality Contracts

| Component | Vector Notation | Dimension | Description |
|---|---|---|---|
| Deep CNN Embedding | $\mathbf{v}_{cnn}$ | $\mathbb{R}^{2048}$ | ResNet-50 global average pooling layer (`avg_pool`) |
| Clinical Feature Input | $\mathbf{F}_{clin}$ | $\mathbb{R}^{16}$ | 16-element lesion and anatomical geometry vector |
| Clinical Representation | $\mathbf{v}_{clin}$ | $\mathbb{R}^{32}$ | Dense projection of $\mathbf{F}_{clin}$ ($\text{Linear}(16, 32) + \text{ReLU}$) |
| **Fused Multimodal Latent Space** | $\mathbf{z}$ | $\mathbf{\mathbb{R}^{2080}}$ | **Exact concatenation $[\mathbf{v}_{cnn} \,\|\, \mathbf{v}_{clin}]$ — Immutable Contract** |
| Hidden Classification Layer | $\mathbf{h}_1$ | $\mathbb{R}^{128}$ | Fully connected hidden layer ($\text{Linear}(2080, 128) + \text{ReLU}$) |
| Output Logits | $\mathbf{logits}$ | $\mathbb{R}^{5}$ | 5-class unnormalized severity logits |
| Categorical Probabilities | $\mathbf{P}$ | $\mathbb{R}^{5}$ | Softmax multi-class probability vector $[P_0, P_1, P_2, P_3, P_4]$ |

> [!CAUTION]
> **IMMUTABLE CONTRACT**: The fusion dimension is strictly $2048 + 32 = 2080$. AI/ML code must never independently change these dimensions, truncate vectors, or modify the projection layer.

---

## 3. ICDR Target Classes

The classifier maps fundus images to the five validated stages of the International Clinical Diabetic Retinopathy (ICDR) scale:

| Level | Severity Name | Clinical Pathological Criteria | Action Protocol |
|---|---|---|---|
| **0** | **No DR** | Absence of any microaneurysms, hemorrhages, or exudates. Normal vascular tree. | Annual routine screening |
| **1** | **Mild NPDR** | Microaneurysms only. No other retinal lesions detected. | 6–12 month follow-up |
| **2** | **Moderate NPDR** | More than just microaneurysms, but less than severe NPDR (isolated blot hemorrhages, hard exudates, cotton-wool spots). | **Referable** — specialist evaluation within 4–8 weeks |
| **3** | **Severe NPDR** | Meets clinical **4:2:1 Rule**: severe hemorrhages in all 4 quadrants, OR venous beading in $\ge 2$ quadrants, OR prominent IRMA in $\ge 1$ quadrant. | **Referable Urgent** — specialist evaluation within 2–4 weeks |
| **4** | **Proliferative DR (PDR)** | Neovascularization (on disc or elsewhere) and/or preretinal/vitreous hemorrhage. High risk of irreversible vision loss. | **Referable Emergency** — prompt panretinal photocoagulation / anti-VEGF |

---

## 4. Referable DR Formulation & Threshold Calibration

### 4.1 Referable Risk Score
In rural screening, separating patients requiring specialist intervention from those who do not is the primary life-saving task. Referable DR is defined as $\text{ICDR Level} \ge 2$:

$$\text{Score}_{ref} = P_2 + P_3 + P_4$$

### 4.2 Calibrated Operating Point Selection
- The production decision threshold $\tau^*$ is **NEVER** fixed arbitrarily to $0.5$ or hardcoded without validation.
- An empirical ROC sweep is performed on the held-out validation set across $\tau \in [0.10, 0.90]$ with step $0.01$.
- The operating point $\tau^*$ is selected to satisfy the clinical constraint:
  $$\tau^* = \arg\max_{\tau} \{\text{Specificity}(\tau) \mid \text{Sensitivity}(\tau) \ge 0.900\}$$
- Expected range based on clinical validation: $\tau^* \in [0.32, 0.38]$.
- Production binary referral flag:
  $$\text{is\_referable} = \begin{cases} \text{true}, & \text{Score}_{ref} \ge \tau^* \\ \text{false}, & \text{Score}_{ref} < \tau^* \end{cases}$$
- The exact value of $\tau^*$ must be recorded in runtime configuration and returned in all API prediction metadata.

---

## 5. Authoritative 16-Feature Clinical Contract ($\mathbf{F}_{clin} \in \mathbb{R}^{16}$)

The 16-element clinical vector is assembled by partitioning the retina into four anatomical quadrants intersecting at the foveal center $(X_{fovea}, Y_{fovea})$:
- **ST**: Superior-Temporal
- **SN**: Superior-Nasal
- **IT**: Inferior-Temporal
- **IN**: Inferior-Nasal

```
                 Superior-Nasal (SN)   |  Superior-Temporal (ST)
                                       |
                   [Optic Disc]        |        (Fovea Center)
                 ──────────────────────┼──────────────────────
                                       |
                  Inferior-Nasal (IN)  |  Inferior-Temporal (IT)
```

### Exact Immutable Index Ordering:

| Index | Feature Key | Data Type | Units / Range | Clinical Description |
|---|---|---|---|---|
| `F[0]` | `ma_count_st` | `float32` | $[0, \infty)$ | Superior-Temporal microaneurysm count (Gaussian sub-pixel fitted) |
| `F[1]` | `ma_count_sn` | `float32` | $[0, \infty)$ | Superior-Nasal microaneurysm count |
| `F[2]` | `ma_count_it` | `float32` | $[0, \infty)$ | Inferior-Temporal microaneurysm count |
| `F[3]` | `ma_count_in` | `float32` | $[0, \infty)$ | Inferior-Nasal microaneurysm count |
| `F[4]` | `hem_area_st` | `float32` | $[0, \infty) \text{ px}^2$ | Superior-Temporal intraretinal hemorrhage area |
| `F[5]` | `hem_area_sn` | `float32` | $[0, \infty) \text{ px}^2$ | Superior-Nasal intraretinal hemorrhage area |
| `F[6]` | `hem_area_it` | `float32` | $[0, \infty) \text{ px}^2$ | Inferior-Temporal intraretinal hemorrhage area |
| `F[7]` | `hem_area_in` | `float32` | $[0, \infty) \text{ px}^2$ | Inferior-Nasal intraretinal hemorrhage area |
| `F[8]` | `vessel_density_st` | `float32` | $[0.0, 100.0] \%$ | Superior-Temporal Frangi vascular area density ratio |
| `F[9]` | `vessel_density_sn` | `float32` | $[0.0, 100.0] \%$ | Superior-Nasal Frangi vascular area density ratio |
| `F[10]` | `vessel_density_it` | `float32` | $[0.0, 100.0] \%$ | Inferior-Temporal Frangi vascular area density ratio |
| `F[11]` | `vessel_density_in` | `float32` | $[0.0, 100.0] \%$ | Inferior-Nasal Frangi vascular area density ratio |
| `F[12]` | `total_microaneurysms` | `float32` | $[0, \infty)$ | Total retinal microaneurysm count ($\sum_{q=1}^4 \text{MA}_q$) |
| `F[13]` | `total_hemorrhage_area_px` | `float32` | $[0, \infty) \text{ px}^2$ | Total hemorrhage area in pixels ($\sum_{q=1}^4 \text{Hem}_q$) |
| `F[14]` | `total_exudates_count` | `float32` | $[0, \infty)$ | Total hard exudate deposits segmented in CIE L\*a\*b\* space |
| `F[15]` | `min_exudate_distance_to_fovea_px` | `float32` | $[0, \infty) \text{ px}$ | Minimum Euclidean distance from any exudate centroid to fovea center (maculopathy indicator) |

> [!WARNING]
> **NON-REORDERING RULE**: AI/ML code must **NEVER** silently reorder, truncate, or reinterpret these features. Any deviation from this 16-element ordering breaks integration with MATLAB, the FastAPI serialization schema, and the mobile report view.

---

## 6. Model Training & Validation Protocol

### 6.1 Backbone & Weights
- **Backbone**: ResNet-50 pretrained on ImageNet-1K (`IMAGENET1K_V2`).
- Truncate final `fc` layer ($2048 \to 1000$).
- Forward image through global average pooling to obtain $v_{cnn} \in \mathbb{R}^{2048}$.

### 6.2 Class Skew Mitigation & Loss Objective
Fundus screening exhibits extreme class skew (~70% normal, <5% severe/PDR). Standard cross-entropy leads to majority class collapse.
- **Focal Loss Objective**:
  $$\mathcal{L}_{\text{focal}} = -\sum_{i=0}^4 \alpha_i (1 - p_i)^\gamma \log(p_i)$$
  where $\gamma = 2.0$, and $\alpha_i = \frac{N_{\text{total}}}{5 \cdot N_i}$ (inverse class frequency).
- **Comparison**: Compare against Weighted Cross-Entropy in all ablation experiments.

### 6.3 Augmentation Strategy
All augmentations must remain medically reasonable:
- Random rotations: $[-180^\circ, +180^\circ]$ (retinal fundus orientation is rotation-invariant).
- Horizontal and vertical flips: $p = 0.5$.
- Random scaling: $[0.9, 1.1]$ (simulates variations in handheld lens working distance).
- Mild ColorJitter (brightness $\pm 10\%$, contrast $\pm 15\%$).
- **Strictly Prohibited**: Distortions that alter vascular morphology (shearing, extreme elastic deformations).

### 6.4 Reproducibility Standard
- Master seed locked: `torch.manual_seed(42)`, `np.random.seed(42)`.
- All checkpoints versioned with schema: `model_{backbone}_{dataset}_{timestamp}.pth`.
- Accompanying JSON metadata records: exact train/val/test patient splits, hyperparameters, calibrated threshold $\tau^*$, and confusion matrix.

---

## 7. Explainability Protocol (Grad-CAM)

### 7.1 Layer Hook
- Grad-CAM targets the final convolutional layer of ResNet-50: `layer4.2.conv3` (matching MATLAB's `activation_49_relu`).

### 7.2 Gradient-Weighted Class Activation Map
- For target class $c$ (the predicted ICDR grade):
  $$\alpha_k^c = \frac{1}{U \cdot V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y_c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- Normalize map to $[0, 1]$.
- Upsample via bicubic interpolation to match original image dimensions $(H, W)$.
- Apply JET colormap and alpha-blend over original fundus image at **45% opacity** ($\alpha = 0.45$).
- Co-localize segmented lesions: Draw detected microaneurysms as green circles ($\odot$) and hard exudate contours in yellow.
- **Clinical Caveat**: Grad-CAM heatmaps serve as visual attention evidence and do not constitute legal proof of specific lesions unless confirmed by the anatomical segmentation mask.

---

## 8. Integration API Contract (FastAPI)

### 8.1 Stable Prediction Interface
```python
def predict(
    image: Union[bytes, Image.Image, np.ndarray],
    clinical_features: Optional[Union[List[float], Dict[str, float]]] = None,
    lens_diopter: float = 20.0,
) -> Dict[str, Any]:
    ...
```

### 8.2 JSON Output Schema (`/api/v1/screen`)
```json
{
  "status": "success",
  "gradeable": true,
  "error_code": "NONE",
  "icdr_grade": 2,
  "icdr_label": "Moderate NPDR",
  "probabilities": {
    "P0": 0.032,
    "P1": 0.084,
    "P2": 0.741,
    "P3": 0.118,
    "P4": 0.025
  },
  "referable_risk": 0.884,
  "threshold": 0.35,
  "is_referable": true,
  "confidence": 0.741,
  "clinical_features": {
    "ma_count_st": 8.0,
    "ma_count_sn": 3.0,
    "ma_count_it": 14.0,
    "ma_count_in": 1.0,
    "hem_area_st": 12.5,
    "hem_area_sn": 0.0,
    "hem_area_it": 48.2,
    "hem_area_in": 4.0,
    "vessel_density_st": 14.2,
    "vessel_density_sn": 11.8,
    "vessel_density_it": 16.5,
    "vessel_density_in": 13.1,
    "total_microaneurysms": 26.0,
    "total_hemorrhage_area_px": 64.7,
    "total_exudates_count": 12.0,
    "min_exudate_distance_to_fovea_px": 215.4
  },
  "cam_overlay_path": "outputs/heatmaps/cam_patient_102.jpg",
  "cam_overlay_url": "/static/heatmaps/cam_patient_102.jpg",
  "inference_latency_ms": 142,
  "model_version": "ResNet50-Hybrid-v1.0.0"
}
```

---

## 9. Error Codes & Quality Rejection Contract

| Error Code | Trigger Condition | Field Guidance / User Message |
|---|---|---|
| `ERR_BLUR_RETAKE` | Tenengrad focus score $< 20.0$ | Image out of focus. Clean condensing lens and hold steady. |
| `ERR_CORNEAL_GLARE` | Mean intensity in central 40% $> 215$ | Corneal flash reflection detected; tilt phone/lens angle slightly. |
| `ERR_LOW_LIGHT` | Mean pixel intensity $< 35$ | Illumination too weak; increase LED torch brightness. |
| `ERR_INADEQUATE_CONTRAST` | Shannon entropy $H < 4.0$ | Insufficient retinal dynamic range; re-align pupil aperture. |
| `ERR_NOT_FUNDUS` | $R_{avg} < G_{avg} \cdot 0.95 \land R_{avg} < B_{avg} \cdot 0.95$ | The captured image is not a valid retinal fundus scan. |
| `ERR_PUPIL_OCCLUDED` | Retinal disc coverage $< 75\%$ of reticle | Retina not centered in aperture; align optical pupil axis. |
