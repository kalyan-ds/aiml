"""
Retinal AI: High-Performance Grad-CAM Explainability Generator
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Optimizations for CPU Screening:
  1. Local Head Backpropagation: Isolates gradient flow to the classification head
     and average pooling layer directly w.r.t layer4 activations, eliminating 40+
     convolutional backward passes through ResNet-50.
  2. Vectorized JET Colormap LUT: Uses a precomputed 256-level RGB lookup table for
     instantaneous O(1) alpha blending, bypassing slow matplotlib per-pixel mapping.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMGenerator:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    on the ResNet-50 backbone for the predicted ICDR severity grade.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Precompute vectorized JET colormap lookup table (256, 3)
        jet = matplotlib.colormaps["jet"]
        self._jet_lut = (jet(np.linspace(0, 1, 256))[:, :3] * 255.0).astype(np.float32)

    def generate(
        self,
        image_tensor: torch.Tensor,
        clinical_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate normalized 2D Grad-CAM heatmap for the specified or predicted class.
        Uses targeted backward propagation w.r.t layer4 spatial activations.
        """
        # Step 1: Forward CNN backbone up to layer4 activations without graph overhead
        with torch.no_grad():
            x = self.model.conv1(image_tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            raw_activations = self.model.layer4(x)  # (1, 2048, H_act, W_act)

        # Step 2: Attach gradient tracking only to layer4 activations
        activations = raw_activations.detach().clone().requires_grad_(True)
        pooled = self.model.avgpool(activations)
        v_cnn = torch.flatten(pooled, 1)  # (1, 2048)

        # Step 3: Clinical branch (deterministic evaluation)
        with torch.no_grad():
            v_clin = self.model.clinical_branch(clinical_tensor)  # (1, 32)

        # Step 4: Fusion and classification head
        z = torch.cat([v_cnn, v_clin], dim=1)  # (1, 2080)
        logits = self.model.classifier(z)       # (1, 5)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Step 5: Fast backward pass exclusively through head to activations
        self.model.classifier.zero_grad()
        score = logits[0, target_class]
        score.backward()

        grads = activations.grad.detach()  # (1, 2048, H_act, W_act)
        acts = activations.detach()        # (1, 2048, H_act, W_act)

        # Global average pool the gradients across spatial dimensions
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)    # (1, 1, H_act, W_act)

        # Apply ReLU to isolate positive influences
        cam = F.relu(cam)

        # Bilinear upsample to fundus frame dimensions
        h_in, w_in = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(
            cam,
            size=(h_in, w_in),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    def create_overlay(
        self,
        original_image: Image.Image,
        cam_map: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Blends the Grad-CAM heatmap over the original fundus image using
        vectorized JET LUT lookup at alpha=0.45 (45% opacity).
        """
        w, h = original_image.size
        if cam_map.shape != (h, w):
            cam_img = Image.fromarray((cam_map * 255).astype(np.uint8))
            cam_img = cam_img.resize((w, h), resample=Image.Resampling.BILINEAR)
            cam_map = np.array(cam_img, dtype=np.float32) / 255.0

        # Fast Vectorized LUT Mapping: O(1) array indexing
        indices = np.clip((cam_map * 255.0).astype(np.int32), 0, 255)
        colored_cam_rgb = self._jet_lut[indices]  # (H, W, 3) in float32

        orig_np = np.array(original_image, dtype=np.float32)

        # Linear alpha blend: (1 - alpha) * orig + alpha * cam
        blended = (1.0 - alpha) * orig_np + alpha * colored_cam_rgb
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def save_overlay(
        self,
        original_image: Image.Image,
        cam_map: np.ndarray,
        output_path: str,
        alpha: float = 0.45,
    ) -> str:
        """
        Generates and saves the blended heatmap overlay to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        blended = self.create_overlay(original_image, cam_map, alpha=alpha)
        blended.save(output_path, format="JPEG", quality=90)
        return os.path.abspath(output_path)
