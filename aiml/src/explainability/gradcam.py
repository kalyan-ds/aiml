"""
Retinal AI: Grad-CAM Explainability Generator
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Target Layer: ResNet-50 final convolutional block `layer4[2]`
Overlay: JET colormap upsampled to original fundus resolution, blended at alpha=0.45 (45% opacity)
Clinical Caveat: Grad-CAM heatmaps highlight visual discriminative regions for the model's
predicted ICDR grade and do not constitute independent histological or lesion validation.
"""

import os
import io
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
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

        # Identify target layer: layer4[2] of ResNet-50
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "layer4"):
            self.target_layer = model.layer4[2]
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            self.target_layer = model.backbone.layer4[2]
        else:
            raise ValueError("Could not automatically locate ResNet-50 layer4[2] in model.")

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where the first element is the gradient of loss w.r.t layer output
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        clinical_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate normalized 2D Grad-CAM heatmap for the specified or predicted class.
        Args:
            image_tensor: (1, 3, H, W) normalized image tensor
            clinical_tensor: (1, 16) clinical feature tensor
            target_class: Target ICDR class index (0..4). If None, uses argmax(logits).
        Returns:
            cam_map: (H, W) float32 numpy array normalized to [0.0, 1.0]
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(image_tensor, clinical_tensor)  # (1, 5)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Backward pass w.r.t target class logit
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pool the gradients
        # activations: (1, C, H_act, W_act)
        # gradients:   (1, C, H_act, W_act)
        grads = self.gradients.detach()
        acts = self.activations.detach()

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)    # (1, 1, H_act, W_act)

        # Apply ReLU to retain only positive influences
        cam = F.relu(cam)

        # Upsample to input image dimensions (512, 512)
        cam = F.interpolate(
            cam,
            size=(image_tensor.shape[2], image_tensor.shape[3]),
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
        colormap_name: str = "jet",
    ) -> Image.Image:
        """
        Blends the Grad-CAM heatmap over the original fundus image.
        Args:
            original_image: PIL RGB image
            cam_map: 2D numpy array [0, 1]
            alpha: Heatmap blend opacity (0.45 per project spec)
            colormap_name: Colormap name ('jet' per spec)
        Returns:
            blended_pil: PIL RGB Image with heatmap overlay
        """
        w, h = original_image.size
        # Resize CAM map to match original image dimensions if needed
        if cam_map.shape != (h, w):
            cam_img = Image.fromarray((cam_map * 255).astype(np.uint8))
            cam_img = cam_img.resize((w, h), resample=Image.Resampling.BILINEAR)
            cam_map = np.array(cam_img, dtype=np.float32) / 255.0

        # Apply colormap
        cmap = matplotlib.colormaps[colormap_name]
        colored_cam = cmap(cam_map)  # (H, W, 4) in [0, 1]
        colored_cam_rgb = (colored_cam[:, :, :3] * 255).astype(np.float32)

        orig_np = np.array(original_image, dtype=np.float32)

        # Blend: (1 - alpha) * original + alpha * heatmap
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
        Returns the absolute file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        blended = self.create_overlay(original_image, cam_map, alpha=alpha)
        blended.save(output_path, format="JPEG", quality=92)
        return os.path.abspath(output_path)
