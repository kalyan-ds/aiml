"""
Retinal AI: CNN Image Preprocessing Pipeline
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Deterministic CNN Preprocessing Protocol:
  1. Input: RGB fundus image (PIL Image, numpy array, or raw bytes)
  2. Strict RGB verification (convert RGBA/L/CMYK -> RGB; rejects invalid images)
  3. Spatial Resize: 512 x 512 pixels (Bicubic / Bilinear interpolation)
  4. Tensor Conversion: float32 in [0.0, 1.0]
  5. ImageNet Standardization:
     - Mean: [0.485, 0.456, 0.406]
     - Std:  [0.229, 0.224, 0.225]

Note: No CLAHE, morphology, green-channel extraction, or vessel masking is
applied inside the CNN branch to prevent double preprocessing.
"""

import io
from typing import Union, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


class FundusPreprocessor:
    """
    Deterministic ImageNet-standardized fundus image preprocessor for ResNet-50.
    """

    PREPROCESSING_VERSION = "rgb_512_imagenet_v1"
    TARGET_SIZE: Tuple[int, int] = (512, 512)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        self.transform = T.Compose([
            T.Resize(self.TARGET_SIZE, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # Converts to [0.0, 1.0] and (C, H, W)
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def load_image(self, image: Union[bytes, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load and validate raw input into a PIL RGB Image.
        """
        if isinstance(image, bytes):
            try:
                pil_img = Image.open(io.BytesIO(image))
                pil_img.load()
            except Exception as e:
                raise ValueError(f"Failed to decode image bytes: {e}")
        elif isinstance(image, Image.Image):
            pil_img = image
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0 and image.min() >= 0.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                pil_img = Image.fromarray(image, mode="L")
            elif image.ndim == 3:
                if image.shape[2] == 3:
                    pil_img = Image.fromarray(image, mode="RGB")
                elif image.shape[2] == 4:
                    pil_img = Image.fromarray(image, mode="RGBA")
                else:
                    raise ValueError(f"Unsupported numpy array channel count: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported numpy array dimension: {image.ndim}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Ensure RGB 3-channel
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return pil_img

    def __call__(
        self, image: Union[bytes, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess fundus image for ResNet-50 inference.
        Returns:
            tensor: (1, 3, 512, 512) normalized PyTorch tensor ready for model input.
            pil_rgb: 512x512 PIL RGB image (used for Grad-CAM overlay).
        """
        pil_img = self.load_image(image)
        resized_pil = pil_img.resize(self.TARGET_SIZE, resample=Image.Resampling.BILINEAR)
        tensor = self.transform(pil_img)  # (3, 512, 512)
        batch_tensor = tensor.unsqueeze(0)  # (1, 3, 512, 512)
        return batch_tensor, resized_pil
