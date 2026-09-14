"""
Retinal AI: Fundus Dataset Loader with Medically Validated Augmentations
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Permitted Augmentations:
  - Random rotations: [-180, +180] deg (fundus orientation is rotation-invariant)
  - Horizontal and vertical flips (p = 0.5)
  - Random scale: [0.9, 1.1] (simulates condensing lens focal distance variation)
  - Mild ColorJitter: brightness +/- 10%, contrast +/- 15%
  - Strictly prohibited: Shearing, elastic deformation, or lesion-altering morphology.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from aiml.src.preprocessing.image_transforms import FundusPreprocessor
from aiml.src.features.contract_validator import FeatureContractValidator


def get_medical_train_transforms() -> T.Compose:
    """
    Returns data augmentation pipeline strictly obeying medical invariant rules.
    """
    return T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomRotation(degrees=(-180, 180), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.10, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=FundusPreprocessor.IMAGENET_MEAN, std=FundusPreprocessor.IMAGENET_STD),
    ])


def get_medical_val_transforms() -> T.Compose:
    """
    Deterministic validation/test preprocessing pipeline.
    """
    return T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=FundusPreprocessor.IMAGENET_MEAN, std=FundusPreprocessor.IMAGENET_STD),
    ])


class RetinalFundusDataset(Dataset):
    """
    PyTorch Dataset pairing fundus images with canonical 16-D clinical feature vectors
    and 5-class ICDR severity labels.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        is_training: bool = False,
    ):
        """
        Args:
            samples: List of dicts, each containing:
                     - 'image': PIL Image, image file path, or numpy array
                     - 'clinical_features': 16-element list or dict
                     - 'label': int (0..4) ICDR grade
                     - 'patient_id': Optional[str] for patient-level grouping
            is_training: If True, applies training augmentations.
        """
        self.samples = samples
        self.is_training = is_training
        self.transform = get_medical_train_transforms() if is_training else get_medical_val_transforms()
        self.preprocessor = FundusPreprocessor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        # Load image
        raw_img = sample["image"]
        if isinstance(raw_img, str):
            pil_img = Image.open(raw_img).convert("RGB")
        elif isinstance(raw_img, Image.Image):
            pil_img = raw_img.convert("RGB")
        elif isinstance(raw_img, np.ndarray):
            pil_img = self.preprocessor.load_image(raw_img)
        else:
            raise TypeError(f"Unsupported image type in sample: {type(raw_img)}")

        image_tensor = self.transform(pil_img)

        # Validate and convert clinical features
        raw_features = sample["clinical_features"]
        ordered_feats, _ = FeatureContractValidator.validate_and_serialize(raw_features)
        clinical_tensor = torch.tensor(ordered_feats, dtype=torch.float32)

        label = int(sample["label"])
        return image_tensor, clinical_tensor, label
