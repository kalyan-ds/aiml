"""
Retinal AI: Multimodal Fusion Model Architecture (ResNet-50 + 16-D Clinical Branch)
Project: Smart India Hackathon 2026 - Retinal AI Diabetic Retinopathy Screening
Primary Specification: MATLAB Retinal AI SIH Presentation Guide.pdf & AI_SPEC.md

Frozen Latent Dimensions:
  - CNN Backbone: Pretrained ResNet-50 GAP -> v_cnn in R^2048
  - Clinical Branch: BatchNorm1d(16) -> Linear(16, 32) + ReLU -> v_clin in R^32
  - Fused Representation: z = [v_cnn ; v_clin] in R^2080  (IMMUTABLE CONTRACT)
  - Classification Head: Dropout(0.4) -> Linear(2080, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 5)
  - Output: 5 raw ICDR severity logits
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Optional


class RetinalFusionModel(nn.Module):
    """
    Multimodal Retinal Fusion Classifier combining deep fundus representations
    with canonical 16-D handcrafted clinical lesion vectors.
    """

    ARCHITECTURE_VERSION = "resnet50_fusion_v1.0.0"
    FUSION_DIM = 2080
    CNN_EMBEDDING_DIM = 2048
    CLINICAL_INPUT_DIM = 16
    CLINICAL_EMBEDDING_DIM = 32
    NUM_CLASSES = 5

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate_fusion: float = 0.4,
        dropout_rate_head: float = 0.2,
        use_clinical_batchnorm: bool = True,
    ):
        super().__init__()

        # --- CNN Backbone (ResNet-50) ---
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        # Retain convolutional layers and global average pooling; drop original fc
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool

        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4  # Hook target for Grad-CAM: layer4[2]
        self.avgpool = self.backbone.avgpool

        # --- Clinical Feature Branch (16 -> 32) ---
        clinical_layers = []
        if use_clinical_batchnorm:
            clinical_layers.append(nn.BatchNorm1d(self.CLINICAL_INPUT_DIM))
        clinical_layers.extend([
            nn.Linear(self.CLINICAL_INPUT_DIM, self.CLINICAL_EMBEDDING_DIM),
            nn.ReLU(inplace=True),
        ])
        self.clinical_branch = nn.Sequential(*clinical_layers)

        # --- Multimodal Fusion & Classification Head (2080 -> 128 -> 5) ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate_fusion),
            nn.Linear(self.FUSION_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate_head),
            nn.Linear(128, self.NUM_CLASSES),
        )

    def extract_cnn_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spatial feature map and 2048-D global average pooled embedding.
        Returns:
            activations: Spatial feature map from layer4 (B, 2048, 16, 16 for 512x512 input)
            v_cnn: Flattened 2048-D embedding (B, 2048)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        activations = self.layer4(x)

        pooled = self.avgpool(activations)
        v_cnn = torch.flatten(pooled, 1)  # (B, 2048)
        return activations, v_cnn

    def forward(
        self,
        image: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            image: Preprocessed RGB fundus tensor (B, 3, 512, 512)
            clinical_features: Canonical 16-D clinical feature vector (B, 16)
        Returns:
            logits: Raw unnormalized 5-class logits (B, 5)
        """
        if clinical_features.shape[1] != self.CLINICAL_INPUT_DIM:
            raise ValueError(
                f"Clinical feature dimension mismatch. Expected {self.CLINICAL_INPUT_DIM}, "
                f"got {clinical_features.shape[1]}"
            )

        _, v_cnn = self.extract_cnn_features(image)  # (B, 2048)
        v_clin = self.clinical_branch(clinical_features)  # (B, 32)

        # Multimodal fusion: exact 2080-D concatenation
        z = torch.cat([v_cnn, v_clin], dim=1)  # (B, 2080)

        logits = self.classifier(z)  # (B, 5)
        return logits
