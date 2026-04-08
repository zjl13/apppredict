from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class TorchvisionImageBackbone(nn.Module):
    """Extract pooled image features from a lightweight torchvision backbone."""

    def __init__(self, backbone_name: str = 'mobilenet_v3_small') -> None:
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'mobilenet_v3_small':
            try:
                model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            except Exception:
                model = models.mobilenet_v3_small(weights=None)
            self.features = model.features
            self.pool = model.avgpool
            self.output_dim = 576
            return

        if backbone_name == 'resnet18':
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                model = models.resnet18(weights=None)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.pool = nn.Identity()
            self.output_dim = 512
            return

        raise ValueError(f'Unsupported backbone: {backbone_name}')

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features)
        return torch.flatten(pooled, 1)


class TreeEncoder(nn.Module):
    """Encode hashed UI tree features into a compact vector."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        encoder_type: str = 'simple',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == 'simple':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            return

        if encoder_type == 'deep':
            intermediate_dim = max(hidden_dim * 2, 256)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            return

        raise ValueError(f'Unsupported tree encoder type: {encoder_type}')

    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(tree_features)


class MultimodalSceneClassifier(nn.Module):
    """Lightweight dual-branch model for image + UI tree classification."""

    def __init__(
        self,
        backbone_name: str = 'mobilenet_v3_small',
        image_dim: int = 256,
        tree_input_dim: int = 512,
        tree_dim: int = 128,
        fusion_dim: int = 256,
        num_classes: int = 22,
        fusion_mode: str = 'concat',
        tree_encoder_type: str = 'simple',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.image_backbone = TorchvisionImageBackbone(backbone_name)

        if fusion_mode == 'concat':
            self.image_proj = nn.Linear(self.image_backbone.output_dim, image_dim)
            self.tree_encoder = TreeEncoder(
                tree_input_dim,
                tree_dim,
                encoder_type=tree_encoder_type,
                dropout=dropout,
            )
            self.fusion = nn.Sequential(
                nn.Linear(image_dim + tree_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(fusion_dim, num_classes)
            return

        if fusion_mode == 'gated':
            self.image_proj = nn.Sequential(
                nn.Linear(self.image_backbone.output_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.tree_encoder = TreeEncoder(
                tree_input_dim,
                fusion_dim,
                encoder_type=tree_encoder_type,
                dropout=dropout,
            )
            self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(fusion_dim, num_classes)
            return

        raise ValueError(f'Unsupported fusion mode: {fusion_mode}')

    def forward(self, images: torch.Tensor, tree_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_proj(self.image_backbone(images))
        tree_features = self.tree_encoder(tree_features)

        if self.fusion_mode == 'gated':
            gate = torch.sigmoid(self.gate(torch.cat([image_features, tree_features], dim=-1)))
            gated_mix = gate * image_features + (1.0 - gate) * tree_features
            fusion_input = torch.cat([gated_mix, torch.abs(image_features - tree_features)], dim=-1)
        else:
            fusion_input = torch.cat([image_features, tree_features], dim=-1)

        fused = self.fusion(fusion_input)
        logits = self.classifier(fused)
        return logits, fused
