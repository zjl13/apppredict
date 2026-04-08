from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class TorchvisionImageBackbone(nn.Module):
    """Extract pooled image features from a torchvision backbone."""

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

        if backbone_name == 'resnet34':
            try:
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            except Exception:
                model = models.resnet34(weights=None)
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
    """Dual-branch model for image + UI tree classification."""

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
        use_aux_heads: bool = False,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.use_aux_heads = use_aux_heads
        self.branch_dropout = max(float(branch_dropout), 0.0)
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
            aux_image_dim = image_dim
            aux_tree_dim = tree_dim
        elif fusion_mode == 'gated':
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
            aux_image_dim = fusion_dim
            aux_tree_dim = fusion_dim
        else:
            raise ValueError(f'Unsupported fusion mode: {fusion_mode}')

        self.classifier = nn.Linear(fusion_dim, num_classes)
        if self.use_aux_heads:
            self.image_aux_classifier = nn.Linear(aux_image_dim, num_classes)
            self.tree_aux_classifier = nn.Linear(aux_tree_dim, num_classes)

    def _apply_branch_dropout(
        self,
        image_features: torch.Tensor,
        tree_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.branch_dropout <= 0.0:
            return image_features, tree_features

        keep_prob = max(1.0 - self.branch_dropout, 1e-6)
        image_mask = (torch.rand((image_features.size(0), 1), device=image_features.device) < keep_prob).float()
        tree_mask = (torch.rand((tree_features.size(0), 1), device=tree_features.device) < keep_prob).float()
        both_dropped = (image_mask + tree_mask) == 0
        image_mask = torch.where(both_dropped, torch.ones_like(image_mask), image_mask)
        tree_mask = torch.where(both_dropped, torch.ones_like(tree_mask), tree_mask)
        return image_features * (image_mask / keep_prob), tree_features * (tree_mask / keep_prob)

    def forward(
        self,
        images: torch.Tensor,
        tree_features: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        image_features = self.image_proj(self.image_backbone(images))
        tree_features = self.tree_encoder(tree_features)

        aux_outputs: dict[str, torch.Tensor] = {}
        if self.use_aux_heads:
            aux_outputs = {
                'image_logits': self.image_aux_classifier(image_features),
                'tree_logits': self.tree_aux_classifier(tree_features),
            }

        fusion_image, fusion_tree = self._apply_branch_dropout(image_features, tree_features)

        if self.fusion_mode == 'gated':
            gate = torch.sigmoid(self.gate(torch.cat([fusion_image, fusion_tree], dim=-1)))
            gated_mix = gate * fusion_image + (1.0 - gate) * fusion_tree
            fusion_input = torch.cat([gated_mix, torch.abs(fusion_image - fusion_tree)], dim=-1)
        else:
            fusion_input = torch.cat([fusion_image, fusion_tree], dim=-1)

        fused = self.fusion(fusion_input)
        logits = self.classifier(fused)
        if return_aux:
            return logits, fused, aux_outputs
        return logits, fused
