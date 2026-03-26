"""
model.py
Clasificación de retinopatía diabética con EfficientNet-B4, transfer learning en dos fases.

Fase 1 — backbone congelado: solo entrena el head. LR alto (~1e-3), pocas épocas.
Fase 2 — fine-tuning completo: LR bajo (~1e-5) para no destruir features de ImageNet.
Output: logits crudos. CrossEntropyLoss aplica softmax internamente — no duplicar.
Para probabilidades en inferencia: torch.softmax(logits, dim=1).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)


@dataclass
class ModelConfig:
    """
    Parámetros del modelo.
    pretrained=False solo en tests para evitar descarga de pesos de red.
    freeze_backbone=True activa fase 1; llamar unfreeze_backbone() para pasar a fase 2.
    """
    num_classes: int = 5
    dropout_rate: float = 0.4
    freeze_backbone: bool = True
    pretrained: bool = True
    model_type: str = "efficientnet_b4"


class RetinaModel(nn.Module):
    """
    EfficientNet-B4 con head reemplazado para clasificación de retinopatía.
    Reemplaza Linear(1792, 1000) por Dropout + Linear(1792, num_classes).
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = self._build_backbone()

        if self.config.freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits de shape (B, num_classes)."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """
        Congela todos los parámetros excepto el classifier (fase 1).
        Úsalo al inicio para aprovechar pesos de ImageNet sin destruirlos.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Descongela todo el backbone para fine-tuning completo (fase 2).
        Usar LR bajo (~1e-5) para no destruir los features aprendidos.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Cuenta parámetros del modelo; si trainable_only=True, solo los entrenables."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def _build_backbone(self) -> nn.Module:
        """Carga EfficientNet y reemplaza el head de clasificación."""
        _backbones = {
            "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
            "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
        }
        if self.config.model_type not in _backbones:
            raise ValueError(f"Backbone no soportado: '{self.config.model_type}'. Opciones: {list(_backbones)}")

        builder, weights_enum = _backbones[self.config.model_type]
        weights = weights_enum if self.config.pretrained else None
        model = builder(weights=weights)

        in_features = model.classifier[1].in_features  # 1280 para B0, 1792 para B4
        model.classifier = nn.Sequential(
            nn.Dropout(p=self.config.dropout_rate, inplace=True),
            nn.Linear(in_features, self.config.num_classes),
        )
        return model


def make_model(kind: str, **kwargs) -> RetinaModel:
    """
    Factory para crear modelos con distintos backbones.

    kind: "efficientnet_b4" | "efficientnet_b0"
    **kwargs sobreescribe campos de ModelConfig.
    """
    registry = {
        "efficientnet_b4": lambda **kw: ModelConfig(model_type="efficientnet_b4", **kw),
        "efficientnet_b0": lambda **kw: ModelConfig(model_type="efficientnet_b0", **kw),
    }

    if kind not in registry:
        raise ValueError(
            f"Unknown model kind '{kind}'. Options: {list(registry)}"
        )

    config = registry[kind](**kwargs)
    return RetinaModel(config)
