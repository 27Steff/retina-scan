"""
augmentation.py
Data augmentation para imágenes de retina durante entrenamiento.

Principio rector — colores diagnósticos: el rojo de hemorragias y el amarillo de
exudados son clínicamente distintos, por eso los cambios de tono/saturación están
prohibidos. Transformaciones geométricas son libres; brillo/contraste con rangos ±10%.
Debe aplicarse ANTES de la normalización — albumentations trabaja sobre uint8 [0, 255].
"""

import cv2
import numpy as np
import albumentations as A
from dataclasses import dataclass, replace
from typing import Optional

ImageArray = np.ndarray  # shape (H, W, 3), dtype uint8, rango [0, 255]


@dataclass
class AugmentationConfig:
    """
    Parámetros de aumentación, calibrados para imágenes médicas de retina.
    Los parámetros de color tienen rangos conservadores a propósito —
    cambiarlos requiere justificación clínica, no solo experimental.
    """
    # Geométricas — seguras, no afectan colores
    flip_prob: float = 0.5
    rotation_limit: int = 180
    rotation_prob: float = 0.8

    # Color — rangos pequeños para preservar diagnóstico
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    color_prob: float = 0.5

    # Distorsión elástica — simula artefactos de cámara
    elastic_prob: float = 0.3
    elastic_alpha: float = 30.0
    elastic_sigma: float = 3.0

    # Coarse dropout — simula oclusiones y sombras de lente
    coarse_dropout_prob: float = 0.3
    coarse_dropout_holes: int = 8
    coarse_dropout_height: int = 32
    coarse_dropout_width: int = 32


class RetinaAugmentor:
    """
    Pipeline de aumentación para imágenes de retina.
    Acepta y devuelve uint8 [0, 255] — la normalización a float32 es
    responsabilidad de RetinaPreprocessor.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._transform = self._build_pipeline()

    def apply(self, img: ImageArray) -> ImageArray:
        """Aplica aumentación a una imagen RGB uint8 (H, W, 3)."""
        self._validate(img)
        result = self._transform(image=img)
        return result["image"]

    def _validate(self, img: ImageArray) -> None:
        """Lanza ValueError si la imagen no tiene el formato esperado."""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Se esperaba imagen RGB (H, W, 3). Recibido shape: {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(
                f"Se esperaba dtype uint8. Recibido: {img.dtype}. "
                "La aumentación debe aplicarse ANTES de la normalización."
            )

    def _build_pipeline(self) -> A.Compose:
        """
        Construye el pipeline de albumentations. Los transforms solo se agregan
        cuando su probabilidad es > 0, lo que mantiene el pipeline de validación
        vacío (identidad) sin condiciones especiales en apply().
        """
        c = self.config
        transforms = []

        if c.flip_prob > 0:
            transforms.append(A.HorizontalFlip(p=c.flip_prob))
            transforms.append(A.VerticalFlip(p=c.flip_prob))

        # Rotación libre hasta 180° — las cámaras de fondo de ojo no tienen orientación fija
        if c.rotation_limit > 0 and c.rotation_prob > 0:
            transforms.append(A.Rotate(
                limit=c.rotation_limit,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=c.rotation_prob,
            ))

        if (c.brightness_limit > 0 or c.contrast_limit > 0) and c.color_prob > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=c.brightness_limit,
                contrast_limit=c.contrast_limit,
                p=c.color_prob,
            ))

        if c.elastic_prob > 0:
            transforms.append(A.ElasticTransform(
                alpha=c.elastic_alpha,
                sigma=c.elastic_sigma,
                p=c.elastic_prob,
            ))

        if c.coarse_dropout_prob > 0:
            transforms.append(A.CoarseDropout(
                num_holes_range=(1, c.coarse_dropout_holes),
                hole_height_range=(8, c.coarse_dropout_height),
                hole_width_range=(8, c.coarse_dropout_width),
                fill=0,
                p=c.coarse_dropout_prob,
            ))

        return A.Compose(transforms)


def make_augmentor(kind: str, **kwargs) -> RetinaAugmentor:
    """
    Factory con presets para augmentors.

    kind:
        "train" — aumentación completa
        "light" — solo geométricas suaves, para test-time augmentation (TTA)
        "val"   — identidad, sin cambios
    **kwargs sobreescribe cualquier campo de AugmentationConfig.
    """
    presets = {
        "train": AugmentationConfig(),
        "light": AugmentationConfig(
            elastic_prob=0.0,
            coarse_dropout_prob=0.0,
            brightness_limit=0.05,
            contrast_limit=0.05,
            rotation_limit=45,
        ),
        "val": AugmentationConfig(
            flip_prob=0.0,
            rotation_limit=0,
            rotation_prob=0.0,
            brightness_limit=0.0,
            contrast_limit=0.0,
            color_prob=0.0,
            elastic_prob=0.0,
            coarse_dropout_prob=0.0,
        ),
    }

    if kind not in presets:
        raise ValueError(
            f"Unknown augmentor kind '{kind}'. Options: {list(presets)}"
        )

    config = replace(presets[kind], **kwargs)
    return RetinaAugmentor(config)
