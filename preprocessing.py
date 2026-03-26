"""
preprocessing.py
Preprocesamiento de imágenes de fondo de ojo para detección de retinopatía diabética.

Técnica principal — CLAHE en espacio LAB: mejora contraste por bloques solo sobre el
canal de luminosidad, preservando los colores diagnósticos (rojo de hemorragias vs
amarillo de exudados). Normalización con stats de ImageNet requerida para pesos
preentrenados de EfficientNet.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

ImageArray = np.ndarray  # shape (H, W, 3), dtype uint8, rango [0, 255]
PathLike = Union[str, Path]


@dataclass
class PreprocessConfig:
    """Parámetros de preprocesamiento — defaults razonables para imágenes APTOS."""
    image_size: int = 512
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    remove_black_border: bool = True
    normalize: bool = True
    # Stats de ImageNet — no cambiar si se usan pesos preentrenados
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, ...]  = (0.229, 0.224, 0.225)


class RetinaPreprocessor:
    """
    Pipeline de preprocesamiento: crop de borde negro → resize → CLAHE → normalización.
    Acepta imágenes uint8 RGB; devuelve float32 si normalize=True, uint8 si False.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size,
        )

    def process_path(self, image_path: PathLike) -> ImageArray:
        """Carga y procesa una imagen desde disco (.jpg, .png, .tiff)."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {path}. "
                             "Verifica que el archivo no esté corrupto.")

        # OpenCV carga en BGR — convertimos a RGB inmediatamente
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.process_array(img)

    def process_array(self, img: ImageArray) -> ImageArray:
        """Procesa una imagen ya en memoria como numpy array RGB (H, W, 3) uint8."""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Se esperaba imagen RGB (H, W, 3). "
                             f"Recibido shape: {img.shape}")

        if self.config.remove_black_border:
            img = self._crop_black_border(img)

        img = self._resize(img)
        img = self._apply_clahe(img)

        if self.config.normalize:
            img = self._normalize(img)

        return img

    def to_tensor(self, img: ImageArray) -> "np.ndarray":
        """Transpone (H, W, C) → (C, H, W) float32 para compatibilidad con PyTorch."""
        return np.transpose(img.astype(np.float32), (2, 0, 1))

    def _crop_black_border(self, img: ImageArray) -> ImageArray:
        """
        Elimina el borde negro circular alrededor de la retina.
        Umbraliza en gris, encuentra el bounding box de píxeles > 10 y recorta.
        El fondo negro no aporta información y puede confundir al modelo.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return img

        x, y, w, h = cv2.boundingRect(coords)
        # Pequeño margen para no cortar el borde de la retina
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        return img[y:y+h, x:x+w]

    def _resize(self, img: ImageArray) -> ImageArray:
        """Interpolación cúbica — más lenta que lineal pero preserva bordes de lesiones pequeñas."""
        size = self.config.image_size
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    def _apply_clahe(self, img: ImageArray) -> ImageArray:
        """
        Aplica CLAHE solo al canal L del espacio LAB.
        Mejora contraste sin distorsionar colores — importante porque rojo vs amarillo
        tienen significados diagnósticos distintos en retina.
        """
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        l_enhanced = self._clahe.apply(l)
        img_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)

    def _normalize(self, img: ImageArray) -> np.ndarray:
        """uint8 [0, 255] → float32 [~-2.5, ~2.5] usando stats de ImageNet."""
        img_float = img.astype(np.float32) / 255.0
        mean = np.array(self.config.imagenet_mean, dtype=np.float32)
        std  = np.array(self.config.imagenet_std,  dtype=np.float32)
        return (img_float - mean) / std
