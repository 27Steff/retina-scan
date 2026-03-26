"""
dataset.py
Dataset de PyTorch para detección de retinopatía diabética (APTOS 2019).

Pipeline en __getitem__: preprocessing (crop→resize→CLAHE, uint8) →
augmentation si train (uint8) → normalización ImageNet (float32) → tensor (C,H,W).
La normalización va al final porque albumentations requiere uint8 y CoarseDropout
rellena con 0=negro; normalizar antes rompería eso.
Formato CSV: columnas id_code, diagnosis. Imágenes en {images_dir}/{id_code}.png.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, replace
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union

from preprocessing import RetinaPreprocessor, PreprocessConfig
from augmentation import RetinaAugmentor, AugmentationConfig, make_augmentor

PathLike = Union[str, Path]


@dataclass
class DatasetConfig:
    """
    Configuración del dataset. Embebe PreprocessConfig y AugmentationConfig.
    El campo normalize de preprocess se ignora — la normalización siempre
    ocurre después de la augmentation.
    """
    augment: bool = True
    preprocess: PreprocessConfig = field(
        default_factory=lambda: PreprocessConfig(image_size=380)
    )
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    # Mismos valores que PreprocessConfig para consistencia
    imagenet_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, ...]  = (0.229, 0.224, 0.225)


class RetinaDataset(Dataset):
    """
    Dataset PyTorch para imágenes de fondo de ojo (APTOS 2019).
    Clases de severidad: 0=No DR, 1=Leve, 2=Moderada, 3=Severa, 4=Proliferativa.
    """

    NUM_CLASSES: int = 5

    def __init__(
        self,
        csv_path: PathLike,
        images_dir: PathLike,
        config: Optional[DatasetConfig] = None,
    ):
        self.config = config or DatasetConfig()
        self.images_dir = Path(images_dir)
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        # Forzamos normalize=False aquí — la normalización ocurre DESPUÉS de augmentation
        preprocess_cfg = replace(self.config.preprocess, normalize=False)
        self.preprocessor = RetinaPreprocessor(preprocess_cfg)

        self.augmentor = (
            RetinaAugmentor(self.config.augmentation)
            if self.config.augment
            else make_augmentor("val")
        )

        # Reshape a (3,1,1) para broadcasting sobre tensor (C,H,W)
        self._mean = np.array(self.config.imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        self._std  = np.array(self.config.imagenet_std,  dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Retorna (tensor float32 (C,H,W), label int [0-4])."""
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['id_code']}.png"
        label = int(row['diagnosis'])

        img = self.preprocessor.process_path(img_path)
        img = self.augmentor.apply(img)
        tensor = self._to_normalized_tensor(img)

        return torch.from_numpy(tensor), label

    def label_counts(self) -> dict:
        """
        Distribución de clases en el dataset.
        Útil para calcular pesos de WeightedRandomSampler — APTOS tiene mayoría clase 0.
        """
        return self.df['diagnosis'].value_counts().sort_index().to_dict()

    def _to_normalized_tensor(self, img: np.ndarray) -> np.ndarray:
        """uint8 (H,W,3) → float32 (C,H,W) normalizado con stats de ImageNet."""
        tensor = self.preprocessor.to_tensor(img)
        return (tensor / 255.0 - self._mean) / self._std


def make_dataset(
    kind: str,
    csv_path: PathLike,
    images_dir: PathLike,
    **kwargs,
) -> RetinaDataset:
    """
    Factory con presets para datasets.

    kind: "train" (augment=True) | "val" (augment=False)
    **kwargs sobreescribe campos de DatasetConfig.
    """
    presets = {
        "train": DatasetConfig(augment=True),
        "val":   DatasetConfig(augment=False),
    }

    if kind not in presets:
        raise ValueError(
            f"Unknown dataset kind '{kind}'. Options: {list(presets)}"
        )

    config = replace(presets[kind], **kwargs)
    return RetinaDataset(csv_path, images_dir, config)
