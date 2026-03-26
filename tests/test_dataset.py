"""
test_dataset.py
---------------
Tests para dataset.py

Ejecutar con:
    pytest test_dataset.py -v
"""

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from dataset import DatasetConfig, RetinaDataset, make_dataset
from preprocessing import PreprocessConfig


# Helpers y fixtures

def make_fake_retina(h: int = 380, w: int = 380, seed: int = 42) -> np.ndarray:
    """Imagen sintética de retina en BGR (para guardar con cv2.imwrite)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 10
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = [60, 80, 180]  # BGR — cv2 guarda en BGR
    img[mask] = np.clip(
        img[mask].astype(np.int32) + rng.randint(0, 20, (mask.sum(), 3)),
        0, 255,
    ).astype(np.uint8)
    return img


@pytest.fixture
def fake_dataset(tmp_path) -> tuple:
    """
    Crea un dataset falso completo en tmp_path:
      - 6 imágenes .png con ids únicos
      - CSV con id_code y diagnosis (2 ejemplos de cada clase 0-4, más 1 extra de clase 0)

    Retorna (csv_path, images_dir).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = [
        {"id_code": "img_00", "diagnosis": 0},
        {"id_code": "img_01", "diagnosis": 0},
        {"id_code": "img_02", "diagnosis": 1},
        {"id_code": "img_03", "diagnosis": 2},
        {"id_code": "img_04", "diagnosis": 3},
        {"id_code": "img_05", "diagnosis": 4},
    ]

    for i, row in enumerate(rows):
        img_bgr = make_fake_retina(seed=i)
        cv2.imwrite(str(images_dir / f"{row['id_code']}.png"), img_bgr)

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return csv_path, images_dir


@pytest.fixture
def val_dataset(fake_dataset) -> RetinaDataset:
    csv_path, images_dir = fake_dataset
    return make_dataset("val", csv_path, images_dir)


# Tests de DatasetConfig

class TestDatasetConfig:

    def test_default_augment_is_true(self):
        config = DatasetConfig()
        assert config.augment is True

    def test_default_image_size(self):
        """EfficientNet-B4 espera 380×380 por defecto."""
        config = DatasetConfig()
        assert config.preprocess.image_size == 380

    def test_custom_augment_false(self):
        config = DatasetConfig(augment=False)
        assert config.augment is False

    def test_custom_preprocess_propagates(self):
        config = DatasetConfig(preprocess=PreprocessConfig(image_size=224))
        assert config.preprocess.image_size == 224

    def test_imagenet_mean_values(self):
        config = DatasetConfig()
        assert config.imagenet_mean == (0.485, 0.456, 0.406)
        assert config.imagenet_std  == (0.229, 0.224, 0.225)


# Tests de RetinaDataset

class TestRetinaDataset:

    def test_len_matches_csv_rows(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = RetinaDataset(csv_path, images_dir)
        assert len(ds) == 6

    def test_getitem_returns_tuple(self, val_dataset):
        item = val_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_tensor_is_torch_float32(self, val_dataset):
        tensor, _ = val_dataset[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_tensor_shape_is_chw(self, val_dataset):
        """PyTorch espera (C, H, W) — canales primero."""
        tensor, _ = val_dataset[0]
        c, h, w = tensor.shape
        assert c == 3
        assert h == w  # imagen cuadrada

    def test_tensor_shape_matches_config(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        config = DatasetConfig(
            augment=False,
            preprocess=PreprocessConfig(image_size=224),
        )
        ds = RetinaDataset(csv_path, images_dir, config)
        tensor, _ = ds[0]
        assert tensor.shape == (3, 224, 224)

    def test_label_is_int(self, val_dataset):
        _, label = val_dataset[0]
        assert isinstance(label, int)

    def test_label_values_match_csv(self, fake_dataset):
        """Las etiquetas deben corresponder exactamente al CSV."""
        csv_path, images_dir = fake_dataset
        ds = make_dataset("val", csv_path, images_dir)
        expected = [0, 0, 1, 2, 3, 4]
        for idx, expected_label in enumerate(expected):
            _, label = ds[idx]
            assert label == expected_label, (
                f"Índice {idx}: esperado {expected_label}, obtenido {label}"
            )

    def test_normalized_values_in_imagenet_range(self, val_dataset):
        """
        Normalización ImageNet produce valores en aprox [-3, 3].
        Fuera de ese rango indica un error en el pipeline de normalización.
        """
        tensor, _ = val_dataset[0]
        assert tensor.min().item() > -4.0, "Valores demasiado negativos"
        assert tensor.max().item() <  4.0, "Valores demasiado positivos"

    def test_val_pipeline_is_deterministic(self, val_dataset):
        """
        Con augment=False el mismo índice debe producir el mismo tensor.
        Preprocessing + val augmentation (identidad) = pipeline determinista.
        """
        t1, _ = val_dataset[0]
        t2, _ = val_dataset[0]
        assert torch.equal(t1, t2)

    def test_normalize_is_forced_false_in_preprocess(self, fake_dataset):
        """
        Aunque el usuario pase normalize=True en PreprocessConfig,
        el dataset debe ignorarlo (normaliza después de augmentar).
        El tensor final debe estar normalizado exactamente una vez.
        """
        csv_path, images_dir = fake_dataset
        config_with_normalize = DatasetConfig(
            augment=False,
            preprocess=PreprocessConfig(normalize=True),  # debe ser ignorado
        )
        ds = RetinaDataset(csv_path, images_dir, config_with_normalize)
        tensor, _ = ds[0]
        # Si normalizara dos veces, los valores estarían muy fuera de rango
        assert tensor.min().item() > -4.0
        assert tensor.max().item() <  4.0

    def test_missing_image_raises_file_not_found(self, tmp_path):
        """Imagen faltante debe lanzar FileNotFoundError."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        csv_path = tmp_path / "labels.csv"
        pd.DataFrame([{"id_code": "no_existe", "diagnosis": 0}]).to_csv(
            csv_path, index=False
        )
        ds = RetinaDataset(csv_path, images_dir)
        with pytest.raises(FileNotFoundError):
            ds[0]

    def test_label_counts_returns_all_classes(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = RetinaDataset(csv_path, images_dir)
        counts = ds.label_counts()
        assert set(counts.keys()) == {0, 1, 2, 3, 4}

    def test_label_counts_sum_equals_len(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = RetinaDataset(csv_path, images_dir)
        assert sum(ds.label_counts().values()) == len(ds)


# Tests de make_dataset

class TestMakeDataset:

    def test_train_returns_dataset(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = make_dataset("train", csv_path, images_dir)
        assert isinstance(ds, RetinaDataset)

    def test_val_returns_dataset(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = make_dataset("val", csv_path, images_dir)
        assert isinstance(ds, RetinaDataset)

    def test_train_has_augment_true(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = make_dataset("train", csv_path, images_dir)
        assert ds.config.augment is True

    def test_val_has_augment_false(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = make_dataset("val", csv_path, images_dir)
        assert ds.config.augment is False

    def test_unknown_kind_raises(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        with pytest.raises(ValueError, match="Unknown"):
            make_dataset("test", csv_path, images_dir)

    def test_kwargs_override_augment(self, fake_dataset):
        csv_path, images_dir = fake_dataset
        ds = make_dataset("train", csv_path, images_dir, augment=False)
        assert ds.config.augment is False

    def test_presets_are_independent(self, fake_dataset):
        """Modificar un dataset no debe afectar otro creado con el mismo preset."""
        csv_path, images_dir = fake_dataset
        ds1 = make_dataset("train", csv_path, images_dir)
        ds2 = make_dataset("train", csv_path, images_dir)
        ds1.config.augment = False
        assert ds2.config.augment is True
