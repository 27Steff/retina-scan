"""
test_trainer.py
---------------
Tests para trainer.py

Estrategia: se usa un modelo mínimo (_TinyModel) en lugar de EfficientNet-B4
para que los tests sean rápidos. La lógica del training loop no depende de
la arquitectura interna del modelo — solo de que sea un nn.Module con la
interfaz correcta.

Ejecutar con:
    pytest tests/test_trainer.py -v
"""

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from model import ModelConfig
from dataset import DatasetConfig, RetinaDataset, make_dataset
from preprocessing import PreprocessConfig
from trainer import TrainerConfig, RetinaTrainer, make_trainer


# Helpers

class _TinyModel(nn.Module):
    """
    Modelo mínimo con la misma interfaz que RetinaModel.
    AdaptiveAvgPool → Linear — acepta cualquier tamaño de imagen.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.config = ModelConfig(pretrained=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(x).flatten(1))

    def freeze_backbone(self) -> None:
        pass  # no backbone — no-op para compatibilidad con fit()

    def unfreeze_backbone(self) -> None:
        pass


def make_fake_retina(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Imagen BGR sintética para guardar con cv2.imwrite."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 5
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = [60, 80, 180]
    img[mask] = np.clip(
        img[mask].astype(np.int32) + rng.randint(0, 20, (mask.sum(), 3)),
        0, 255,
    ).astype(np.uint8)
    return img


@pytest.fixture
def fake_datasets(tmp_path):
    """
    Dataset sintético: 10 imágenes train + 5 val, todas las clases representadas.
    Usa image_size=64 para mantener los tests rápidos.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    train_rows = [
        {"id_code": f"train_{i:02d}", "diagnosis": i % 5}
        for i in range(10)
    ]
    val_rows = [
        {"id_code": f"val_{i:02d}", "diagnosis": i % 5}
        for i in range(5)
    ]

    for i, row in enumerate(train_rows + val_rows):
        cv2.imwrite(
            str(images_dir / f"{row['id_code']}.png"),
            make_fake_retina(seed=i),
        )

    small_preprocess = PreprocessConfig(image_size=64)
    train_csv = tmp_path / "train.csv"
    val_csv   = tmp_path / "val.csv"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv,   index=False)

    train_ds = make_dataset(
        "train", train_csv, images_dir,
        preprocess=small_preprocess,
    )
    val_ds = make_dataset(
        "val", val_csv, images_dir,
        preprocess=small_preprocess,
    )
    return train_ds, val_ds


@pytest.fixture
def tiny_config(tmp_path):
    """Config mínima para tests: 1 época, batch pequeño, sin multiprocessing."""
    return TrainerConfig(
        phase1_epochs=1,
        phase2_epochs=0,
        batch_size=4,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_best=True,
    )


@pytest.fixture
def tiny_model():
    return _TinyModel()


@pytest.fixture
def trainer(tiny_config):
    return RetinaTrainer(tiny_config)


# Tests de TrainerConfig

class TestTrainerConfig:

    def test_default_epochs(self):
        config = TrainerConfig()
        assert config.phase1_epochs == 5
        assert config.phase2_epochs == 15

    def test_default_learning_rates(self):
        config = TrainerConfig()
        assert config.phase1_lr > config.phase2_lr, (
            "phase1_lr debe ser mayor — backbone congelado permite LR alto"
        )

    def test_device_is_string(self):
        config = TrainerConfig()
        assert isinstance(config.device, str)
        assert config.device in ("cuda", "mps", "cpu")

    def test_custom_values(self):
        config = TrainerConfig(phase1_epochs=3, batch_size=32, device="cpu")
        assert config.phase1_epochs == 3
        assert config.batch_size == 32
        assert config.device == "cpu"


# Tests de _compute_class_weights

class TestComputeClassWeights:

    def test_weights_shape(self, trainer, fake_datasets):
        train_ds, _ = fake_datasets
        weights = trainer._compute_class_weights(train_ds)
        assert weights.shape == (5,)

    def test_weights_are_positive(self, trainer, fake_datasets):
        train_ds, _ = fake_datasets
        weights = trainer._compute_class_weights(train_ds)
        assert (weights > 0).all()

    def test_weights_are_float_tensor(self, trainer, fake_datasets):
        train_ds, _ = fake_datasets
        weights = trainer._compute_class_weights(train_ds)
        assert weights.dtype == torch.float32

    def test_rare_class_gets_higher_weight(self, tmp_path):
        """Una clase con menos muestras debe recibir más peso."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        # clase 0: 8 muestras, clase 4: 2 muestras — clase 4 debe pesar más
        rows = (
            [{"id_code": f"a{i}", "diagnosis": 0} for i in range(8)] +
            [{"id_code": f"b{i}", "diagnosis": 4} for i in range(2)]
        )
        for i, row in enumerate(rows):
            cv2.imwrite(str(images_dir / f"{row['id_code']}.png"), make_fake_retina(seed=i))
        csv = tmp_path / "labels.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        ds = make_dataset("train", csv, images_dir, preprocess=PreprocessConfig(image_size=64))

        weights = RetinaTrainer()._compute_class_weights(ds)
        assert weights[4] > weights[0], (
            "Clase 4 (2 muestras) debe pesar más que clase 0 (8 muestras)"
        )


# Tests de _quadratic_kappa

class TestQuadraticKappa:

    def test_perfect_predictions(self, trainer):
        labels = [0, 1, 2, 3, 4]
        preds  = [0, 1, 2, 3, 4]
        kappa = trainer._quadratic_kappa(labels, preds)
        assert kappa == pytest.approx(1.0)

    def test_range_is_valid(self, trainer):
        labels = [0, 1, 2, 3, 4, 0, 1]
        preds  = [1, 0, 3, 2, 4, 0, 2]
        kappa = trainer._quadratic_kappa(labels, preds)
        assert -1.0 <= kappa <= 1.0

    def test_all_same_prediction_returns_zero(self, trainer):
        """Cuando el modelo predice siempre clase 0, QWK = 0 (no lanza excepción)."""
        labels = [0, 1, 2, 3, 4]
        preds  = [0, 0, 0, 0, 0]
        kappa = trainer._quadratic_kappa(labels, preds)
        assert kappa == pytest.approx(0.0)

    def test_large_errors_penalized_more(self, trainer):
        """Error 0→4 debe resultar en QWK más bajo que error 0→1."""
        # Necesitamos labels diversas para que sklearn pueda calcular kappa
        labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        # Errores pequeños: solo desplazamiento de 1 clase
        small_error = trainer._quadratic_kappa(labels, [0, 1, 2, 3, 4, 1, 0, 3, 2, 4])
        # Errores grandes: predicciones en el extremo opuesto
        large_error = trainer._quadratic_kappa(labels, [4, 0, 0, 0, 0, 4, 0, 0, 0, 0])
        assert small_error > large_error


# Tests de _save_checkpoint

class TestSaveCheckpoint:

    def test_checkpoint_file_created(self, trainer, tiny_model, tmp_path):
        trainer._save_checkpoint(tiny_model, epoch=1, val_kappa=0.75)
        expected_path = Path(trainer.config.checkpoint_dir) / "best_model.pt"
        assert expected_path.exists()

    def test_checkpoint_contains_expected_keys(self, trainer, tiny_model):
        trainer._save_checkpoint(tiny_model, epoch=3, val_kappa=0.82)
        path = Path(trainer.config.checkpoint_dir) / "best_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_kappa" in checkpoint
        assert "model_config" in checkpoint

    def test_checkpoint_kappa_value(self, trainer, tiny_model):
        trainer._save_checkpoint(tiny_model, epoch=5, val_kappa=0.91)
        path = Path(trainer.config.checkpoint_dir) / "best_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["val_kappa"] == pytest.approx(0.91)


# Tests de _train_epoch y _val_epoch

class TestTrainValEpoch:

    @pytest.fixture
    def loaders(self, fake_datasets):
        train_ds, val_ds = fake_datasets
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=4, shuffle=False, num_workers=0
        )
        return train_loader, val_loader

    @pytest.fixture
    def setup(self, tiny_model):
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        return tiny_model, optimizer, criterion, device

    def test_train_epoch_returns_finite_loss(self, trainer, setup, loaders):
        model, optimizer, criterion, device = setup
        train_loader, _ = loaders
        loss = trainer._train_epoch(model, train_loader, optimizer, criterion, device)
        assert np.isfinite(loss)

    def test_train_epoch_loss_is_positive(self, trainer, setup, loaders):
        model, optimizer, criterion, device = setup
        train_loader, _ = loaders
        loss = trainer._train_epoch(model, train_loader, optimizer, criterion, device)
        assert loss > 0

    def test_val_epoch_returns_finite_loss_and_kappa(self, trainer, setup, loaders):
        model, _, criterion, device = setup
        _, val_loader = loaders
        loss, kappa = trainer._val_epoch(model, val_loader, criterion, device)
        assert np.isfinite(loss)
        assert np.isfinite(kappa)

    def test_val_epoch_kappa_in_valid_range(self, trainer, setup, loaders):
        model, _, criterion, device = setup
        _, val_loader = loaders
        _, kappa = trainer._val_epoch(model, val_loader, criterion, device)
        assert -1.0 <= kappa <= 1.0

    def test_train_sets_model_to_train_mode(self, trainer, setup, loaders):
        model, optimizer, criterion, device = setup
        model.eval()  # poner en eval primero
        train_loader, _ = loaders
        trainer._train_epoch(model, train_loader, optimizer, criterion, device)
        assert model.training

    def test_val_sets_model_to_eval_mode(self, trainer, setup, loaders):
        model, _, criterion, device = setup
        model.train()  # poner en train primero
        _, val_loader = loaders
        trainer._val_epoch(model, val_loader, criterion, device)
        assert not model.training


# Tests de fit()

class TestFit:

    def test_fit_returns_dict(self, trainer, tiny_model, fake_datasets):
        train_ds, val_ds = fake_datasets
        history = trainer.fit(tiny_model, train_ds, val_ds)
        assert isinstance(history, dict)

    def test_history_has_expected_keys(self, trainer, tiny_model, fake_datasets):
        train_ds, val_ds = fake_datasets
        history = trainer.fit(tiny_model, train_ds, val_ds)
        assert "train_loss" in history
        assert "val_loss"   in history
        assert "val_kappa"  in history
        assert "best_val_kappa" in history

    def test_history_length_matches_epochs(self, tiny_config, tiny_model, fake_datasets):
        """Con phase1_epochs=1 y phase2_epochs=0 debe haber exactamente 1 entrada."""
        trainer = RetinaTrainer(tiny_config)
        train_ds, val_ds = fake_datasets
        history = trainer.fit(tiny_model, train_ds, val_ds)
        assert len(history["train_loss"]) == 1
        assert len(history["val_kappa"])  == 1

    def test_two_phase_history_length(self, tmp_path, tiny_model, fake_datasets):
        """phase1=1 + phase2=1 debe dar 2 entradas en el historial."""
        config = TrainerConfig(
            phase1_epochs=1, phase2_epochs=1,
            batch_size=4, num_workers=0, device="cpu",
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        trainer = RetinaTrainer(config)
        train_ds, val_ds = fake_datasets
        history = trainer.fit(tiny_model, train_ds, val_ds)
        assert len(history["train_loss"]) == 2

    def test_best_val_kappa_is_max_of_val_kappa(self, trainer, tiny_model, fake_datasets):
        train_ds, val_ds = fake_datasets
        history = trainer.fit(tiny_model, train_ds, val_ds)
        assert history["best_val_kappa"] == max(history["val_kappa"])

    def test_checkpoint_saved_when_save_best_true(self, trainer, tiny_model, fake_datasets):
        train_ds, val_ds = fake_datasets
        trainer.fit(tiny_model, train_ds, val_ds)
        checkpoint_path = Path(trainer.config.checkpoint_dir) / "best_model.pt"
        assert checkpoint_path.exists()

    def test_no_checkpoint_when_save_best_false(self, tmp_path, tiny_model, fake_datasets):
        config = TrainerConfig(
            phase1_epochs=1, phase2_epochs=0,
            batch_size=4, num_workers=0, device="cpu",
            checkpoint_dir=str(tmp_path / "ckpt"),
            save_best=False,
        )
        trainer = RetinaTrainer(config)
        train_ds, val_ds = fake_datasets
        trainer.fit(tiny_model, train_ds, val_ds)
        checkpoint_path = Path(trainer.config.checkpoint_dir) / "best_model.pt"
        assert not checkpoint_path.exists()


# Tests de make_trainer

class TestMakeTrainer:

    def test_standard_kind(self):
        assert isinstance(make_trainer("standard"), RetinaTrainer)

    def test_fast_kind(self):
        t = make_trainer("fast")
        assert t.config.phase1_epochs < TrainerConfig().phase1_epochs

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            make_trainer("slow")

    def test_kwargs_override_config(self):
        t = make_trainer("standard", phase1_epochs=99, device="cpu")
        assert t.config.phase1_epochs == 99
        assert t.config.device == "cpu"
