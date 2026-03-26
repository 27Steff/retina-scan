"""
trainer.py
Loop de entrenamiento en dos fases para el modelo de retinopatía diabética.

Fase 1: backbone congelado (LR alto, pocas épocas). Fase 2: fine-tuning completo (LR bajo).
Clases desbalanceadas (clase 0 ~49%) compensadas con pesos inversamente proporcionales
en CrossEntropyLoss. El mejor modelo se guarda por QWK — métrica oficial de APTOS —
no por loss, porque QWK penaliza errores grandes (0 vs 4) más que errores pequeños (0 vs 1).
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from model import RetinaModel


@dataclass
class TrainerConfig:
    """
    Hiperparámetros de entrenamiento.
    patience=0 desactiva early stopping; >0 detiene tras N épocas sin mejora de QWK.
    """
    phase1_epochs: int = 5
    phase2_epochs: int = 15
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-5
    batch_size: int = 16
    num_workers: int = 4
    weight_decay: float = 1e-4
    device: str = field(
        default_factory=lambda: (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    patience: int = 0


@dataclass
class CPUTrainerConfig(TrainerConfig):
    """
    Config para entrenamiento local en CPU con EfficientNet-B0 e imágenes 224 px.
    Total de épocas = 10, early stopping con patience=3.
    """
    phase1_epochs: int = 3
    phase2_epochs: int = 7
    batch_size: int = 4
    num_workers: int = 0
    device: str = "cpu"
    patience: int = 3
    model_type: str = "efficientnet_b0"
    image_size: int = 224


class RetinaTrainer:
    """
    Orquesta el entrenamiento en dos fases del modelo de retina.
    El historial retornado por fit() es una lista plana (fase 1 + fase 2) con métricas por época.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self._best_kappa: float = -1.0

    def fit(
        self,
        model: RetinaModel,
        train_dataset,
        val_dataset,
    ) -> dict:
        """
        Ejecuta el entrenamiento completo en dos fases.
        Retorna dict con train_loss, val_loss, val_kappa (listas por época) y best_val_kappa.
        """
        self._best_kappa = -1.0
        history = {"train_loss": [], "val_loss": [], "val_kappa": []}

        class_weights = self._compute_class_weights(train_dataset)

        if self.config.phase1_epochs > 0:
            print(f"\n{'='*50}")
            print(f"Fase 1 — Classifier only ({self.config.phase1_epochs} épocas)")
            print(f"{'='*50}")
            self._run_phase(
                model, train_dataset, val_dataset,
                phase=1, class_weights=class_weights, history=history,
            )
            model.unfreeze_backbone()

        if self.config.phase2_epochs > 0:
            print(f"\n{'='*50}")
            print(f"Fase 2 — Fine-tuning completo ({self.config.phase2_epochs} épocas)")
            print(f"{'='*50}")
            self._run_phase(
                model, train_dataset, val_dataset,
                phase=2, class_weights=class_weights, history=history,
            )

        history["best_val_kappa"] = self._best_kappa
        return history

    def _run_phase(
        self,
        model: RetinaModel,
        train_dataset,
        val_dataset,
        phase: int,
        class_weights: torch.Tensor,
        history: dict,
    ) -> None:
        """Ejecuta todas las épocas de una fase de entrenamiento."""
        lr = self.config.phase1_lr if phase == 1 else self.config.phase2_lr
        epochs = self.config.phase1_epochs if phase == 1 else self.config.phase2_epochs
        device = torch.device(self.config.device)

        model.to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        epochs_no_improve = 0
        phase_best_kappa = -1.0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_kappa = self._val_epoch(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_kappa"].append(val_kappa)

            improved = val_kappa > self._best_kappa
            if improved:
                self._best_kappa = val_kappa
                phase_best_kappa = val_kappa
                epochs_no_improve = 0
                if self.config.save_best:
                    self._save_checkpoint(model, epoch, val_kappa)
            else:
                epochs_no_improve += 1

            print(
                f"  Época {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val QWK: {val_kappa:.4f}"
                + (" ✓" if improved else "")
            )

            if self.config.patience > 0 and epochs_no_improve >= self.config.patience:
                print(f"  Early stopping: {self.config.patience} épocas sin mejora.")
                break

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """Ejecuta una época de entrenamiento. Retorna loss promedio."""
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _val_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple:
        """Retorna (val_loss, quadratic_weighted_kappa)."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())

        kappa = self._quadratic_kappa(all_labels, all_preds)
        return total_loss / len(loader), kappa

    def _compute_class_weights(self, dataset) -> torch.Tensor:
        """
        Pesos inversamente proporcionales a la frecuencia: weight[c] = total / (n_classes * count[c]).
        Clases raras como grado 3 (severa) reciben más peso, forzando atención del modelo.
        """
        counts = dataset.label_counts()
        total = sum(counts.values())
        n_classes = dataset.NUM_CLASSES
        weights = torch.ones(n_classes)
        for cls, count in counts.items():
            weights[cls] = total / (n_classes * max(count, 1))
        return weights

    def _quadratic_kappa(self, labels: list, preds: list) -> float:
        """QWK — métrica oficial de APTOS. Retorna 0.0 si todas las predicciones son iguales."""
        try:
            return float(cohen_kappa_score(labels, preds, weights="quadratic"))
        except ValueError:
            return 0.0

    def _save_checkpoint(self, model: RetinaModel, epoch: int, val_kappa: float) -> None:
        """Guarda el estado del modelo cuando mejora el QWK de validación."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": model.config,
                "val_kappa": val_kappa,
            },
            path,
        )


def make_trainer(kind: str, **kwargs) -> RetinaTrainer:
    """
    Factory con presets para trainers.

    kind:
        "standard" — configuración completa para entrenamiento real
        "fast"     — pocas épocas para verificar que el pipeline funciona
        "cpu"      — entrenamiento local en CPU con B0 e imagen 224 px
    **kwargs sobreescribe campos de TrainerConfig.
    """
    presets = {
        "standard": TrainerConfig(),
        "fast": TrainerConfig(
            phase1_epochs=2,
            phase2_epochs=3,
            batch_size=8,
        ),
        "cpu": CPUTrainerConfig(),
    }

    if kind not in presets:
        raise ValueError(
            f"Unknown trainer kind '{kind}'. Options: {list(presets)}"
        )

    from dataclasses import replace
    config = replace(presets[kind], **kwargs)
    return RetinaTrainer(config)
