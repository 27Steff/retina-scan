"""
metrics.py
Métricas de evaluación para el modelo de retinopatía diabética.

Más allá del QWK: un modelo puede tener QWK=0.85 y fallar sistemáticamente en casos
severos (clase 3), lo cual es inaceptable en medicina. Este módulo añade métricas
de derivación (referral): grados 2-4 deben ser referidos al oftalmólogo.
referral_sensitivity es la métrica crítica — un false negative es un paciente severo
no derivado, con riesgo de pérdida de visión.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union
from sklearn.metrics import (
    cohen_kappa_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

Labels = Union[List[int], np.ndarray]


@dataclass
class MetricsResult:
    """Resultado completo de una evaluación con QWK, métricas por clase y referral."""
    qwk: float
    accuracy: float
    confusion_matrix: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    referral_sensitivity: float   # recall de grados 2-4: ¿cuántos casos referibles detectamos?
    referral_specificity: float   # especificidad de grados 0-1: ¿cuántos sanos identificamos?
    class_names: tuple = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")

    def summary(self) -> str:
        """Resumen formateado para consola o logs."""
        lines = [
            "── Métricas de Evaluación ──────────────────────────",
            f"  QWK              : {self.qwk:.4f}",
            f"  Accuracy         : {self.accuracy:.4f}",
            "",
            "── Referral (derivación al oftalmólogo) ────────────",
            f"  Sensitivity      : {self.referral_sensitivity:.4f}  "
            "(de casos referibles, ¿cuántos detectamos?)",
            f"  Specificity      : {self.referral_specificity:.4f}  "
            "(de casos no referibles, ¿cuántos identificamos?)",
            "",
            "── Por clase ────────────────────────────────────────",
            f"  {'Clase':<18} {'Precision':>9} {'Recall':>9} {'F1':>9}",
            "  " + "-" * 48,
        ]
        for i, name in enumerate(self.class_names):
            lines.append(
                f"  {name:<18} "
                f"{self.per_class_precision[i]:>9.4f} "
                f"{self.per_class_recall[i]:>9.4f} "
                f"{self.per_class_f1[i]:>9.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializa a dict plano para JSON o MLflow."""
        result = {
            "qwk": self.qwk,
            "accuracy": self.accuracy,
            "referral_sensitivity": self.referral_sensitivity,
            "referral_specificity": self.referral_specificity,
        }
        for i, name in enumerate(self.class_names):
            key = name.lower().replace(" ", "_")
            result[f"{key}_precision"] = float(self.per_class_precision[i])
            result[f"{key}_recall"]    = float(self.per_class_recall[i])
            result[f"{key}_f1"]        = float(self.per_class_f1[i])
        return result


@dataclass
class MetricsConfig:
    """Parámetros de evaluación. referral_threshold=2 es el estándar clínico para retinopatía moderada+."""
    num_classes: int = 5
    class_names: tuple = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
    referral_threshold: int = 2


class RetinaMetrics:
    """Calcula QWK, métricas por clase y métricas de derivación clínica."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()

    def compute(self, labels: Labels, preds: Labels) -> MetricsResult:
        """Calcula todas las métricas dado un conjunto de predicciones."""
        labels = np.asarray(labels, dtype=np.int64)
        preds  = np.asarray(preds,  dtype=np.int64)
        self._validate(labels, preds)

        return MetricsResult(
            qwk=self._qwk(labels, preds),
            accuracy=float(accuracy_score(labels, preds)),
            confusion_matrix=self._confusion_matrix(labels, preds),
            per_class_precision=self._per_class_precision(labels, preds),
            per_class_recall=self._per_class_recall(labels, preds),
            per_class_f1=self._per_class_f1(labels, preds),
            referral_sensitivity=self._referral_sensitivity(labels, preds),
            referral_specificity=self._referral_specificity(labels, preds),
            class_names=self.config.class_names,
        )

    def _validate(self, labels: np.ndarray, preds: np.ndarray) -> None:
        """Lanza ValueError si las entradas no son válidas."""
        if len(labels) != len(preds):
            raise ValueError(
                f"labels y preds deben tener la misma longitud. "
                f"Recibido: {len(labels)} y {len(preds)}"
            )
        if len(labels) == 0:
            raise ValueError("labels y preds no pueden estar vacíos.")
        n = self.config.num_classes
        for name, arr in [("labels", labels), ("preds", preds)]:
            if arr.min() < 0 or arr.max() >= n:
                raise ValueError(
                    f"{name} debe contener valores en [0, {n-1}]. "
                    f"Encontrado: min={arr.min()}, max={arr.max()}"
                )

    def _qwk(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Quadratic Weighted Kappa. Retorna 0.0 si no es computable."""
        try:
            return float(cohen_kappa_score(labels, preds, weights="quadratic"))
        except ValueError:
            return 0.0

    def _confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        n = self.config.num_classes
        return confusion_matrix(labels, preds, labels=list(range(n)))

    def _per_class_precision(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        precision, _, _, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(self.config.num_classes)),
            zero_division=0,
        )
        return precision.astype(np.float32)

    def _per_class_recall(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        _, recall, _, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(self.config.num_classes)),
            zero_division=0,
        )
        return recall.astype(np.float32)

    def _per_class_f1(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(self.config.num_classes)),
            zero_division=0,
        )
        return f1.astype(np.float32)

    def _referral_labels(self, arr: np.ndarray) -> np.ndarray:
        """Convierte etiquetas 0-4 a binario: 1=referible, 0=no referible."""
        return (arr >= self.config.referral_threshold).astype(np.int64)

    def _referral_sensitivity(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Recall de grados >= threshold. nan si no hay casos referibles en el conjunto."""
        binary_labels = self._referral_labels(labels)
        binary_preds  = self._referral_labels(preds)

        if binary_labels.sum() == 0:
            return float("nan")

        tp = int(((binary_preds == 1) & (binary_labels == 1)).sum())
        fn = int(((binary_preds == 0) & (binary_labels == 1)).sum())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _referral_specificity(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Especificidad de grados < threshold. nan si no hay casos no-referibles."""
        binary_labels = self._referral_labels(labels)
        binary_preds  = self._referral_labels(preds)

        if (binary_labels == 0).sum() == 0:
            return float("nan")

        tn = int(((binary_preds == 0) & (binary_labels == 0)).sum())
        fp = int(((binary_preds == 1) & (binary_labels == 0)).sum())
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def make_metrics(kind: str, **kwargs) -> RetinaMetrics:
    """
    Factory con presets para métricas.

    kind: "standard" (5 clases, threshold=2) | "binary" (threshold=1, detección DR vs no-DR)
    **kwargs sobreescribe campos de MetricsConfig.
    """
    from dataclasses import replace
    registry = {
        "standard": MetricsConfig(),
        "binary":   MetricsConfig(referral_threshold=1),
    }

    if kind not in registry:
        raise ValueError(
            f"Unknown metrics kind '{kind}'. Options: {list(registry)}"
        )

    config = replace(registry[kind], **kwargs)
    return RetinaMetrics(config)
