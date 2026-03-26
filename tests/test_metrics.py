"""
test_metrics.py
---------------
Tests para metrics.py

Ejecutar con:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
from metrics import MetricsConfig, MetricsResult, RetinaMetrics, make_metrics


# Fixtures

@pytest.fixture
def metrics():
    return RetinaMetrics()


@pytest.fixture
def perfect_labels():
    return [0, 1, 2, 3, 4]


@pytest.fixture
def perfect_preds():
    return [0, 1, 2, 3, 4]


@pytest.fixture
def mixed_result(metrics):
    """Resultado con mezcla de errores para tests de valores."""
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    preds  = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]
    return metrics.compute(labels, preds)


# Tests de MetricsConfig

class TestMetricsConfig:

    def test_default_num_classes(self):
        assert MetricsConfig().num_classes == 5

    def test_default_referral_threshold(self):
        assert MetricsConfig().referral_threshold == 2

    def test_class_names_length(self):
        config = MetricsConfig()
        assert len(config.class_names) == config.num_classes

    def test_custom_threshold(self):
        config = MetricsConfig(referral_threshold=1)
        assert config.referral_threshold == 1


# Tests de RetinaMetrics.compute — validación de entradas

class TestComputeValidation:

    def test_different_lengths_raises(self, metrics):
        with pytest.raises(ValueError, match="longitud"):
            metrics.compute([0, 1, 2], [0, 1])

    def test_empty_input_raises(self, metrics):
        with pytest.raises(ValueError, match="vacíos"):
            metrics.compute([], [])

    def test_label_out_of_range_raises(self, metrics):
        with pytest.raises(ValueError, match="labels"):
            metrics.compute([0, 5], [0, 1])  # clase 5 no existe

    def test_pred_out_of_range_raises(self, metrics):
        with pytest.raises(ValueError, match="preds"):
            metrics.compute([0, 1], [0, 6])

    def test_negative_label_raises(self, metrics):
        with pytest.raises(ValueError):
            metrics.compute([-1, 0], [0, 0])


# Tests de MetricsResult — estructura

class TestMetricsResultStructure:

    def test_returns_metrics_result(self, metrics, perfect_labels, perfect_preds):
        result = metrics.compute(perfect_labels, perfect_preds)
        assert isinstance(result, MetricsResult)

    def test_confusion_matrix_shape(self, mixed_result):
        assert mixed_result.confusion_matrix.shape == (5, 5)

    def test_per_class_arrays_length(self, mixed_result):
        assert len(mixed_result.per_class_precision) == 5
        assert len(mixed_result.per_class_recall)    == 5
        assert len(mixed_result.per_class_f1)        == 5

    def test_per_class_arrays_dtype(self, mixed_result):
        assert mixed_result.per_class_precision.dtype == np.float32
        assert mixed_result.per_class_recall.dtype    == np.float32
        assert mixed_result.per_class_f1.dtype        == np.float32


# Tests de QWK

class TestQWK:

    def test_perfect_predictions_qwk_is_1(self, metrics, perfect_labels, perfect_preds):
        result = metrics.compute(perfect_labels, perfect_preds)
        assert result.qwk == pytest.approx(1.0)

    def test_qwk_in_valid_range(self, mixed_result):
        assert -1.0 <= mixed_result.qwk <= 1.0

    def test_all_same_pred_returns_zero(self, metrics):
        labels = [0, 1, 2, 3, 4]
        preds  = [2, 2, 2, 2, 2]
        result = metrics.compute(labels, preds)
        assert result.qwk == pytest.approx(0.0)

    def test_large_errors_lower_qwk(self, metrics):
        labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        small_error = metrics.compute(labels, [0, 1, 2, 3, 4, 1, 0, 3, 2, 4]).qwk
        large_error = metrics.compute(labels, [4, 0, 0, 0, 0, 4, 0, 0, 0, 0]).qwk
        assert small_error > large_error


# Tests de accuracy

class TestAccuracy:

    def test_perfect_accuracy(self, metrics, perfect_labels, perfect_preds):
        result = metrics.compute(perfect_labels, perfect_preds)
        assert result.accuracy == pytest.approx(1.0)

    def test_zero_accuracy(self, metrics):
        result = metrics.compute([0, 1, 2, 3, 4], [4, 3, 0, 1, 2])
        assert result.accuracy == pytest.approx(0.0)

    def test_accuracy_in_valid_range(self, mixed_result):
        assert 0.0 <= mixed_result.accuracy <= 1.0


# Tests de métricas por clase

class TestPerClassMetrics:

    def test_perfect_precision_recall_f1(self, metrics, perfect_labels, perfect_preds):
        result = metrics.compute(perfect_labels, perfect_preds)
        np.testing.assert_allclose(result.per_class_precision, 1.0, atol=1e-5)
        np.testing.assert_allclose(result.per_class_recall,    1.0, atol=1e-5)
        np.testing.assert_allclose(result.per_class_f1,        1.0, atol=1e-5)

    def test_precision_recall_in_valid_range(self, mixed_result):
        assert (mixed_result.per_class_precision >= 0).all()
        assert (mixed_result.per_class_precision <= 1).all()
        assert (mixed_result.per_class_recall >= 0).all()
        assert (mixed_result.per_class_recall <= 1).all()

    def test_missing_class_in_preds_zero_precision(self, metrics):
        """Si el modelo nunca predice clase 3, su precision debe ser 0."""
        labels = [0, 1, 2, 3, 4]
        preds  = [0, 1, 2, 2, 4]   # clase 3 nunca predicha
        result = metrics.compute(labels, preds)
        assert result.per_class_precision[3] == pytest.approx(0.0)


# Tests de confusion matrix

class TestConfusionMatrix:

    def test_perfect_predictions_is_diagonal(self, metrics, perfect_labels, perfect_preds):
        result = metrics.compute(perfect_labels, perfect_preds)
        cm = result.confusion_matrix
        # Solo la diagonal debe tener valores
        off_diagonal = cm - np.diag(np.diag(cm))
        assert off_diagonal.sum() == 0

    def test_confusion_matrix_row_sums(self, metrics):
        """Suma de cada fila = número de muestras de esa clase."""
        labels = [0, 0, 1, 2, 3, 3, 4, 4, 4, 4]
        preds  = [0, 1, 1, 2, 3, 4, 4, 3, 4, 4]
        result = metrics.compute(labels, preds)
        label_counts = np.bincount(labels, minlength=5)
        np.testing.assert_array_equal(
            result.confusion_matrix.sum(axis=1), label_counts
        )

    def test_confusion_matrix_total_sum(self, metrics):
        labels = [0, 1, 2, 3, 4]
        preds  = [1, 2, 3, 4, 0]
        result = metrics.compute(labels, preds)
        assert result.confusion_matrix.sum() == len(labels)


# Tests de referral metrics

class TestReferralMetrics:

    def test_perfect_referral_sensitivity(self, metrics):
        """Predicciones perfectas → sensitivity = 1.0"""
        labels = [0, 1, 2, 3, 4]
        preds  = [0, 1, 2, 3, 4]
        result = metrics.compute(labels, preds)
        assert result.referral_sensitivity == pytest.approx(1.0)

    def test_perfect_referral_specificity(self, metrics):
        labels = [0, 1, 2, 3, 4]
        preds  = [0, 1, 2, 3, 4]
        result = metrics.compute(labels, preds)
        assert result.referral_specificity == pytest.approx(1.0)

    def test_model_misses_all_referrals_sensitivity_zero(self, metrics):
        """Si el modelo nunca detecta casos referibles → sensitivity = 0."""
        labels = [0, 1, 2, 3, 4]
        preds  = [0, 0, 0, 0, 0]   # siempre predice "no DR"
        result = metrics.compute(labels, preds)
        assert result.referral_sensitivity == pytest.approx(0.0)

    def test_referral_threshold_used_correctly(self):
        """threshold=1 hace que grado 1 también sea referible."""
        metrics_strict = RetinaMetrics(MetricsConfig(referral_threshold=1))
        labels = [0, 1, 2]
        preds  = [0, 1, 2]
        result = metrics_strict.compute(labels, preds)
        # Con threshold=1, grados 1 y 2 son referibles
        assert result.referral_sensitivity == pytest.approx(1.0)

    def test_no_referral_cases_returns_nan(self, metrics):
        """Si no hay casos referibles en el set, sensitivity = nan."""
        labels = [0, 0, 1, 1]
        preds  = [0, 1, 0, 1]
        result = metrics.compute(labels, preds)
        assert np.isnan(result.referral_sensitivity)

    def test_sensitivity_in_valid_range(self, mixed_result):
        s = mixed_result.referral_sensitivity
        assert 0.0 <= s <= 1.0

    def test_specificity_in_valid_range(self, mixed_result):
        s = mixed_result.referral_specificity
        assert 0.0 <= s <= 1.0


# Tests de MetricsResult.summary y to_dict

class TestMetricsResultOutput:

    def test_summary_is_string(self, mixed_result):
        assert isinstance(mixed_result.summary(), str)

    def test_summary_contains_qwk(self, mixed_result):
        assert "QWK" in mixed_result.summary()

    def test_summary_contains_all_class_names(self, mixed_result):
        summary = mixed_result.summary()
        for name in ("No DR", "Mild", "Moderate", "Severe", "Proliferative"):
            assert name in summary, f"'{name}' no aparece en el resumen"

    def test_to_dict_has_qwk(self, mixed_result):
        d = mixed_result.to_dict()
        assert "qwk" in d

    def test_to_dict_values_are_float(self, mixed_result):
        d = mixed_result.to_dict()
        for key, val in d.items():
            assert isinstance(val, float), f"'{key}' no es float: {type(val)}"

    def test_to_dict_contains_per_class_metrics(self, mixed_result):
        d = mixed_result.to_dict()
        assert "no_dr_f1"          in d
        assert "moderate_recall"   in d
        assert "proliferative_f1"  in d


# Tests de make_metrics

class TestMakeMetrics:

    def test_standard_kind(self):
        m = make_metrics("standard")
        assert isinstance(m, RetinaMetrics)
        assert m.config.referral_threshold == 2

    def test_binary_kind(self):
        m = make_metrics("binary")
        assert m.config.referral_threshold == 1

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            make_metrics("nonexistent")

    def test_kwargs_override_threshold(self):
        m = make_metrics("standard", referral_threshold=3)
        assert m.config.referral_threshold == 3
