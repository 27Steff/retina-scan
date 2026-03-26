"""
test_model.py
-------------
Tests para model.py

Ejecutar con:
    pytest tests/test_model.py -v

Nota: todos los tests usan pretrained=False para evitar descargas de red.
Los pesos de ImageNet no son necesarios para verificar la arquitectura.
"""

import pytest
import torch
import torch.nn as nn
from model import ModelConfig, RetinaModel, make_model


# Fixtures

@pytest.fixture
def model():
    """Modelo sin pesos preentrenados — rápido, sin red."""
    return RetinaModel(ModelConfig(pretrained=False))


@pytest.fixture
def batch():
    """Batch pequeño para tests rápidos. EfficientNet acepta cualquier tamaño."""
    return torch.randn(2, 3, 64, 64)



# Tests de ModelConfig

class TestModelConfig:

    def test_default_num_classes(self):
        assert ModelConfig().num_classes == 5

    def test_default_freeze_backbone(self):
        """El backbone debe estar congelado por defecto — estrategia fase 1."""
        assert ModelConfig().freeze_backbone is True

    def test_default_pretrained(self):
        assert ModelConfig().pretrained is True

    def test_custom_values(self):
        config = ModelConfig(num_classes=3, dropout_rate=0.2, pretrained=False)
        assert config.num_classes == 3
        assert config.dropout_rate == 0.2
        assert config.pretrained is False


# Tests de RetinaModel — arquitectura

class TestRetinaModelArchitecture:

    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_classifier_output_matches_num_classes(self, model):
        """El head lineal debe tener exactamente num_classes salidas."""
        linear = model.backbone.classifier[1]
        assert isinstance(linear, nn.Linear)
        assert linear.out_features == 5

    def test_classifier_input_features_b4(self, model):
        """EfficientNet-B4 tiene 1792 features antes del classifier."""
        linear = model.backbone.classifier[1]
        assert linear.in_features == 1792

    def test_classifier_dropout_matches_config(self, model):
        dropout = model.backbone.classifier[0]
        assert isinstance(dropout, nn.Dropout)
        assert dropout.p == ModelConfig().dropout_rate

    def test_custom_num_classes(self):
        model = RetinaModel(ModelConfig(num_classes=3, pretrained=False))
        linear = model.backbone.classifier[1]
        assert linear.out_features == 3


# Tests de RetinaModel — forward pass

class TestRetinaModelForward:

    def test_output_shape(self, model, batch):
        """Output debe ser (batch_size, num_classes)."""
        logits = model(batch)
        assert logits.shape == (2, 5)

    def test_output_dtype_float32(self, model, batch):
        logits = model(batch)
        assert logits.dtype == torch.float32

    def test_output_is_logits_not_probabilities(self, model, batch):
        """
        El modelo devuelve logits — los valores no deben sumar 1.
        CrossEntropyLoss aplica softmax internamente.
        """
        logits = model(batch)
        row_sums = logits.softmax(dim=1).sum(dim=1)
        # softmax sí suma 1, pero los logits crudos no
        assert not torch.allclose(logits.sum(dim=1), torch.ones(2))
        # softmax sí suma 1 — verifica que el modelo al menos produce
        # una distribución válida cuando se aplica softmax
        assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)

    def test_different_batch_sizes(self, model):
        """El modelo debe aceptar cualquier tamaño de batch."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 64, 64)
            out = model(x)
            assert out.shape == (batch_size, 5)

    def test_no_gradient_on_frozen_backbone(self, batch):
        """Con backbone congelado, solo el clasificador tiene gradiente."""
        model = RetinaModel(ModelConfig(pretrained=False, freeze_backbone=True))
        logits = model(batch)
        loss = logits.sum()
        loss.backward()

        # Features del backbone: sin gradiente
        for name, param in model.backbone.features.named_parameters():
            assert param.grad is None, (
                f"Backbone param '{name}' tiene gradiente pero debería estar congelado"
            )

        # Clasificador: con gradiente
        for name, param in model.backbone.classifier.named_parameters():
            assert param.grad is not None, (
                f"Classifier param '{name}' no tiene gradiente"
            )


# Tests de freeze / unfreeze

class TestFreezeUnfreeze:

    def test_frozen_by_default(self, model):
        """Con freeze_backbone=True, solo el clasificador es trainable."""
        trainable = model.num_parameters(trainable_only=True)
        total = model.num_parameters(trainable_only=False)
        assert trainable < total

    def test_unfreeze_makes_all_trainable(self, model):
        model.unfreeze_backbone()
        trainable = model.num_parameters(trainable_only=True)
        total = model.num_parameters(trainable_only=False)
        assert trainable == total

    def test_refreeze_after_unfreeze(self, model):
        model.unfreeze_backbone()
        model.freeze_backbone()
        trainable_frozen = model.num_parameters(trainable_only=True)
        model.unfreeze_backbone()
        trainable_all = model.num_parameters(trainable_only=True)
        assert trainable_frozen < trainable_all

    def test_frozen_trainable_params_are_classifier_only(self, model):
        """Con backbone congelado, todos los params trainables son del classifier."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert "classifier" in name, (
                    f"Param '{name}' es trainable pero no es del classifier"
                )

    def test_num_parameters_total_is_positive(self, model):
        assert model.num_parameters() > 0

    def test_num_parameters_trainable_less_than_total(self, model):
        assert model.num_parameters(trainable_only=True) < model.num_parameters()


# Tests de make_model

class TestMakeModel:

    def test_known_kind_returns_model(self):
        model = make_model("efficientnet_b4", pretrained=False)
        assert isinstance(model, RetinaModel)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            make_model("resnet50")

    def test_kwargs_override_config(self):
        model = make_model("efficientnet_b4", pretrained=False, dropout_rate=0.2)
        dropout = model.backbone.classifier[0]
        assert dropout.p == 0.2

    def test_kwargs_num_classes(self):
        model = make_model("efficientnet_b4", pretrained=False, num_classes=3)
        assert model.backbone.classifier[1].out_features == 3
