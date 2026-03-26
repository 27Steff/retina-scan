"""
test_explainability.py
----------------------
Tests para explainability.py

Nota: todos los tests usan pretrained=False para evitar descargas de red.

Ejecutar con:
    pytest tests/test_explainability.py -v
"""

import numpy as np
import pytest
import torch

from model import RetinaModel, ModelConfig
from explainability import GradCAMConfig, GradCAM, make_explainer


# Fixtures

@pytest.fixture(scope="module")
def model():
    """Modelo sin pesos preentrenados — compartido entre tests del módulo."""
    m = RetinaModel(ModelConfig(pretrained=False, freeze_backbone=False))
    m.eval()
    return m


@pytest.fixture
def cam(model):
    """GradCAM fresco por test, con cleanup garantizado."""
    c = GradCAM(model)
    yield c
    c.remove()


@pytest.fixture
def img_tensor():
    """Tensor de imagen preprocesada (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def img_uint8():
    """Imagen uint8 para el overlay."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# Tests de GradCAMConfig

class TestGradCAMConfig:

    def test_default_alpha(self):
        assert GradCAMConfig().alpha == 0.4

    def test_default_colormap(self):
        assert GradCAMConfig().colormap == "jet"

    def test_custom_values(self):
        config = GradCAMConfig(alpha=0.6, colormap="inferno")
        assert config.alpha == 0.6
        assert config.colormap == "inferno"


# Tests de GradCAM — target layer

class TestTargetLayer:

    def test_auto_detects_last_feature_block(self, cam, model):
        """El target layer debe ser el último bloque de features de EfficientNet."""
        assert cam.target_layer is model.backbone.features[-1]

    def test_custom_target_layer(self, model):
        custom_layer = model.backbone.features[-2]
        cam = GradCAM(model, target_layer=custom_layer)
        assert cam.target_layer is custom_layer
        cam.remove()

    def test_hooks_registered_on_init(self, cam):
        """Deben haber exactamente 2 hooks: forward y backward."""
        assert len(cam._hooks) == 2


# Tests de generate — shape y rango

class TestGenerate:

    def test_heatmap_shape_matches_feature_map(self, cam, img_tensor):
        """El heatmap sin resize tiene la resolución de los feature maps."""
        heatmap = cam.generate(img_tensor)
        assert heatmap.ndim == 2

    def test_heatmap_values_in_0_1(self, cam, img_tensor):
        heatmap = cam.generate(img_tensor)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_heatmap_dtype_float32(self, cam, img_tensor):
        heatmap = cam.generate(img_tensor)
        assert heatmap.dtype == np.float32

    def test_accepts_3d_tensor(self, cam):
        """Debe aceptar (3, H, W) además de (1, 3, H, W)."""
        tensor_3d = torch.randn(3, 64, 64)
        heatmap = cam.generate(tensor_3d)
        assert heatmap.ndim == 2

    def test_wrong_ndim_raises(self, cam):
        with pytest.raises(ValueError):
            cam.generate(torch.randn(64, 64))  # 2D — inválido

    def test_default_class_idx_uses_argmax(self, cam, img_tensor):
        """Sin class_idx, debe usar la clase con mayor logit."""
        # Ejecutamos dos veces: auto y con argmax explícito
        with torch.no_grad():
            logits = cam.model(img_tensor)
        predicted_class = int(logits.argmax(dim=1).item())

        heatmap_auto     = cam.generate(img_tensor)
        heatmap_explicit = cam.generate(img_tensor, class_idx=predicted_class)

        np.testing.assert_array_equal(heatmap_auto, heatmap_explicit)

    def test_different_class_idx_gives_different_heatmap(self, cam, img_tensor):
        """
        Clases distintas deben producir heatmaps distintos (en general).
        Usamos clase 0 vs clase 4 — los más extremos.
        """
        heatmap_0 = cam.generate(img_tensor, class_idx=0)
        heatmap_4 = cam.generate(img_tensor, class_idx=4)
        # No pueden ser idénticos para clases opuestas
        assert not np.array_equal(heatmap_0, heatmap_4)

    def test_activations_set_after_generate(self, cam, img_tensor):
        """Los hooks deben haber capturado activaciones tras el forward."""
        cam.generate(img_tensor)
        assert cam._activations is not None

    def test_gradients_set_after_generate(self, cam, img_tensor):
        """Los hooks deben haber capturado gradientes tras el backward."""
        cam.generate(img_tensor)
        assert cam._gradients is not None


# Tests de overlay

class TestOverlay:

    def test_overlay_shape_matches_image(self, cam, img_tensor, img_uint8):
        heatmap = cam.generate(img_tensor)
        overlay = cam.overlay(heatmap, img_uint8)
        assert overlay.shape == img_uint8.shape

    def test_overlay_dtype_uint8(self, cam, img_tensor, img_uint8):
        heatmap = cam.generate(img_tensor)
        overlay = cam.overlay(heatmap, img_uint8)
        assert overlay.dtype == np.uint8

    def test_overlay_values_in_uint8_range(self, cam, img_tensor, img_uint8):
        heatmap = cam.generate(img_tensor)
        overlay = cam.overlay(heatmap, img_uint8)
        assert overlay.min() >= 0
        assert overlay.max() <= 255

    def test_overlay_different_image_size(self, cam, img_tensor):
        """El overlay debe redimensionar el heatmap al tamaño de la imagen."""
        big_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        heatmap = cam.generate(img_tensor)
        overlay = cam.overlay(heatmap, big_img)
        assert overlay.shape == (256, 256, 3)

    def test_colormap_inferno(self, model, img_tensor, img_uint8):
        config = GradCAMConfig(colormap="inferno")
        cam = GradCAM(model, config)
        heatmap = cam.generate(img_tensor)
        overlay = cam.overlay(heatmap, img_uint8)
        assert overlay.shape == img_uint8.shape
        cam.remove()

    def test_alpha_zero_returns_original_image(self, model, img_uint8):
        """Alpha=0 → el overlay debe ser igual a la imagen original."""
        config = GradCAMConfig(alpha=0.0)
        cam = GradCAM(model, config)
        tensor = torch.randn(1, 3, 64, 64)
        heatmap = cam.generate(tensor)
        overlay = cam.overlay(heatmap, img_uint8)
        np.testing.assert_array_equal(overlay, img_uint8)
        cam.remove()


# ---------------------------------------------------------------------------
# Tests de remove y context manager
# ---------------------------------------------------------------------------

class TestHooksLifecycle:

    def test_remove_clears_hooks_list(self, model):
        cam = GradCAM(model)
        assert len(cam._hooks) == 2
        cam.remove()
        assert len(cam._hooks) == 0

    def test_context_manager_removes_hooks(self, model, img_tensor):
        with GradCAM(model) as cam:
            cam.generate(img_tensor)
            inner_cam = cam  # guardar referencia
        # Después del with, los hooks deben estar liberados
        assert len(inner_cam._hooks) == 0

    def test_context_manager_returns_heatmap(self, model, img_tensor):
        with GradCAM(model) as cam:
            heatmap = cam.generate(img_tensor)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0


# ---------------------------------------------------------------------------
# Tests de make_explainer
# ---------------------------------------------------------------------------

class TestMakeExplainer:

    def test_gradcam_kind_returns_gradcam(self, model):
        cam = make_explainer("gradcam", model)
        assert isinstance(cam, GradCAM)
        cam.remove()

    def test_unknown_kind_raises(self, model):
        with pytest.raises(ValueError, match="Unknown"):
            make_explainer("scorecam", model)

    def test_kwargs_override_alpha(self, model):
        cam = make_explainer("gradcam", model, alpha=0.7)
        assert cam.config.alpha == 0.7
        cam.remove()

    def test_kwargs_override_colormap(self, model):
        cam = make_explainer("gradcam", model, colormap="hot")
        assert cam.config.colormap == "hot"
        cam.remove()
