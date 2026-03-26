"""
test_augmentation.py
--------------------
Tests para augmentation.py

Ejecutar con:
    pytest test_augmentation.py -v
"""

import numpy as np
import pytest
from augmentation import AugmentationConfig, RetinaAugmentor, make_augmentor


# Helpers

def make_fake_retina(h: int = 512, w: int = 512, seed: int = 42) -> np.ndarray:
    """
    Crea una imagen sintética de retina para tests.
    Usa una semilla fija para ser determinista, importante en tests de flip
    donde necesitamos que la imagen sea asimétrica.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 20
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = [180, 80, 60]
    # Ruido para romper la simetría — necesario para que flip cambie la imagen
    img[mask] = np.clip(
        img[mask].astype(np.int32) + rng.randint(0, 30, (mask.sum(), 3)),
        0, 255,
    ).astype(np.uint8)
    return img


# Tests de configuración

class TestAugmentationConfig:

    def test_default_values(self):
        config = AugmentationConfig()
        assert config.flip_prob == 0.5
        assert config.rotation_limit == 180
        assert config.rotation_prob == 0.8
        assert config.brightness_limit == 0.1
        assert config.contrast_limit == 0.1
        assert config.elastic_prob == 0.3
        assert config.coarse_dropout_prob == 0.3

    def test_custom_values(self):
        config = AugmentationConfig(flip_prob=0.0, rotation_limit=45)
        assert config.flip_prob == 0.0
        assert config.rotation_limit == 45

    def test_brightness_limits_are_conservative(self):
        """Los límites por defecto deben ser pequeños para preservar diagnóstico."""
        config = AugmentationConfig()
        assert config.brightness_limit <= 0.15, (
            "Brightness limit demasiado alto — riesgo de alterar colores diagnósticos"
        )
        assert config.contrast_limit <= 0.15, (
            "Contrast limit demasiado alto — riesgo de alterar colores diagnósticos"
        )


# Tests de RetinaAugmentor

class TestRetinaAugmentor:

    @pytest.fixture
    def augmentor(self):
        return RetinaAugmentor()

    @pytest.fixture
    def img(self):
        return make_fake_retina()

    # --- Shape y dtype ---

    def test_output_shape_preserved(self, augmentor, img):
        """La aumentación no debe cambiar el shape de la imagen."""
        result = augmentor.apply(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self, augmentor, img):
        """El output debe ser uint8 — la normalización viene después."""
        result = augmentor.apply(img)
        assert result.dtype == np.uint8

    def test_non_square_input_shape_preserved(self):
        """Imágenes rectangulares deben conservar su shape."""
        augmentor = RetinaAugmentor()
        img = make_fake_retina(h=400, w=600)
        result = augmentor.apply(img)
        assert result.shape == (400, 600, 3)

    # --- Validación de entrada ---

    def test_wrong_shape_2d_raises(self, augmentor):
        with pytest.raises(ValueError, match="RGB"):
            augmentor.apply(np.zeros((100, 100), dtype=np.uint8))

    def test_wrong_shape_1_channel_raises(self, augmentor):
        with pytest.raises(ValueError, match="RGB"):
            augmentor.apply(np.zeros((100, 100, 1), dtype=np.uint8))

    def test_float32_input_raises(self, augmentor):
        """Debe fallar con float32 — la aumentación ocurre antes de normalizar."""
        with pytest.raises(ValueError, match="uint8"):
            augmentor.apply(np.zeros((100, 100, 3), dtype=np.float32))

    # --- Comportamiento estocástico ---

    def test_augmentation_is_applied_across_trials(self, img):
        """
        Con configuración por defecto (flip_prob=0.5, rotation_prob=0.8),
        en 20 intentos al menos uno debe producir un resultado distinto al original.
        La probabilidad de que los 20 sean identidad es prácticamente 0.
        """
        augmentor = RetinaAugmentor()
        found_different = any(
            not np.array_equal(augmentor.apply(img.copy()), img)
            for _ in range(20)
        )
        assert found_different, "La aumentación nunca cambió la imagen en 20 intentos"

    def test_flip_p1_changes_image(self):
        """Con flip_prob=1.0 el output SIEMPRE debe diferir del input (imagen asimétrica)."""
        config = AugmentationConfig(
            flip_prob=1.0,
            rotation_limit=0,
            rotation_prob=0.0,
            brightness_limit=0.0,
            contrast_limit=0.0,
            color_prob=0.0,
            elastic_prob=0.0,
            coarse_dropout_prob=0.0,
        )
        augmentor = RetinaAugmentor(config)
        img = make_fake_retina()  # imagen asimétrica gracias al ruido
        result = augmentor.apply(img.copy())
        assert not np.array_equal(result, img), (
            "HorizontalFlip con p=1 no cambió la imagen"
        )

    def test_val_preset_is_identity(self):
        """
        El preset 'val' (todos los probs = 0) debe ser una transformación identidad.
        La imagen de validación nunca debe ser modificada.
        """
        augmentor = make_augmentor("val")
        img = make_fake_retina()
        result = augmentor.apply(img.copy())
        np.testing.assert_array_equal(result, img)

    def test_output_values_stay_uint8_range(self, augmentor, img):
        """Los valores de píxeles deben permanecer en [0, 255]."""
        result = augmentor.apply(img)
        assert result.min() >= 0
        assert result.max() <= 255


# Tests de make_augmentor

class TestMakeAugmentor:

    def test_train_returns_augmentor(self):
        assert isinstance(make_augmentor("train"), RetinaAugmentor)

    def test_val_returns_augmentor(self):
        assert isinstance(make_augmentor("val"), RetinaAugmentor)

    def test_light_returns_augmentor(self):
        assert isinstance(make_augmentor("light"), RetinaAugmentor)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            make_augmentor("nonexistent")

    def test_kwargs_override_config(self):
        """Los kwargs deben sobreescribir los valores del preset."""
        aug = make_augmentor("train", flip_prob=0.0, rotation_limit=10)
        assert aug.config.flip_prob == 0.0
        assert aug.config.rotation_limit == 10

    def test_train_pipeline_has_more_transforms_than_val(self):
        """
        El preset 'train' debe tener más transforms que 'val'.
        Esto verifica que la lógica de construcción del pipeline funciona.
        """
        train = make_augmentor("train")
        val = make_augmentor("val")
        assert len(train._transform.transforms) > len(val._transform.transforms)

    def test_val_pipeline_is_empty(self):
        """El preset 'val' debe producir un pipeline vacío (identidad pura)."""
        val = make_augmentor("val")
        assert len(val._transform.transforms) == 0

    def test_light_has_fewer_transforms_than_train(self):
        """'light' debe ser más conservador que 'train'."""
        train = make_augmentor("train")
        light = make_augmentor("light")
        assert len(light._transform.transforms) < len(train._transform.transforms)

    def test_presets_are_independent(self):
        """
        Modificar el config de un augmentor no debe afectar a otro.
        (Verifica que replace() crea instancias independientes.)
        """
        aug1 = make_augmentor("train")
        aug2 = make_augmentor("train")
        aug1.config.flip_prob = 0.0
        assert aug2.config.flip_prob == 0.5
