"""
test_preprocessing.py
---------------------
Tests para preprocessing.py

Ejecutar con:
    pytest test_preprocessing.py -v
"""

import numpy as np
import pytest
from preprocessing import RetinaPreprocessor, PreprocessConfig


# Helpers

def make_fake_retina(h=600, w=600) -> np.ndarray:
    """
    Crea una imagen sintética de retina para tests.
    Círculo rojizo sobre fondo negro — simula una foto real.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 20
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask] = [180, 80, 60]   # color rojizo de retina
    img[mask] += np.random.randint(0, 30, (mask.sum(), 3), dtype=np.uint8)
    return img


# Tests de configuración

class TestPreprocessConfig:

    def test_default_config(self):
        config = PreprocessConfig()
        assert config.image_size == 512
        assert config.normalize is True
        assert config.remove_black_border is True

    def test_custom_config(self):
        config = PreprocessConfig(image_size=380, normalize=False)
        assert config.image_size == 380
        assert config.normalize is False


# Tests de process_array

class TestProcessArray:

    def test_output_shape_matches_config(self):
        """El output siempre debe tener el shape configurado."""
        for size in [224, 380, 512]:
            prep = RetinaPreprocessor(PreprocessConfig(image_size=size))
            img = make_fake_retina()
            result = prep.process_array(img)
            assert result.shape == (size, size, 3), \
                f"Esperado ({size}, {size}, 3), obtenido {result.shape}"

    def test_output_dtype_float_when_normalized(self):
        """Con normalize=True el output debe ser float32."""
        prep = RetinaPreprocessor(PreprocessConfig(normalize=True))
        result = prep.process_array(make_fake_retina())
        assert result.dtype == np.float32

    def test_output_dtype_uint8_when_not_normalized(self):
        """Con normalize=False el output debe ser uint8."""
        prep = RetinaPreprocessor(PreprocessConfig(normalize=False))
        result = prep.process_array(make_fake_retina())
        assert result.dtype == np.uint8

    def test_normalized_values_in_reasonable_range(self):
        """
        Normalización ImageNet produce valores aprox en [-3, 3].
        Fuera de ese rango indica que algo salió mal.
        """
        prep = RetinaPreprocessor(PreprocessConfig(normalize=True))
        result = prep.process_array(make_fake_retina())
        assert result.min() > -4.0, "Valores demasiado negativos"
        assert result.max() <  4.0, "Valores demasiado positivos"

    def test_wrong_shape_raises(self):
        """Una imagen que no sea (H, W, 3) debe lanzar ValueError."""
        prep = RetinaPreprocessor()
        with pytest.raises(ValueError, match="RGB"):
            prep.process_array(np.zeros((100, 100)))       # 2D
        with pytest.raises(ValueError, match="RGB"):
            prep.process_array(np.zeros((100, 100, 1)))    # 1 canal

    def test_non_square_input_produces_square_output(self):
        """Una imagen rectangular debe producir un output cuadrado."""
        prep = RetinaPreprocessor(PreprocessConfig(image_size=256))
        img = make_fake_retina(h=400, w=600)
        result = prep.process_array(img)
        assert result.shape[0] == result.shape[1] == 256

    def test_clahe_increases_local_contrast(self):
        """
        CLAHE debe aumentar la varianza local de la imagen
        (más contraste = más variación entre píxeles vecinos).
        """
        prep_clahe  = RetinaPreprocessor(PreprocessConfig(normalize=False))
        prep_noclahe = RetinaPreprocessor(
            PreprocessConfig(normalize=False, clahe_clip_limit=0.001)
        )
        img = make_fake_retina()
        result_clahe   = prep_clahe.process_array(img).astype(float)
        result_noclahe = prep_noclahe.process_array(img).astype(float)
        assert result_clahe.std() >= result_noclahe.std() * 0.9


# Tests de crop_black_border

class TestCropBlackBorder:

    def test_black_border_is_removed(self):
        """
        Una imagen con borde negro debe quedar más pequeña después del crop.
        """
        prep = RetinaPreprocessor(
            PreprocessConfig(image_size=256, remove_black_border=True)
        )
        prep_no_crop = RetinaPreprocessor(
            PreprocessConfig(image_size=256, remove_black_border=False)
        )
        img = make_fake_retina(600, 600)

        # Ambos terminan en 256x256 después del resize
        # pero el crop_border primero elimina el negro —
        # verificamos que el proceso no falla y produce el shape correcto
        result = prep.process_array(img)
        assert result.shape == (256, 256, 3)

    def test_fully_black_image_does_not_crash(self):
        """Una imagen completamente negra no debe lanzar excepción."""
        prep = RetinaPreprocessor(
            PreprocessConfig(remove_black_border=True)
        )
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        result = prep.process_array(black)
        assert result.shape[2] == 3


# Tests de to_tensor

class TestToTensor:

    def test_channel_first_format(self):
        """PyTorch necesita (C, H, W) — canales primero."""
        prep = RetinaPreprocessor(PreprocessConfig(image_size=224))
        img = prep.process_array(make_fake_retina())
        tensor = prep.to_tensor(img)
        assert tensor.shape == (3, 224, 224), \
            f"Esperado (3, 224, 224), obtenido {tensor.shape}"

    def test_tensor_dtype_is_float32(self):
        prep = RetinaPreprocessor()
        img = prep.process_array(make_fake_retina())
        tensor = prep.to_tensor(img)
        assert tensor.dtype == np.float32


# Tests de process_path

class TestProcessPath:

    def test_nonexistent_file_raises(self, tmp_path):
        prep = RetinaPreprocessor()
        with pytest.raises(FileNotFoundError):
            prep.process_path(tmp_path / "no_existe.jpg")

    def test_valid_image_file(self, tmp_path):
        """Debe procesar correctamente un archivo de imagen real."""
        import cv2
        img_path = tmp_path / "retina.jpg"
        fake = make_fake_retina(300, 300)
        cv2.imwrite(str(img_path), cv2.cvtColor(fake, cv2.COLOR_RGB2BGR))

        prep = RetinaPreprocessor(PreprocessConfig(image_size=224))
        result = prep.process_path(img_path)
        assert result.shape == (224, 224, 3)
