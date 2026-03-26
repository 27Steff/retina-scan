"""
test_api.py
-----------
Tests para api.py

Estrategia: se inyecta un RetinaModel sin pesos preentrenados para evitar
descargas de red. El TestClient de FastAPI simula requests HTTP sin levantar
un servidor real.

Ejecutar con:
    pytest tests/test_api.py -v
"""

import cv2
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app, APIConfig
from model import RetinaModel, ModelConfig


# Helpers y fixtures

def make_image_bytes(h: int = 100, w: int = 100, fmt: str = ".png") -> bytes:
    """Imagen PNG/JPEG sintética codificada en bytes para upload."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, r = w // 2, h // 2, min(w, h) // 2 - 5
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = [60, 80, 180]
    _, buffer = cv2.imencode(fmt, img)
    return buffer.tobytes()


@pytest.fixture(scope="module")
def client():
    """
    TestClient con modelo sin pesos preentrenados.
    Usado como context manager para que el lifespan (carga del modelo) se ejecute.
    """
    model = RetinaModel(ModelConfig(pretrained=False, freeze_backbone=False))
    app = create_app(model=model)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def png_image():
    return make_image_bytes(fmt=".png")


@pytest.fixture
def jpg_image():
    return make_image_bytes(fmt=".jpg")

# Tests de GET /health

class TestHealthEndpoint:

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_response_has_device(self, client):
        data = client.get("/health").json()
        assert "device" in data

    def test_response_has_model_type(self, client):
        data = client.get("/health").json()
        assert "model_type" in data
        assert data["model_type"] == "efficientnet_b4"


# Tests de POST /predict

class TestPredictEndpoint:

    def test_returns_200_with_valid_png(self, client, png_image):
        response = client.post(
            "/predict",
            files={"file": ("retina.png", png_image, "image/png")},
        )
        assert response.status_code == 200

    def test_returns_200_with_valid_jpg(self, client, jpg_image):
        response = client.post(
            "/predict",
            files={"file": ("retina.jpg", jpg_image, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_has_predicted_class(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert "predicted_class" in data

    def test_predicted_class_in_valid_range(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert 0 <= data["predicted_class"] <= 4

    def test_response_has_class_name(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert data["class_name"] in [
            "No DR", "Mild", "Moderate", "Severe", "Proliferative"
        ]

    def test_probabilities_length_is_5(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert len(data["probabilities"]) == 5

    def test_probabilities_sum_to_1(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        total = sum(data["probabilities"])
        assert abs(total - 1.0) < 1e-5

    def test_confidence_matches_max_probability(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert abs(data["confidence"] - max(data["probabilities"])) < 1e-5

    def test_confidence_in_valid_range(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_referral_recommended_false_for_class_0_or_1(self, client):
        """
        Con un modelo no entrenado no podemos garantizar la clase predicha,
        pero sí podemos verificar la lógica: si clase < 2, no referir.
        """
        data = client.post(
            "/predict",
            files={"file": ("img.png", make_image_bytes())},
        ).json()
        expected = data["predicted_class"] >= 2
        assert data["referral_recommended"] == expected

    def test_response_has_clinical_note(self, client, png_image):
        data = client.post(
            "/predict",
            files={"file": ("img.png", png_image)},
        ).json()
        assert "clinical_note" in data
        assert len(data["clinical_note"]) > 0

    def test_invalid_file_returns_400(self, client):
        response = client.post(
            "/predict",
            files={"file": ("texto.txt", b"esto no es una imagen", "text/plain")},
        )
        assert response.status_code == 400

    def test_empty_file_returns_400(self, client):
        response = client.post(
            "/predict",
            files={"file": ("vacío.png", b"", "image/png")},
        )
        assert response.status_code == 400


# Tests de POST /explain

class TestExplainEndpoint:

    def test_returns_200(self, client, png_image):
        response = client.post(
            "/explain",
            files={"file": ("retina.png", png_image, "image/png")},
        )
        assert response.status_code == 200

    def test_returns_jpeg_content_type(self, client, png_image):
        response = client.post(
            "/explain",
            files={"file": ("retina.png", png_image)},
        )
        assert response.headers["content-type"] == "image/jpeg"

    def test_response_is_valid_jpeg(self, client, png_image):
        """El contenido retornado debe ser una imagen JPEG decodificable."""
        response = client.post(
            "/explain",
            files={"file": ("retina.png", png_image)},
        )
        nparr = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None

    def test_explain_invalid_file_returns_400(self, client):
        response = client.post(
            "/explain",
            files={"file": ("texto.txt", b"no es imagen", "text/plain")},
        )
        assert response.status_code == 400


# Tests de APIConfig

class TestAPIConfig:

    def test_default_image_size(self):
        assert APIConfig().image_size == 380

    def test_default_model_type(self):
        assert APIConfig().model_type == "efficientnet_b4"

    def test_device_is_valid_string(self):
        device = APIConfig().device
        assert device in ("cuda", "mps", "cpu")

    def test_custom_values(self):
        config = APIConfig(image_size=224, device="cpu")
        assert config.image_size == 224
        assert config.device == "cpu"
