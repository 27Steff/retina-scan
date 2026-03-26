"""
api.py
API REST para detección de retinopatía diabética.

Endpoints: GET /health, POST /predict (clasificación JSON), POST /explain (Grad-CAM JPEG).

Decisiones de diseño:
    create_app() es una factory — permite inyectar modelo en tests sin checkpoint en disco.
    lifespan carga el modelo una sola vez al iniciar (EfficientNet-B4 tarda ~2s por carga).
    Endpoints síncronos (def, no async) — FastAPI los ejecuta en thread pool,
    evitando que la inferencia bloquee el event loop.
"""

import cv2
from pathlib import Path
import numpy as np
import torch
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from explainability import GradCAM
from model import RetinaModel, make_model
from preprocessing import RetinaPreprocessor, PreprocessConfig


_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
_CLASS_CLINICAL = [
    "No diabetic retinopathy detected.",
    "Mild diabetic retinopathy. Annual follow-up recommended.",
    "Moderate diabetic retinopathy. Referral to ophthalmologist recommended.",
    "Severe diabetic retinopathy. Urgent referral required.",
    "Proliferative diabetic retinopathy. Emergency ophthalmologic evaluation required.",
]


class PredictionResponse(BaseModel):
    """Respuesta del endpoint /predict."""
    predicted_class: int
    class_name: str
    clinical_note: str
    confidence: float
    probabilities: List[float]
    referral_recommended: bool


class HealthResponse(BaseModel):
    """Respuesta del endpoint /health."""
    status: str
    model_loaded: bool
    model_type: str
    device: str


@dataclass
class APIConfig:
    """
    Configuración del servidor.
    checkpoint=None usa pesos de ImageNet sin fine-tuning (útil en dev).
    image_size debe coincidir con el tamaño usado en entrenamiento.
    """
    checkpoint: Optional[str] = None
    model_type: str = "efficientnet_b4"
    device: str = field(
        default_factory=lambda: (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    image_size: int = 380


def create_app(
    config: Optional[APIConfig] = None,
    model: Optional[RetinaModel] = None,
) -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    Si model se provee, se usa directamente (útil en tests para evitar descarga de pesos).
    """
    config = config or APIConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if model is not None:
            app.state.model = model
        else:
            app.state.model = _load_model(config)

        app.state.model.eval()
        app.state.model.to(config.device)
        app.state.preprocessor = RetinaPreprocessor(
            PreprocessConfig(image_size=config.image_size, normalize=True)
        )
        app.state.config = config
        yield

    app = FastAPI(
        title="RetinaScan API",
        description=(
            "Detección de retinopatía diabética a partir de fotografías de fondo de ojo. "
            "Clasifica en 5 grados de severidad y provee mapas de calor Grad-CAM "
            "para validación clínica."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Estado del servidor y del modelo."""
        return HealthResponse(
            status="ok",
            model_loaded=hasattr(app.state, "model"),
            model_type=config.model_type,
            device=config.device,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(file: UploadFile = File(...)):
        """Clasifica una imagen de fondo de ojo en 5 grados de retinopatía."""
        img = _read_image(file)
        tensor = _preprocess(img, app.state.preprocessor, config.device)

        with torch.no_grad():
            logits = app.state.model(tensor)

        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted = int(probs.argmax())

        return PredictionResponse(
            predicted_class=predicted,
            class_name=_CLASS_NAMES[predicted],
            clinical_note=_CLASS_CLINICAL[predicted],
            confidence=float(probs[predicted]),
            probabilities=[float(p) for p in probs],
            referral_recommended=predicted >= 2,
        )

    @app.post("/explain")
    def explain(file: UploadFile = File(...)):
        """Retorna la imagen con Grad-CAM superpuesto — para validación clínica de la predicción."""
        img = _read_image(file)
        tensor = _preprocess_no_norm(img, config)

        with GradCAM(app.state.model) as cam:
            heatmap = cam.generate(tensor)

        img_resized = cv2.resize(img, (config.image_size, config.image_size))
        overlay = cam.overlay(heatmap, img_resized)

        img_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    return app


app = create_app(APIConfig(
    checkpoint="checkpoints/best_model.pt",
    model_type="efficientnet_b0",
    image_size=224,
))


def _load_model(config: APIConfig) -> RetinaModel:
    """Carga desde checkpoint si existe; si no, inicializa sin pesos preentrenados."""
    if config.checkpoint and Path(config.checkpoint).exists():
        model = make_model(config.model_type)
        ckpt = torch.load(config.checkpoint, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        if config.checkpoint:
            print(f"Warning: checkpoint '{config.checkpoint}' not found. Run train.py first.")
        model = make_model(config.model_type, pretrained=False)

    return model


def _read_image(file: UploadFile) -> np.ndarray:
    """Lee el archivo subido y lo decodifica como imagen RGB. Lanza HTTP 400 si no es válida."""
    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    nparr = np.frombuffer(contents, dtype=np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo decodificar la imagen. "
                   "Formatos soportados: JPEG, PNG, TIFF.",
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _preprocess(
    img: np.ndarray,
    preprocessor: RetinaPreprocessor,
    device: str,
) -> torch.Tensor:
    """Preprocesa imagen y la convierte a tensor (1, 3, H, W) float32."""
    processed = preprocessor.process_array(img)
    tensor = preprocessor.to_tensor(processed)
    return torch.from_numpy(tensor).unsqueeze(0).to(device)


def _preprocess_no_norm(img: np.ndarray, config: APIConfig) -> torch.Tensor:
    """Preprocesa sin normalización — Grad-CAM necesita float32 pero sin ImageNet stats."""
    preprocessor = RetinaPreprocessor(
        PreprocessConfig(image_size=config.image_size, normalize=False)
    )
    processed = preprocessor.process_array(img)
    tensor = preprocessor.to_tensor(processed)
    return torch.from_numpy(tensor / 255.0).unsqueeze(0).float()
