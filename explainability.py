"""
explainability.py
Grad-CAM para visualizar qué regiones de la retina activan la predicción.

Crítico en medicina: un modelo que predice "grado 3" sin mostrar dónde vio las
lesiones no puede ser validado clínicamente. También es la primera señal de
overfitting espacial — si el mapa apunta al borde negro, el modelo aprendió un
artefacto, no una lesión.
Algoritmo: gradientes ∂score/∂features → GAP por canal → suma ponderada → ReLU → normalizar.
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from model import RetinaModel

_COLORMAPS = {
    "jet":     cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot":     cv2.COLORMAP_HOT,
}


@dataclass
class GradCAMConfig:
    """
    alpha: transparencia del heatmap en el overlay (0=solo imagen, 1=solo heatmap).
    colormap: "jet" (azul→rojo), "inferno" (negro→amarillo), "hot".
    """
    alpha: float = 0.4
    colormap: str = "jet"


class GradCAM:
    """
    Grad-CAM para EfficientNet-B4 (cualquier modelo con backbone.features).
    Los hooks deben liberarse: cam.remove() o usar como context manager.
    """

    def __init__(
        self,
        model: RetinaModel,
        config: Optional[GradCAMConfig] = None,
        target_layer: Optional[torch.nn.Module] = None,
    ):
        self.model = model
        self.config = config or GradCAMConfig()
        self.target_layer = target_layer or self._find_target_layer()

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks = self._register_hooks()

    def generate(
        self,
        img_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Genera el heatmap Grad-CAM para una imagen.
        img_tensor: (1, 3, H, W) o (3, H, W). class_idx=None usa la clase predicha.
        Retorna ndarray (H, W) float32 en [0, 1].
        """
        img_tensor = self._prepare_tensor(img_tensor)

        self.model.eval()
        self.model.zero_grad()

        logits = self.model(img_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        return self._compute_heatmap(self._gradients, self._activations)

    def overlay(
        self,
        heatmap: np.ndarray,
        img_uint8: np.ndarray,
    ) -> np.ndarray:
        """Superpone el heatmap sobre la imagen original. Retorna uint8 (H, W, 3)."""
        h, w = img_uint8.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        colormap = _COLORMAPS.get(self.config.colormap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap,
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(
            img_uint8, 1.0 - self.config.alpha,
            heatmap_colored, self.config.alpha,
            0,
        )

    def remove(self) -> None:
        """Elimina los hooks para liberar memoria."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()

    def _find_target_layer(self) -> torch.nn.Module:
        """
        features[-1] es el último bloque conv de EfficientNet — semánticamente
        rico y con suficiente resolución espacial para localizar lesiones.
        """
        return self.model.backbone.features[-1]

    def _register_hooks(self) -> list:
        """Registra forward hook (activaciones) y backward hook (gradientes)."""
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        return [h_fwd, h_bwd]

    def _prepare_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Añade dimensión de batch si necesario y habilita gradientes."""
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.dim() != 4:
            raise ValueError(
                f"img_tensor debe ser (3, H, W) o (1, 3, H, W). "
                f"Recibido: {img_tensor.shape}"
            )
        device = next(self.model.parameters()).device
        return img_tensor.float().to(device).requires_grad_(True)

    def _compute_heatmap(
        self,
        gradients: torch.Tensor,
        activations: torch.Tensor,
    ) -> np.ndarray:
        """GAP sobre gradientes → pesos por canal → suma ponderada → ReLU → normalizar a [0,1]."""
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * activations).sum(dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        h_max = heatmap.max()
        if h_max > 0:
            heatmap = heatmap / h_max
        return heatmap.astype(np.float32)


def make_explainer(kind: str, model: RetinaModel, **kwargs) -> GradCAM:
    """
    Factory para objetos de explicabilidad.

    kind: "gradcam" (único soportado actualmente)
    **kwargs sobreescribe campos de GradCAMConfig.
    """
    registry = {
        "gradcam": GradCAMConfig,
    }

    if kind not in registry:
        raise ValueError(
            f"Unknown explainer kind '{kind}'. Options: {list(registry)}"
        )

    config = registry[kind](**kwargs)
    return GradCAM(model, config)
