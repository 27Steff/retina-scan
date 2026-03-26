"""
segmentation.py
Segmentación de lesiones retinales usando SAM (Segment Anything Model, Meta).

Pipeline Grad-CAM → SAM: Grad-CAM localiza la lesión (peak de activación),
SAM delimita su contorno preciso. Ninguno solo es suficiente — SAM no sabe qué
buscar en retina; Grad-CAM no produce contornos. Juntos permiten medir área de lesión.

Dependency injection: SAM requiere ~375 MB de pesos que no se pueden descargar en tests.
El predictor se inyecta externamente en lugar de cargarse siempre en __init__.
Pesos: https://github.com/facebookresearch/segment-anything#model-checkpoints
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False

ImageArray = np.ndarray  # shape (H, W, 3), dtype uint8

_MASK_COLORS = [
    [255,  80,  80],
    [ 80, 200,  80],
    [ 80, 130, 255],
    [255, 220,  50],
    [220,  80, 220],
]


@dataclass
class SegmentationConfig:
    """
    model_type: "vit_b" (375 MB, recomendado para retina), "vit_l" (1.2 GB), "vit_h" (2.4 GB).
    min_mask_area: filtra artefactos pequeños (100 px razonable para 380×380).
    pred_iou_thresh: IoU mínimo predicho por SAM para aceptar una máscara.
    """
    model_type: str = "vit_b"
    checkpoint: Optional[str] = None
    device: str = "cpu"
    min_mask_area: int = 100
    pred_iou_thresh: float = 0.88
    overlay_alpha: float = 0.4


class RetinaSegmentor:
    """
    Segmentador de lesiones retinales usando SAM con prompts de Grad-CAM.
    Si se provee checkpoint en config, carga SAM automáticamente;
    si no, el predictor debe inyectarse como argumento.
    """

    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
        predictor=None,
    ):
        self.config = config or SegmentationConfig()
        self.predictor = predictor

        if self.predictor is None and self.config.checkpoint:
            self.predictor = self._load_predictor()

    def from_heatmap(
        self,
        img_uint8: ImageArray,
        heatmap: np.ndarray,
    ) -> List[dict]:
        """
        Segmenta la lesión más activante del heatmap de Grad-CAM.
        Extrae el peak de activación y lo usa como prompt de foreground para SAM.
        Retorna lista de dicts {"mask", "score", "area"} ordenada por score desc.
        """
        peak_xy = self._peak_from_heatmap(heatmap, img_uint8.shape[:2])
        return self.from_point(img_uint8, peak_xy)

    def from_point(
        self,
        img_uint8: ImageArray,
        point_xy: Tuple[int, int],
        foreground: bool = True,
    ) -> List[dict]:
        """
        Segmenta usando un punto como prompt (x, y) en coordenadas de imagen.
        Retorna lista de dicts [{"mask": array, "score": float, "area": int}, ...].
        """
        self._require_predictor()
        self.predictor.set_image(img_uint8)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([point_xy]),
            point_labels=np.array([1 if foreground else 0]),
            multimask_output=True,
        )
        return self._build_mask_dicts(masks, scores)

    def from_box(
        self,
        img_uint8: ImageArray,
        box_xyxy: Tuple[int, int, int, int],
    ) -> List[dict]:
        """
        Segmenta dentro de un bounding box (x1, y1, x2, y2).
        Útil cuando el clínico puede marcar la región de interés.
        """
        self._require_predictor()
        self.predictor.set_image(img_uint8)
        masks, scores, _ = self.predictor.predict(
            box=np.array(box_xyxy, dtype=np.float32),
            multimask_output=False,
        )
        return self._build_mask_dicts(masks, scores)

    def overlay_masks(
        self,
        img_uint8: ImageArray,
        masks: List[dict],
    ) -> ImageArray:
        """Superpone las máscaras sobre la imagen con colores distintos por máscara."""
        result = img_uint8.copy()
        alpha = self.config.overlay_alpha

        for i, mask_dict in enumerate(masks):
            color = np.array(_MASK_COLORS[i % len(_MASK_COLORS)], dtype=np.uint8)
            colored = np.zeros_like(img_uint8)
            colored[mask_dict["mask"]] = color
            result = cv2.addWeighted(result, 1.0 - alpha, colored, alpha, 0)

        return result

    def _peak_from_heatmap(
        self,
        heatmap: np.ndarray,
        img_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Encuentra el punto de mayor activación y lo escala al tamaño de imagen. Retorna (x, y)."""
        h_map, w_map = heatmap.shape
        h_img, w_img = img_size
        peak_row, peak_col = np.unravel_index(heatmap.argmax(), heatmap.shape)
        x = int(peak_col / w_map * w_img)
        y = int(peak_row / h_map * h_img)
        return (x, y)

    def _build_mask_dicts(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> List[dict]:
        """Filtra por área y pred_iou_thresh, ordena por score descendente."""
        result = []
        for mask, score in zip(masks, scores):
            area = int(mask.sum())
            if area >= self.config.min_mask_area and float(score) >= self.config.pred_iou_thresh:
                result.append({
                    "mask":  mask.astype(bool),
                    "score": float(score),
                    "area":  area,
                })
        return sorted(result, key=lambda x: x["score"], reverse=True)

    def _require_predictor(self) -> None:
        """Lanza RuntimeError si el predictor no está cargado."""
        if self.predictor is None:
            raise RuntimeError(
                "SAM predictor no está cargado. "
                "Opciones:\n"
                "  1. Pasa checkpoint en SegmentationConfig: "
                "SegmentationConfig(checkpoint='sam_vit_b.pth')\n"
                "  2. Inyecta un predictor: RetinaSegmentor(predictor=predictor)\n"
                "  Descarga pesos en: "
                "https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )

    def _load_predictor(self):
        """Carga SAM desde disco. Requiere que segment_anything esté instalado."""
        if not _SAM_AVAILABLE:
            raise ImportError(
                "segment_anything no está instalado.\n"
                "Instalar con: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        sam = sam_model_registry[self.config.model_type](
            checkpoint=self.config.checkpoint
        )
        sam.to(self.config.device)
        return SamPredictor(sam)


def make_segmentor(kind: str, **kwargs) -> RetinaSegmentor:
    """
    Factory para segmentores SAM.

    kind: "vit_b" (recomendado) | "vit_l" | "vit_h"
    Pasar checkpoint= con la ruta a los pesos descargados.
    """
    from dataclasses import replace

    registry = {
        "vit_b": SegmentationConfig(model_type="vit_b"),
        "vit_l": SegmentationConfig(model_type="vit_l"),
        "vit_h": SegmentationConfig(model_type="vit_h"),
    }

    if kind not in registry:
        raise ValueError(
            f"Unknown segmentor kind '{kind}'. Options: {list(registry)}"
        )

    config = replace(registry[kind], **kwargs)
    return RetinaSegmentor(config)
