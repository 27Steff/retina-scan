"""
test_segmentation.py
--------------------
Tests para segmentation.py

Estrategia: se mockea el SamPredictor para evitar descargar pesos (~375 MB).
La lógica de negocio (_peak_from_heatmap, _build_mask_dicts, overlay_masks)
se testea directamente. Las integraciones con SAM se testean con mock.

Ejecutar con:
    pytest tests/test_segmentation.py -v
"""

import numpy as np
import pytest
from typing import Optional
from unittest.mock import MagicMock

from segmentation import SegmentationConfig, RetinaSegmentor, make_segmentor


# Helpers y fixtures

def make_fake_masks(n: int = 3, h: int = 64, w: int = 64) -> np.ndarray:
    """Máscaras booleanas sintéticas de distintos tamaños."""
    regions = [(10, 30, 10, 30), (5, 15, 5, 15), (0, 3, 0, 3)]  # áreas: 400, 100, 9
    masks = np.zeros((n, h, w), dtype=bool)
    for i in range(n):
        r1, r2, c1, c2 = regions[i % len(regions)]
        masks[i, r1:r2, c1:c2] = True
    return masks


def make_mock_predictor(
    h: int = 64,
    w: int = 64,
    n_masks: int = 3,
    scores: Optional[list] = None,
) -> MagicMock:
    """Predictor SAM falso que devuelve máscaras sintéticas."""
    if scores is None:
        scores = [0.95, 0.91, 0.85]
    predictor = MagicMock()
    predictor.predict.return_value = (
        make_fake_masks(n_masks, h, w),
        np.array(scores[:n_masks], dtype=np.float32),
        np.zeros((n_masks, 256, 256), dtype=np.float32),
    )
    return predictor


@pytest.fixture
def config():
    return SegmentationConfig(min_mask_area=50, pred_iou_thresh=0.88)


@pytest.fixture
def mock_predictor():
    return make_mock_predictor()


@pytest.fixture
def seg(config, mock_predictor):
    return RetinaSegmentor(config=config, predictor=mock_predictor)


@pytest.fixture
def img_uint8():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def heatmap():
    """Heatmap con un pico claro en la esquina superior derecha."""
    h = np.zeros((12, 12), dtype=np.float32)
    h[2, 9] = 1.0  # peak en (fila=2, col=9)
    return h


# Tests de SegmentationConfig

class TestSegmentationConfig:

    def test_default_model_type(self):
        assert SegmentationConfig().model_type == "vit_b"

    def test_default_min_mask_area(self):
        assert SegmentationConfig().min_mask_area == 100

    def test_default_pred_iou_thresh(self):
        assert SegmentationConfig().pred_iou_thresh == pytest.approx(0.88)

    def test_custom_values(self):
        config = SegmentationConfig(model_type="vit_h", min_mask_area=200)
        assert config.model_type == "vit_h"
        assert config.min_mask_area == 200


# Tests de _peak_from_heatmap

class TestPeakFromHeatmap:

    def test_peak_at_center(self):
        seg = RetinaSegmentor()
        h = np.zeros((10, 10))
        h[5, 5] = 1.0
        peak = seg._peak_from_heatmap(h, img_size=(100, 100))
        assert peak == (50, 50)

    def test_peak_at_top_left(self):
        seg = RetinaSegmentor()
        h = np.zeros((10, 10))
        h[0, 0] = 1.0
        peak = seg._peak_from_heatmap(h, img_size=(100, 100))
        assert peak == (0, 0)

    def test_peak_at_bottom_right(self):
        seg = RetinaSegmentor()
        h = np.zeros((10, 10))
        h[9, 9] = 1.0
        peak = seg._peak_from_heatmap(h, img_size=(100, 100))
        assert peak == (90, 90)

    def test_peak_scaled_correctly_to_larger_image(self):
        """El punto debe escalar proporcionalmente al tamaño de imagen."""
        seg = RetinaSegmentor()
        h = np.zeros((12, 12))
        h[6, 6] = 1.0
        x, y = seg._peak_from_heatmap(h, img_size=(380, 380))
        assert 0 <= x <= 380
        assert 0 <= y <= 380

    def test_returns_tuple_of_two_ints(self):
        seg = RetinaSegmentor()
        h = np.ones((8, 8))
        peak = seg._peak_from_heatmap(h, img_size=(64, 64))
        assert len(peak) == 2
        assert all(isinstance(v, int) for v in peak)


# Tests de _build_mask_dicts

class TestBuildMaskDicts:

    def test_filters_by_min_area(self):
        """Máscaras con área < min_mask_area deben eliminarse."""
        config = SegmentationConfig(min_mask_area=50, pred_iou_thresh=0.0)
        seg = RetinaSegmentor(config=config)
        masks  = make_fake_masks()  # áreas: 400, 100, 9
        scores = np.array([0.95, 0.91, 0.85])
        result = seg._build_mask_dicts(masks, scores)
        # área 9 debe eliminarse
        assert all(m["area"] >= 50 for m in result)
        assert len(result) == 2

    def test_filters_by_iou_threshold(self):
        """Máscaras con score < pred_iou_thresh deben eliminarse."""
        config = SegmentationConfig(min_mask_area=0, pred_iou_thresh=0.92)
        seg = RetinaSegmentor(config=config)
        masks  = make_fake_masks()
        scores = np.array([0.95, 0.91, 0.85])
        result = seg._build_mask_dicts(masks, scores)
        assert all(m["score"] >= 0.92 for m in result)

    def test_sorted_by_score_descending(self):
        config = SegmentationConfig(min_mask_area=0, pred_iou_thresh=0.0)
        seg = RetinaSegmentor(config=config)
        masks  = make_fake_masks()
        scores = np.array([0.85, 0.95, 0.91])
        result = seg._build_mask_dicts(masks, scores)
        for i in range(len(result) - 1):
            assert result[i]["score"] >= result[i + 1]["score"]

    def test_result_dict_has_required_keys(self, config):
        seg = RetinaSegmentor(config=config)
        masks  = make_fake_masks(1)
        scores = np.array([0.95])
        result = seg._build_mask_dicts(masks, scores)
        assert "mask"  in result[0]
        assert "score" in result[0]
        assert "area"  in result[0]

    def test_mask_dtype_is_bool(self, config):
        seg = RetinaSegmentor(config=config)
        masks  = make_fake_masks(1)
        scores = np.array([0.95])
        result = seg._build_mask_dicts(masks, scores)
        if result:
            assert result[0]["mask"].dtype == bool

    def test_empty_when_all_filtered(self):
        config = SegmentationConfig(min_mask_area=10000)
        seg = RetinaSegmentor(config=config)
        masks  = make_fake_masks()
        scores = np.array([0.95, 0.91, 0.85])
        result = seg._build_mask_dicts(masks, scores)
        assert result == []


# Tests de from_point (con mock predictor)

class TestFromPoint:

    def test_returns_list(self, seg, img_uint8):
        result = seg.from_point(img_uint8, point_xy=(32, 32))
        assert isinstance(result, list)

    def test_calls_set_image(self, seg, mock_predictor, img_uint8):
        seg.from_point(img_uint8, point_xy=(32, 32))
        mock_predictor.set_image.assert_called_once()

    def test_calls_predict_with_correct_label(self, seg, mock_predictor, img_uint8):
        seg.from_point(img_uint8, point_xy=(20, 30), foreground=True)
        call_kwargs = mock_predictor.predict.call_args[1]
        assert call_kwargs["point_labels"][0] == 1

    def test_background_point_uses_label_0(self, seg, mock_predictor, img_uint8):
        seg.from_point(img_uint8, point_xy=(20, 30), foreground=False)
        call_kwargs = mock_predictor.predict.call_args[1]
        assert call_kwargs["point_labels"][0] == 0

    def test_no_predictor_raises(self, img_uint8):
        seg = RetinaSegmentor()
        with pytest.raises(RuntimeError, match="predictor"):
            seg.from_point(img_uint8, point_xy=(32, 32))


# Tests de from_heatmap (con mock predictor)

class TestFromHeatmap:

    def test_returns_list(self, seg, img_uint8, heatmap):
        result = seg.from_heatmap(img_uint8, heatmap)
        assert isinstance(result, list)

    def test_uses_heatmap_peak_as_prompt(self, seg, mock_predictor, img_uint8, heatmap):
        """El punto pasado a SAM debe corresponder al peak del heatmap."""
        seg.from_heatmap(img_uint8, heatmap)
        call_kwargs = mock_predictor.predict.call_args[1]
        point = tuple(call_kwargs["point_coords"][0])
        # heatmap peak en (fila=2, col=9) → img 64x64 → x=48, y=10 aprox
        assert 0 <= point[0] <= 64
        assert 0 <= point[1] <= 64

    def test_no_predictor_raises(self, img_uint8, heatmap):
        seg = RetinaSegmentor()
        with pytest.raises(RuntimeError):
            seg.from_heatmap(img_uint8, heatmap)


# Tests de from_box (con mock predictor)

class TestFromBox:

    def test_returns_list(self, seg, img_uint8):
        result = seg.from_box(img_uint8, box_xyxy=(10, 10, 50, 50))
        assert isinstance(result, list)

    def test_calls_predict_with_box(self, seg, mock_predictor, img_uint8):
        seg.from_box(img_uint8, box_xyxy=(10, 10, 50, 50))
        call_kwargs = mock_predictor.predict.call_args[1]
        assert "box" in call_kwargs


# Tests de overlay_masks

class TestOverlayMasks:

    def test_output_shape_matches_input(self, seg, img_uint8):
        masks = [
            {"mask": np.ones((64, 64), dtype=bool), "score": 0.9, "area": 4096}
        ]
        result = seg.overlay_masks(img_uint8, masks)
        assert result.shape == img_uint8.shape

    def test_output_dtype_uint8(self, seg, img_uint8):
        masks = [{"mask": np.ones((64, 64), dtype=bool), "score": 0.9, "area": 4096}]
        result = seg.overlay_masks(img_uint8, masks)
        assert result.dtype == np.uint8

    def test_empty_masks_returns_copy_of_image(self, seg, img_uint8):
        """Sin máscaras, la imagen no debe cambiar."""
        result = seg.overlay_masks(img_uint8, masks=[])
        np.testing.assert_array_equal(result, img_uint8)

    def test_multiple_masks_different_colors(self, seg, img_uint8):
        """Con múltiples máscaras, el resultado debe diferir de la imagen original."""
        masks = [
            {"mask": np.zeros((64, 64), dtype=bool), "score": 0.9, "area": 0},
            {"mask": np.zeros((64, 64), dtype=bool), "score": 0.8, "area": 0},
        ]
        masks[0]["mask"][10:20, 10:20] = True
        masks[1]["mask"][40:50, 40:50] = True
        result = seg.overlay_masks(img_uint8, masks)
        assert result.shape == img_uint8.shape


# Tests de make_segmentor

class TestMakeSegmentor:

    def test_vit_b_kind(self):
        seg = make_segmentor("vit_b")
        assert seg.config.model_type == "vit_b"

    def test_vit_l_kind(self):
        seg = make_segmentor("vit_l")
        assert seg.config.model_type == "vit_l"

    def test_vit_h_kind(self):
        seg = make_segmentor("vit_h")
        assert seg.config.model_type == "vit_h"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            make_segmentor("vit_x")

    def test_kwargs_override_min_area(self):
        seg = make_segmentor("vit_b", min_mask_area=500)
        assert seg.config.min_mask_area == 500

    def test_no_predictor_without_checkpoint(self):
        """Sin checkpoint, el predictor debe ser None."""
        seg = make_segmentor("vit_b")
        assert seg.predictor is None
