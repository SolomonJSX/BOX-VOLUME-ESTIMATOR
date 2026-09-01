from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch

from src.detection.detector import PreciseBoxDetector

@pytest.fixture
def synthetic_box_image() -> np.ndarray:
    """Генерирует синтетическое RGB-изображение (640x640)

    с прямоугольной 'коробкой' по центру на темном фоне.
    """
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    box_color = (180, 150, 100)
    cv2.rectangle(img, (160, 160), (480, 480), box_color, -1)

    cv2.line(img, (320, 160), (320, 480), (0, 0, 220), 8)
    cv2.line(img, (160, 320), (480, 320), (0, 0, 220), 8)

    return img

@pytest.fixture
def empty_image() -> np.ndarray:
    """Синтетическое пустое серое изображение без выраженных объектов."""
    return np.full((640, 640, 3), 128, dtype=np.uint8)

class TestPreciseBoxDetectorUnit:
    """Модульные тесты с моками для быстрой валидации логики без загрузки весов."""
    @patch("src.detection.detector.YOLOWorld")
    @patch("src.detection.detector.FastSAM")
    def test_extract_box_mask_success_with_fastsam(self, mock_fastsam_cls, mock_yolo_cls, synthetic_box_image):
        # Настройка мока YOLO-World
        mock_yolo = MagicMock()
        mock_yolo_cls.return_value = mock_yolo

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[160.0, 160.0, 480.0, 480.0]])
        mock_boxes.conf = torch.tensor([0.85])
        mock_boxes.__len__.return_value = 1

        mock_detection_result = MagicMock()
        mock_detection_result.boxes = mock_boxes
        mock_yolo.return_value = [mock_detection_result]

        # Настройка мока FastSAM
        mock_fastsam = MagicMock()
        mock_fastsam_cls.return_value = mock_fastsam

        synthetic_raw_mask = np.zeros((640, 640), dtype=np.uint8)
        synthetic_raw_mask[160:480, 160:480] = 1

        mock_seg_result = MagicMock()
        mock_seg_result.masks.data = torch.from_numpy(
            synthetic_raw_mask
        ).unsqueeze(0)
        mock_seg_result.boxes.xyxy = torch.tensor([[160.0, 160.0, 480.0, 480.0]])
        mock_fastsam.return_value = [mock_seg_result]

        # Инициализация и вызов
        detector = PreciseBoxDetector(device="cpu")
        mask = detector.extract_box_mask(synthetic_box_image, conf=0.25)

        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.shape == synthetic_box_image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.max(mask) == 255
        assert np.min(mask) == 0

        assert mask[320, 320] == 255
        assert mask[50, 50] == 0

    @patch("src.detection.detector.YOLOWorld")
    @patch("src.detection.detector.FastSAM")
    def test_extract_box_mask_no_detections(
        self, mock_fastsam_cls, mock_yolo_cls, empty_image
    ):
        mock_yolo = MagicMock()
        mock_yolo_cls.return_value = mock_yolo

        # Пустые детекции от YOLO
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 0
        mock_detection_result = MagicMock()
        mock_detection_result.boxes = mock_boxes
        mock_yolo.return_value = [mock_detection_result]

        mock_fastsam = MagicMock()
        mock_fastsam_cls.return_value = mock_fastsam
        mock_fastsam.return_value = []

        detector = PreciseBoxDetector(device="cpu")
        mask = detector.extract_box_mask(empty_image, conf=0.25)

        assert mask is None

@pytest.mark.integration
class TestPreciseBoxDetectorIntegration:
    """Интеграционный тест с реальным инференсом нейросетей (требует загрузки весов)."""

    def test_real_inference_on_synthetic_data(self, synthetic_box_image):
        detector = PreciseBoxDetector(device="cpu")
        mask = detector.extract_box_mask(synthetic_box_image, conf=0.1)

        if mask is not None:
            assert mask.shape == (640, 640)
            assert mask.dtype == np.uint8
            assert set(np.unique(mask)).issubset({0, 255})