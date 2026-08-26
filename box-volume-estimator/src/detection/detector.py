import cv2
import numpy as np
import torch
from ultralytics import FastSAM, YOLOWorld

from src.config import settings


class PreciseBoxDetector:

    def __init__(
        self,
        yolo_world_model: str = settings.YOLO_WORLD_MODEL,
        fastsam_model: str = settings.FASTSAM_MODEL,
        target_classes: list[str] = settings.BOX_TARGET_CLASSES,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        # 1. Загрузка YOLO-World для поиска по тексту
        self.detector = YOLOWorld(yolo_world_model)
        self.detector.to(self.device)
        self.detector.set_classes(target_classes)

        # 2. Загрузка FastSAM для получения маски по bounding box
        self.segmenter = FastSAM(fastsam_model)

    def extract_box_mask(
        self, image_bgr: np.ndarray, conf: float = 0.25
    ) -> np.ndarray | None:
        """Находит коробку через текстовый поиск и строит точную маску через Box Prompt."""
        h, w = image_bgr.shape[:2]

        # Шаг 1: Open-vocabulary детекция bounding box через YOLO-World
        detection_results = self.detector(
            image_bgr, conf=conf, verbose=False, device=self.device
        )[0]

        if len(detection_results.boxes) == 0:
            return None

        # Сортируем боксы по confidence и берем самый уверенный
        boxes = detection_results.boxes.xyxy.cpu().numpy()
        confs = detection_results.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_box = boxes[best_idx]  # [x1, y1, x2, y2]

        # Шаг 2: Генерация всех масок через FastSAM
        seg_results = self.segmenter(
            image_bgr,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )

        # Шаг 3: Фильтрация маски с использованием bounding box как промпта
        prompt_process = seg_results[0]
        # bboxes принимает список [x1, y1, x2, y2]
        prompt_result = prompt_process.prompt(
            bboxes=[best_box.tolist()]
        )

        if prompt_result is None or len(prompt_result) == 0:
            # Fallback: если FastSAM не выделил маску, возвращаем прямоугольную маску из YOLO-World
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = best_box.astype(int)
            mask[y1:y2, x1:x2] = 255
            return mask

        # Извлекаем тензор маски
        masks = prompt_result[0].masks.data.cpu().numpy()
        if len(masks) == 0:
            return None

        # Берем результирующую бинарную маску
        raw_mask = masks[0]
        mask_resized = cv2.resize(
            raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )

        return (mask_resized > 0).astype(np.uint8) * 255