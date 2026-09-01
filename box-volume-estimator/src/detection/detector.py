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
        self.target_classes = target_classes

        # 1. Загрузка YOLO-World для поиска по тексту
        self.detector = YOLOWorld(yolo_world_model)
        self.detector.to(self.device)
        try:
            self.detector.set_classes(target_classes)
        except Exception:
            pass

        # 2. Загрузка FastSAM для получения точных масок сегментации
        self.segmenter = FastSAM(fastsam_model)

    def extract_box_mask(
        self, image_bgr: np.ndarray, conf: float = 0.20
    ) -> np.ndarray | None:
        """Находит коробку через YOLO-World и уточняет границы через FastSAM с надежными fallback-стратегиями."""
        h, w = image_bgr.shape[:2]
        img_area = float(h * w)

        # Шаг 1: Open-vocabulary детекция bounding box через YOLO-World
        best_box = None
        try:
            detection_results = self.detector(
                image_bgr, conf=conf, verbose=False, device=self.device
            )[0]

            if len(detection_results.boxes) > 0:
                boxes = detection_results.boxes.xyxy.cpu().numpy()
                confs = detection_results.boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                best_box = boxes[best_idx]  # [x1, y1, x2, y2]
        except Exception as e:
            print(f"[Detector] Предупреждение YOLO-World: {e}")

        # Шаг 2: Генерация масок сегментации через FastSAM
        seg_results = None
        try:
            raw_seg = self.segmenter(
                image_bgr,
                device=self.device,
                retina_masks=True,
                imgsz=1024,
                conf=max(0.10, conf * 0.5),
                iou=0.7,
                verbose=False,
            )
            if raw_seg and len(raw_seg) > 0:
                seg_results = raw_seg[0]
        except Exception as e:
            print(f"[Detector] Предупреждение FastSAM: {e}")

        # Шаг 3: Обработка масок FastSAM
        if seg_results is not None and getattr(seg_results, "masks", None) is not None:
            masks_tensor = seg_results.masks.data
            if masks_tensor is not None and len(masks_tensor) > 0:
                masks = masks_tensor.cpu().numpy() if hasattr(masks_tensor, "cpu") else np.asarray(masks_tensor)

                # Стратегия 1: Если YOLO-World нашел bbox -> сопоставляем с масками FastSAM по IoU
                if best_box is not None and getattr(seg_results, "boxes", None) is not None and len(seg_results.boxes) > 0:
                    boxes_tensor = seg_results.boxes.xyxy
                    fastsam_boxes = boxes_tensor.cpu().numpy() if hasattr(boxes_tensor, "cpu") else np.asarray(boxes_tensor)

                    x1 = np.maximum(best_box[0], fastsam_boxes[:, 0])
                    y1 = np.maximum(best_box[1], fastsam_boxes[:, 1])
                    x2 = np.minimum(best_box[2], fastsam_boxes[:, 2])
                    y2 = np.minimum(best_box[3], fastsam_boxes[:, 3])

                    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                    area1 = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
                    area2 = (fastsam_boxes[:, 2] - fastsam_boxes[:, 0]) * (fastsam_boxes[:, 3] - fastsam_boxes[:, 1])
                    union = area1 + area2 - inter
                    ious = inter / np.maximum(union, 1e-6)

                    best_mask_idx = int(np.argmax(ious))
                    if ious[best_mask_idx] > 0.15:
                        raw_mask = masks[best_mask_idx]
                        mask_resized = cv2.resize(
                            raw_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        return (mask_resized > 0).astype(np.uint8) * 255

                # Стратегия 2 (Fallback): Если YOLO-World не нашел объект или IoU мал,
                # выбираем наиболее релевантный центральный сегмент FastSAM (исключая весь фон)
                img_center = np.array([w / 2.0, h / 2.0])
                best_score = -1.0
                chosen_mask = None

                for raw_m in masks:
                    m_resized = cv2.resize(
                        raw_m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    area = float(np.sum(m_resized > 0))
                    area_ratio = area / img_area

                    # Отсекаем слишком мелкий шум (<0.5%) или весь кадр (>85%)
                    if 0.005 < area_ratio < 0.85:
                        moments = cv2.moments((m_resized > 0).astype(np.uint8))
                        if moments["m00"] > 0:
                            cx = moments["m10"] / moments["m00"]
                            cy = moments["m01"] / moments["m00"]
                            dist_to_center = np.linalg.norm(np.array([cx, cy]) - img_center)
                            norm_dist = dist_to_center / (np.linalg.norm(img_center) + 1e-6)

                            score = area_ratio * (1.0 - 0.5 * norm_dist)
                            if score > best_score:
                                best_score = score
                                chosen_mask = (m_resized > 0).astype(np.uint8) * 255

                if chosen_mask is not None:
                    return chosen_mask

        # Стратегия 3 (Fallback): Если есть только прямоугольник от YOLO-World
        if best_box is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = best_box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 255
            return mask

        return None