import numpy as np
import torch
from PIL import Image
from transformers import pipeline
import cv2

class DepthEstimator:
    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.pipe = pipeline(task="depth-estimation", model=model_id, device=device)

    def predict_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        result = self.pipe(pil_img)
        depth_map = np.array(result["depth"], dtype=np.float32)

        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map
