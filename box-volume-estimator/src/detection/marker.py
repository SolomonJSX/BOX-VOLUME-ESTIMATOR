import cv2
import numpy as np

class ArUcoDetector:
    def __init__(self, marker_size_m: float = 0.05, dict_type: int = cv2.aruco.DICT_5X5_100):
        self.marker_size_m = marker_size_m
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)


    def detect(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(corners) == 0:
            return None

        c = corners[0][0]

        # Берем первый найденный маркер
        c = corners[0][0]  # shape: (4, 2)
        # Периметр маркера в пикселях
        pixel_side = (
            np.linalg.norm(c[0] - c[1])
            + np.linalg.norm(c[1] - c[2])
            + np.linalg.norm(c[2] - c[3])
            + np.linalg.norm(c[3] - c[0])
        ) / 4.0

        return {
            "corners": c,
            "id": int(ids[0][0]),
            "pixel_side": float(pixel_side),
            "center": tuple(c.mean(axis=0).astype(int)),
        }
