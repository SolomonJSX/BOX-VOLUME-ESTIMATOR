from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"

    DEFAULT_FOV_DEG: float = 60.0
    ARUCO_MARKER_SIZE_M: float = 0.05

    # Модели для детекции и точной сегментации
    YOLO_WORLD_MODEL: str = "yolov8m-worldv2.pt"
    FASTSAM_MODEL: str = "FastSAM-x.pt"  # или FastSAM-s.pt для работы на CPU
    DEPTH_MODEL_ID: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # Классы для Open-Vocabulary поиска
    BOX_TARGET_CLASSES: list[str] = [
        "box",
        "cardboard box",
        "gift box",
        "package",
        "carton",
        "parcel",
        "present",
        "container",
        "crate",
    ]


settings = Settings()