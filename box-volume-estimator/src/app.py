from pathlib import Path
import sys

# Ensure project root is in sys.path when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import gradio as gr
import numpy as np
import open3d as o3d

from src.config import settings
from src.depth.depth_estimator import DepthEstimator
from src.detection.detector import PreciseBoxDetector
from src.detection.marker import ArUcoDetector
from src.geometry.reconstructor import Box3DReconstructor
from src.utils.visualizer import save_debug_artifacts

print("Загрузка нейросетевых моделей...")
box_detector = PreciseBoxDetector()
depth_estimator = DepthEstimator(model_id=settings.DEPTH_MODEL_ID)
marker_detector = ArUcoDetector(marker_size_m=settings.ARUCO_MARKER_SIZE_M)
reconstructor = Box3DReconstructor(fov_deg=settings.DEFAULT_FOV_DEG)

def process_box_image(
    image: np.ndarray | None,
    marker_size_cm: float,
    fov_deg: float,
    conf_threshold: float = 0.20,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        gr.Warning("Пожалуйста, загрузите или сделайте снимок коробки перед расчетом.")
        return None, None, None, "⚠️ **Загрузите изображение коробки для оценки объема.**"

    try:
        progress(0.1, desc="🔍 Поиск маркера ArUco...")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        reconstructor.fov_deg = fov_deg
        real_marker_size_m = marker_size_cm / 100.0

        marker_detector.marker_size_m = real_marker_size_m
        marker_data = marker_detector.detect(image_bgr)

        progress(0.3, desc="📦 Сегментация коробки (YOLO-World + FastSAM)...")
        mask = box_detector.extract_box_mask(image_bgr, conf=conf_threshold)

        if mask is None:
            gr.Warning("Коробка не обнаружена. Попробуйте снизить порог уверенности (Confidence) или изменить ракурс.")
            return None, None, None, "⚠️ **Коробка не обнаружена.** Попробуйте снизить порог детекции в параметрах или сделать фото с лучшим освещением."

        progress(0.6, desc="🌐 Оценка карты глубины (Depth Anything v2)...")
        depth_map = depth_estimator.predict_depth(image_bgr)

        progress(0.8, desc="📐 3D реконструкция и вычисление объема...")
        dimensions, obb = reconstructor.estimate_volume(
            depth_map=depth_map,
            box_mask=mask,
            marker_data=marker_data,
            real_marker_size_m=real_marker_size_m,
        )

        pcd = reconstructor.build_point_cloud(
            depth_map, mask, image_rgb=image
        )

        progress(0.90, desc="📐 Создание полигональной 3D-сетки и ребер OBB...")
        mesh = reconstructor.create_reconstruction_mesh(
            pcd=pcd,
            obb=obb,
            box_color=[0.28, 0.65, 0.90],
            edge_color=[0.95, 0.15, 0.15],
        )

        progress(0.95, desc="💾 Сохранение 3D-модели и отчета...")
        output_dir = settings.DATA_DIR / "processed"
        artifacts = save_debug_artifacts(
            image_bgr=image_bgr,
            box_mask=mask,
            depth_map=depth_map,
            pcd=pcd,
            obb=obb,
            marker_data=marker_data,
            output_dir=output_dir,
            mesh=mesh,
        )

        marker_status = (
            f"Найден (ID {marker_data['id']})"
            if marker_data
            else "Не найден (масштаб приближенный по FOV)"
        )

        markdown_report = f"""
### 🎁 Результаты измерений:
* **Объем:** **`{dimensions.volume_liters:.2f} л`** ({dimensions.volume_cm3:,.1f} см³)
* **Габариты (Д × Ш × В):** `{dimensions.length_cm} × {dimensions.width_cm} × {dimensions.height_cm} см`
* **Калибровочный маркер:** `{marker_status}`
        """

        mask_view = cv2.cvtColor(
            cv2.imread(str(artifacts["mask"])), cv2.COLOR_BGR2RGB
        )
        depth_view = cv2.cvtColor(
            cv2.imread(str(artifacts["depth"])), cv2.COLOR_BGR2RGB
        )

        gr.Info("Расчет успешно завершен!")
        model_3d_path = str(artifacts.get("glb", artifacts.get("obj", artifacts["ply"])))
        return (
            model_3d_path,  # Путь к 3D-модели (GLB/OBJ/PLY) для встроенного вьювера Gradio
            mask_view,  # Исходное фото с контуром маски
            depth_view,  # Цветовая карта глубины
            markdown_report,  # Markdown сводка
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Error(f"Ошибка во время расчета: {str(e)}")
        return None, None, None, f"❌ **Ошибка:** `{str(e)}`"

with gr.Blocks(title="Box Volume Estimator") as demo:
    gr.Markdown(
        """
    # 🎁 Оценка объема коробки по фотографии
    Загрузите фото коробки с подарком или сделайте снимок с камеры. 
    Для точного метрического масштаба положите рядом ArUco-маркер (по умолчанию 5 см).
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Фотография коробки",
                sources=["upload", "webcam", "clipboard"],
                type="numpy",
            )

            with gr.Accordion("Параметры съемки и детекции", open=False):
                marker_size = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    value=5.0,
                    step=0.5,
                    label="Физический размер маркера (см)",
                )
                camera_fov = gr.Slider(
                    minimum=30.0,
                    maximum=120.0,
                    value=60.0,
                    step=1.0,
                    label="Угол обзора камеры (FOV, град)",
                )
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.90,
                    value=0.20,
                    step=0.05,
                    label="Порог уверенности детекции (Confidence)",
                )

            submit_btn = gr.Button("🚀 Рассчитать объем", variant="primary", size="lg")

        with gr.Column(scale=1):
            report_box = gr.Markdown(value="Ожидание изображения...")
            model_3d = gr.Model3D(label="Интерактивная 3D-модель (Mesh + OBB)")

    with gr.Row():
        with gr.Column():
            mask_output = gr.Image(label="Маска детекции коробки")
        with gr.Column():
            depth_output = gr.Image(label="Карта глубины (Depth Anything v2)")

    submit_btn.click(
        fn=process_box_image,
        inputs=[input_image, marker_size, camera_fov, conf_slider],
        outputs=[model_3d, mask_output, depth_output, report_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), share=False)