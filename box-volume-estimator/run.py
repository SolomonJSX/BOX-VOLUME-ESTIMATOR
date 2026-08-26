from pathlib import Path
import cv2
import open3d as o3d
from rich.console import Console
from rich.table import Table
import typer

from src.config import settings
from src.depth.depth_estimator import DepthEstimator
from src.detection.detector import PreciseBoxDetector
from src.detection.marker import ArUcoDetector
from src.geometry.reconstructor import Box3DReconstructor

app = typer.Typer(help="Оценка объема коробки по фотографии")
console = Console()

@app.command()
def estimate(image_path: Path = typer.Argument(..., help="Путь к изображению коробки с подарком"), visualize_3d: bool = typer.Option(False, "--viz", help="Открыть 3D-окно Open3D")):
    if not image_path.exists():
        console.print(
            f"[bold red]Ошибка:[/bold red] Файл {image_path} не найден."
        )
        raise typer.Exit(1)

    image = cv2.imread(str(image_path))

    if image is None:
        console.print("[bold red]Ошибка:[/bold red] Не удалось прочитать фото.")
        raise typer.Exit(1)

    with console.status("[bold green]Обработка изображения...[/bold green]"):
        # 1. Поиск калибровочного маркера
        marker_detector = ArUcoDetector(
            marker_size_m=settings.ARUCO_MARKER_SIZE_M
        )
        marker_data = marker_detector.detect(image)

        # 2. Сегментация коробки
        box_detector = PreciseBoxDetector(
            yolo_world_model=settings.YOLO_WORLD_MODEL,
            fastsam_model=settings.FASTSAM_MODEL,
            target_classes=settings.BOX_TARGET_CLASSES,
        )
        mask = box_detector.extract_box_mask(image)
        
        if mask is None:
            console.print(
                "[bold red]Ошибка:[/bold red] Коробка не обнаружена на фото."
            )
            raise typer.Exit(1)

        depth_estimator = DepthEstimator(model_id=settings.DEPTH_MODEL_ID)
        depth_map = depth_estimator.predict_depth(image)

        reconstructor = Box3DReconstructor(fov_deg=settings.DEFAULT_FOV_DEG)
        dimensions, obb = reconstructor.estimate_volume(
            depth_map=depth_map,
            box_mask=mask,
            marker_data=marker_data,
            real_marker_size_m=settings.ARUCO_MARKER_SIZE_M,
        )

        table = Table(
        title="🎁 Результаты оценки подарка",
        show_header=True,
        header_style="bold magenta",
        )
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")

        table.add_row(
            "Маркер масштаба",
            "Найден (ArUco)" if marker_data else "Не найден (относительный масштаб)",
        )
        table.add_row(
            "Габариты (Д × Ш × В)",
            f"{dimensions.length_cm} × {dimensions.width_cm} × {dimensions.height_cm} см",
        )
        table.add_row("Объем (см³)", f"{dimensions.volume_cm3:,.2f} см³")
        table.add_row(
            "Объем (литры)", f"[bold yellow]{dimensions.volume_liters} л[/bold yellow]"
        )

        console.print(table)

        # 3D визуализация
        if visualize_3d:
            pcd = reconstructor.build_point_cloud(depth_map, mask)
            obb.color = (1, 0, 0)
            o3d.visualization.draw_geometries(
                [pcd, obb], window_name="3D Оценка Коробки"
            )

        
if __name__ == "__main__":
    app()