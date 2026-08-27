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
from src.utils.visualizer import save_debug_artifacts

app = typer.Typer(help="Оценка объема коробки по фотографии")
console = Console()


@app.command()
def estimate(
    image_path: Path = typer.Argument(
        ..., help="Путь к изображению коробки с подарком"
    ),
    visualize_3d: bool = typer.Option(
        False, "--viz", help="Открыть интерактивное 3D-окно Open3D"
    ),
    save_artifacts: bool = typer.Option(
        True, "--save/--no-save", help="Сохранить артефакты в data/processed"
    ),
):
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

        # 2. Детекция и точная сегментация
        box_detector = PreciseBoxDetector()
        mask = box_detector.extract_box_mask(image)
        if mask is None:
            console.print(
                "[bold red]Ошибка:[/bold red] Коробка не обнаружена на фото."
            )
            raise typer.Exit(1)

        # 3. Оценка карты глубин
        depth_estimator = DepthEstimator(model_id=settings.DEPTH_MODEL_ID)
        depth_map = depth_estimator.predict_depth(image)

        # 4. 3D-реконструкция и расчет объема с RANSAC
        reconstructor = Box3DReconstructor(fov_deg=settings.DEFAULT_FOV_DEG)
        dimensions, obb = reconstructor.estimate_volume(
            depth_map=depth_map,
            box_mask=mask,
            marker_data=marker_data,
            real_marker_size_m=settings.ARUCO_MARKER_SIZE_M,
        )

        pcd = reconstructor.build_point_cloud(depth_map, mask)

        # 5. Сохранение результатов
        saved_paths = {}
        if save_artifacts:
            saved_paths = save_debug_artifacts(
                image_bgr=image,
                box_mask=mask,
                depth_map=depth_map,
                pcd=pcd,
                obb=obb,
                marker_data=marker_data,
                output_dir=settings.DATA_DIR / "processed",
            )

    # Вывод результатов
    table = Table(
        title="🎁 Результаты оценки подарка",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="green")

    table.add_row(
        "Маркер масштаба",
        f"[green]Найден (ID: {marker_data['id']})[/green]"
        if marker_data
        else "[yellow]Не найден (относительный масштаб)[/yellow]",
    )
    table.add_row(
        "Габариты (Д × Ш × В)",
        f"{dimensions.length_cm} × {dimensions.width_cm} × {dimensions.height_cm} см",
    )
    table.add_row("Объем (см³)", f"{dimensions.volume_cm3:,.2f} см³")
    table.add_row(
        "Объем (литры)", f"[bold yellow]{dimensions.volume_liters} л[/bold yellow]"
    )

    if saved_paths:
        table.add_row(
            "Артефакты", f"Сохранены в {settings.DATA_DIR / 'processed'}"
        )

    console.print(table)

    if visualize_3d:
        obb.color = (1, 0, 0)
        o3d.visualization.draw_geometries(
            [pcd, obb], window_name="3D Оценка Коробки"
        )


if __name__ == "__main__":
    app()