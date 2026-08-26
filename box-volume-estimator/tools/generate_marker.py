from pathlib import Path
import cv2
import numpy as np
import typer

app = typer.Typer(help="Генератор ArUco-маркеров для калибровки масштаба")


@app.command()
def generate(
    marker_id: int = typer.Option(
        0, "--id", "-i", help="ID маркера в словаре"
    ),
    marker_size_px: int = typer.Option(
        500, "--size", "-s", help="Размер самого маркера в пикселях"
    ),
    target_size_cm: float = typer.Option(
        5.0, "--cm", "-c", help="Физический размер маркера для печати (см)"
    ),
    output_path: Path = typer.Option(
        Path("data/aruco_marker.png"),
        "--out",
        "-o",
        help="Путь для сохранения PNG",
    ),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Генерация матрицы маркера
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict, marker_id, marker_size_px
    )

    # 2. Создание холста с полями (белая рамка для контраста)
    margin = 80
    canvas_w = marker_size_px + 2 * margin
    canvas_h = marker_size_px + 2 * margin + 90  # Дополнительное место под текст

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    # Вставка маркера по центру
    canvas[margin : margin + marker_size_px, margin : margin + marker_size_px] = (
        marker_img
    )

    # 3. Текстовая аннотация и инструкции
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_line1 = f"ArUco DICT_5X5_100 (ID: {marker_id})"
    text_line2 = f"Target size: {target_size_cm:.1f} x {target_size_cm:.1f} cm"
    text_line3 = "Place marker on the same plane as the box base"

    y_start = marker_size_px + margin + 30
    cv2.putText(
        canvas,
        text_line1,
        (margin, y_start),
        font,
        0.55,
        0,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text_line2,
        (margin, y_start + 25),
        font,
        0.55,
        0,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text_line3,
        (margin, y_start + 50),
        font,
        0.45,
        80,
        1,
        cv2.LINE_AA,
    )

    # 4. Тонкая направляющая рамка по внешнему контуру
    cv2.rectangle(
        canvas,
        (margin - 1, margin - 1),
        (margin + marker_size_px, margin + marker_size_px),
        180,
        1,
    )

    # Сохранение файла
    cv2.imwrite(str(output_path), canvas)
    typer.secho(
        f"Маркер успешно сохранен: {output_path.resolve()}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.echo(
        f"Печатайте изображение так, чтобы черный квадрат был ровно {target_size_cm}x{target_size_cm} см."
    )


if __name__ == "__main__":
    app()