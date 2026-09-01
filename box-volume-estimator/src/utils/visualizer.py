from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

def save_debug_artifacts(
    image_bgr: np.ndarray,
    box_mask: np.ndarray,
    depth_map: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    obb: o3d.geometry.OrientedBoundingBox,
    marker_data: dict | None,
    output_dir: Path,
    mesh: o3d.geometry.TriangleMesh | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # 1. Отрисовка контуров коробки и маркера на фото
    overlay = image_bgr.copy()

    contours, _ = cv2.findContours(
        box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    if marker_data and "corners" in marker_data:
        corners = np.int32(marker_data["corners"].reshape((-1, 1, 2)))
        cv2.polylines(overlay, [corners], True, (0, 0, 255), 2)
        center = marker_data["center"]
        cv2.putText(
            overlay,
            f"ArUco ID: {marker_data['id']}",
            (center[0] - 20, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    mask_path = output_dir / "mask_debug.png"
    cv2.imwrite(str(mask_path), overlay)
    saved_paths["mask"] = mask_path

    # 2. Раскрашенная карта глубин (Colormap Inferno)
    depth_norm = cv2.normalize(
        depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    depth_path = output_dir / "depth_colored.png"
    cv2.imwrite(str(depth_path), depth_colored)
    saved_paths["depth"] = depth_path

    # 3. Сохранение 3D-модели (Mesh / PointCloud)
    ply_path = output_dir / "box_model.ply"
    obj_path = output_dir / "box_model.obj"
    glb_path = output_dir / "box_model.glb"

    if mesh is not None and len(mesh.triangles) > 0:
        o3d.io.write_triangle_mesh(str(ply_path), mesh)
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        saved_paths["ply"] = ply_path
        saved_paths["obj"] = obj_path

        # Экспорт в GLB для идеального рендеринга в Three.js / Gradio
        try:
            import trimesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            vertex_colors = None
            if len(mesh.vertex_colors) > 0:
                vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

            tm = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_colors=vertex_colors
            )
            glb_data = tm.export(file_type="glb")
            with open(glb_path, "wb") as f:
                f.write(glb_data)
            saved_paths["glb"] = glb_path
        except Exception as e:
            print(f"[Visualizer] Предупреждение при экспорте GLB: {e}")
    else:
        o3d.io.write_point_cloud(str(ply_path), pcd)
        saved_paths["ply"] = ply_path

    return saved_paths


