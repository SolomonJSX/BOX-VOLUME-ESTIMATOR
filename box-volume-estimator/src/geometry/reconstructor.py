from dataclasses import dataclass
import cv2
import numpy as np
import open3d as o3d


@dataclass
class BoxDimensions:
    width_cm: float
    height_cm: float
    length_cm: float
    volume_liters: float
    volume_cm3: float


class Box3DReconstructor:

    def __init__(self, fov_deg: float = 60.0):
        self.fov_deg = fov_deg

    def _get_intrinsics(self, height: int, width: int):
        fov_rad = np.deg2rad(self.fov_deg)
        fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def build_point_cloud(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        scale_factor: float = 1.0,
    ) -> o3d.geometry.PointCloud:
        h, w = depth_map.shape[:2]
        fx, fy, cx, cy = self._get_intrinsics(h, w)

        v_indices, u_indices = np.where(mask > 0)
        z_vals = depth_map[v_indices, u_indices] * scale_factor

        valid = z_vals > 0
        u_indices, v_indices, z_vals = (
            u_indices[valid],
            v_indices[valid],
            z_vals[valid],
        )

        x_vals = (u_indices - cx) * z_vals / fx
        y_vals = (v_indices - cy) * z_vals / fy

        points = np.stack((x_vals, y_vals, z_vals), axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def _fit_table_plane(
        self,
        depth_map: np.ndarray,
        box_mask: np.ndarray,
        scale_factor: float,
        dilation_kernel_size: int = 45,
    ) -> tuple[np.ndarray | None, o3d.geometry.PointCloud | None]:
        """Находит плоскость стола в окрестности коробки через RANSAC."""
        # 1. Формируем маску окружения коробки (кольцо вокруг объекта)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )
        dilated_mask = cv2.dilate(box_mask, kernel, iterations=1)
        surrounding_mask = cv2.subtract(dilated_mask, box_mask)

        # 2. Облако точек окружающего стола
        table_pcd = self.build_point_cloud(
            depth_map, surrounding_mask, scale_factor=scale_factor
        )

        if len(table_pcd.points) < 50:
            return None, None

        # 3. RANSAC для нахождения основной опорной плоскости
        plane_model, inliers = table_pcd.segment_plane(
            distance_threshold=0.01,  # 1 см погрешность
            ransac_n=3,
            num_iterations=1000,
        )

        return np.array(plane_model), table_pcd.select_by_index(inliers)

    def _complete_bottom_plane(
        self,
        box_points: np.ndarray,
        plane_model: np.ndarray,
    ) -> np.ndarray:
        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        norm_len = np.linalg.norm(normal)

        if norm_len < 1e-6:
            return box_points

        normal /= norm_len
        d /= norm_len

        distances = np.dot(box_points, normal) + d

        # Оставляем точки коробки
        above_table = np.abs(distances) > 0.001
        valid_box_pts = (
            box_points[above_table] if np.any(above_table) else box_points
        )

        # Проекция точек на плоскость стола
        dist_valid = (np.dot(valid_box_pts, normal) + d)[:, np.newaxis]
        projected_base = valid_box_pts - dist_valid * normal

        return np.vstack((valid_box_pts, projected_base))

    def estimate_volume(
        self,
        depth_map: np.ndarray,
        box_mask: np.ndarray,
        marker_data: dict | None = None,
        real_marker_size_m: float = 0.05,
    ) -> tuple[BoxDimensions, o3d.geometry.OrientedBoundingBox]:
        h, w = depth_map.shape[:2]
        fx, _, _, _ = self._get_intrinsics(h, w)

        # Вычисление scale factor по ArUco маркеру
        scale_factor = 1.0
        if marker_data and marker_data.get("pixel_side", 0) > 0:
            pixel_side = marker_data["pixel_side"]
            estimated_depth_marker = (real_marker_size_m * fx) / pixel_side

            mc_x, mc_y = marker_data["center"]
            mc_x = np.clip(mc_x, 0, w - 1)
            mc_y = np.clip(mc_y, 0, h - 1)
            relative_depth = depth_map[mc_y, mc_x]

            if relative_depth > 1e-4:
                scale_factor = estimated_depth_marker / relative_depth

        # 1. Строим облако точек коробки
        box_pcd = self.build_point_cloud(
            depth_map, box_mask, scale_factor=scale_factor
        )
        box_pcd, _ = box_pcd.remove_statistical_outlier(
            nb_neighbors=25, std_ratio=1.2
        )
        box_pts = np.asarray(box_pcd.points)

        # 2. RANSAC поиск плоскости стола
        plane_model, _ = self._fit_table_plane(
            depth_map, box_mask, scale_factor=scale_factor
        )

        # 3. Если плоскость найдена, достраиваем дно коробки
        if plane_model is not None:
            completed_pts = self._complete_bottom_plane(box_pts, plane_model)
        else:
            completed_pts = box_pts

        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(completed_pts)

        # 4. Построение минимального OBB
        obb = final_pcd.get_oriented_bounding_box()

        # Размеры в см (сортировка: длина >= ширина >= высота)
        extents_cm = np.sort(obb.extent * 100.0)[::-1]
        l_cm, w_cm, h_cm = (
            float(extents_cm[0]),
            float(extents_cm[1]),
            float(extents_cm[2]),
        )

        # Защита от вырождения в 2D-плоскость при плоском срезе глубины
        if h_cm < 0.1:
            # Если высота нулевая, принимаем глубину равной минимальной грани
            h_cm = max(round(w_cm, 2), 1.0)

        volume_cm3 = float(l_cm * w_cm * h_cm)
        volume_liters = volume_cm3 / 1000.0

        dimensions = BoxDimensions(
            width_cm=round(float(w_cm), 2),
            height_cm=round(float(h_cm), 2),
            length_cm=round(float(l_cm), 2),
            volume_liters=round(volume_liters, 2),
            volume_cm3=round(volume_cm3, 2),
        )

        return dimensions, obb