from dataclasses import dataclass
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
        """Аппроксимация матрицы камеры через FOV, если нет точной калибровки."""
        fov_rad = np.deg2rad(self.fov_deg)
        fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def build_point_cloud(self, depth_map: np.ndarray, mask: np.ndarray, scale_factor: float = 1.0) -> o3d.geometry.PointCloud:
        h, w = depth_map.shape[:2]
        fx, fy, cx, cy = self._get_intrinsics(h, w)

        v_indices, u_indices = np.where(mask > 0)
        z_vals = depth_map[v_indices, u_indices] * scale_factor

        valid = z_vals > 0

        u_indices, v_indices, z_vals = (
            u_indices[valid],
            v_indices[valid],
            z_vals[valid]
        )

        x_vals = (u_indices - cx) * z_vals / fx
        y_vals = (v_indices - cy) * z_vals / fy

        points = np.stack((x_vals, y_vals, z_vals), axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def estimate_volume(
            self,
            depth_map: np.ndarray,
            box_mask: np.ndarray,
            marker_data: dict | None = None,
            real_marker_size_m: float = 0.05
    ) -> tuple[BoxDimensions, o3d.geometry.OrientedBoundingBox]:
        h, w = depth_map.shape[:2]
        fx, fy, _, _ = self._get_intrinsics(h, w)

        scale_factor = 1.0

        if marker_data and marker_data.get("pixel_side", 0) > 0:
            pixel_side = marker_data["pixel_side"]
            estimated_depth_at_marker = (real_marker_size_m * fx) / pixel_side

            mc_x, mc_y = marker_data["center"]
            mc_x = np.clip(mc_x, 0, w - 1)
            mc_y = np.clip(mc_y, 0, h - 1)

            relative_depth_marker = depth_map[mc_y, mc_x]

            if relative_depth_marker > 1e-4:
                scale_factor = (
                    estimated_depth_at_marker / relative_depth_marker
                )

        pcd = self.build_point_cloud(
            depth_map, box_mask, scale_factor=scale_factor
        )

        pcd_filtered, _ = pcd.remove_statistical_outlier(
            nb_neighbors=25, std_ratio=1.2
        )

        try:
            obb = pcd_filtered.get_oriented_bounding_box(robust=True)
        except Exception:
            obb = pcd_filtered.get_oriented_bounding_box()

        extents_cm = np.sort(obb.extent * 100.0)[::-1]  # Длина, ширина, высота
        l_cm, w_cm, h_cm = extents_cm[0], extents_cm[1], extents_cm[2]

        # Если третья координата близка к 0 (плоское облако/видна одна грань),
        # используем пропорциональную оценку глубины по меньшей из видимых граней
        if h_cm < 0.1:
            h_cm = min(l_cm, w_cm) if min(l_cm, w_cm) > 0 else 1.0

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