import numpy as np
import open3d as o3d
import pytest

from src.geometry.reconstructor import Box3DReconstructor


class TestBox3DReconstructor:

    @pytest.fixture
    def reconstructor(self) -> Box3DReconstructor:
        return Box3DReconstructor(fov_deg=60.0)

    def test_intrinsics_calculation(self, reconstructor: Box3DReconstructor):
        h, w = 480, 640
        fx, fy, cx, cy = reconstructor._get_intrinsics(h, w)

        # При FOV 60°: fx = (w/2) / tan(30°) ≈ 320 / 0.57735 ≈ 554.256
        expected_fx = (w / 2.0) / np.tan(np.deg2rad(30.0))
        assert np.isclose(fx, expected_fx, rtol=1e-3)
        assert np.isclose(fy, expected_fx, rtol=1e-3)
        assert cx == 320.0
        assert cy == 240.0

    def test_build_point_cloud_structure(
        self, reconstructor: Box3DReconstructor
    ):
        h, w = 100, 100
        depth_map = np.ones((h, w), dtype=np.float32) * 2.0  # глубина 2 метра
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Квадрат 20x20 пикселей

        pcd = reconstructor.build_point_cloud(
            depth_map, mask, scale_factor=1.0
        )

        assert isinstance(pcd, o3d.geometry.PointCloud)
        pts = np.asarray(pcd.points)
        assert pts.shape[0] == 400  # 20 * 20 точек
        # Все точки должны лежать на Z = 2.0
        assert np.allclose(pts[:, 2], 2.0)

    def test_synthetic_cube_volume_estimation(
        self, reconstructor: Box3DReconstructor
    ):
        """Тест расчета объема синтетической 3D коробки, стоящей на столе."""
        h, w = 480, 640
        # Стол на расстоянии 1.0м
        depth_map = np.full((h, w), 1.0, dtype=np.float32)

        # Коробка выступает вперед на 0.8м (высота коробки ~ 20 см)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[140:340, 220:420] = 255
        depth_map[140:340, 220:420] = 0.8

        # Синтетический маркер на плоскости стола
        marker_data = {
            "pixel_side": 50.0,
            "center": (100, 100),
            "id": 0,
        }

        dimensions, obb = reconstructor.estimate_volume(
            depth_map=depth_map,
            box_mask=mask,
            marker_data=marker_data,
            real_marker_size_m=0.05,
        )

        assert dimensions.volume_cm3 > 0.0
        assert dimensions.volume_liters > 0.0
        assert dimensions.width_cm > 0.0
        assert dimensions.height_cm > 0.0
        assert dimensions.length_cm > 0.0

    def test_reconstruction_mesh_and_obb_edges(
        self, reconstructor: Box3DReconstructor
    ):
        obb = o3d.geometry.OrientedBoundingBox(
            center=[0.0, 0.0, 1.0], R=np.eye(3), extent=[0.2, 0.3, 0.4]
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(50, 3))

        box_mesh = reconstructor.create_box_mesh(obb)
        assert isinstance(box_mesh, o3d.geometry.TriangleMesh)
        assert len(box_mesh.vertices) == 8
        assert len(box_mesh.triangles) == 12

        edge_mesh = reconstructor.create_obb_edge_mesh(obb)
        assert isinstance(edge_mesh, o3d.geometry.TriangleMesh)
        assert len(edge_mesh.triangles) > 0

        full_mesh = reconstructor.create_reconstruction_mesh(pcd, obb)
        assert isinstance(full_mesh, o3d.geometry.TriangleMesh)
        assert len(full_mesh.triangles) == len(box_mesh.triangles) + len(edge_mesh.triangles)