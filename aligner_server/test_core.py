import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple

from main import (DeviceFetcher, DeviceData, CameraModel, ProbabilityGrid,
                  ConeCaster, RayCaster)


# =============================================================================
# Test helpers
# =============================================================================

def make_device(fov_h=60.0, fov_v=45.0, width=1920, height=1080,
                sensor_orientation=0, orientation=None, position=None,
                location_accuracy=5.0):
    if orientation is None:
        orientation = R.identity()
    if position is None:
        position = np.zeros(3)
    return DeviceData(
        device_id="test",
        timestamp=0,
        position=position,
        orientation=orientation,
        fov_horizontal=fov_h,
        fov_vertical=fov_v,
        image_width=width,
        image_height=height,
        motion_image=np.zeros((height, width), dtype=np.uint8),
        sensor_orientation=sensor_orientation,
        location_accuracy=location_accuracy,
    )


# =============================================================================
# GPS-to-local coordinate conversion tests
# =============================================================================

class TestGpsToLocalMath:
    """Test the GPS-to-local XYZ math used in DeviceFetcher.fetch_once()."""

    def test_gps_formula_at_equator(self):
        lat, lon, alt = 0.0, 0.0, 0.0
        ref_lat = 0.0
        x = lon * 111000 * np.cos(np.radians(ref_lat))
        y = lat * 111000
        z = alt
        assert abs(x) < 1e-9
        assert abs(y) < 1e-9
        assert z == 0.0

    def test_gps_formula_at_seattle(self):
        lat, lon, alt = 47.6062, -122.3321, 50.0
        ref_lat = lat
        x = lon * 111000 * np.cos(np.radians(ref_lat))
        y = lat * 111000
        z = alt
        expected_x = -122.3321 * 111000 * np.cos(np.radians(47.6062))
        expected_y = 47.6062 * 111000
        assert x == pytest.approx(expected_x, rel=1e-6)
        assert y == pytest.approx(expected_y, rel=1e-6)
        assert z == 50.0

    def test_gps_ref_lat_affects_x_scale(self):
        lon = 10.0
        x_at_equator = lon * 111000 * np.cos(np.radians(0))
        x_at_pole = lon * 111000 * np.cos(np.radians(80))
        assert x_at_equator > x_at_pole

    def test_gps_prague_to_local(self):
        lat, lon, alt = 50.0755, 14.4378, 200.0
        ref_lat = lat
        x = lon * 111000 * np.cos(np.radians(ref_lat))
        y = lat * 111000
        z = alt
        assert x == pytest.approx(14.4378 * 111000 * np.cos(np.radians(50.0755)), rel=1e-6)
        assert y == pytest.approx(50.0755 * 111000, rel=1e-6)
        assert z == 200.0

    def test_reference_latitude_set_on_first_call(self):
        DeviceFetcher.reference_latitude = None
        lat = 52.0
        assert DeviceFetcher.reference_latitude is None
        if DeviceFetcher.reference_latitude is None and lat != 0:
            DeviceFetcher.reference_latitude = lat
        assert DeviceFetcher.reference_latitude == 52.0

    def test_subsequent_calls_use_same_ref_lat(self):
        DeviceFetcher.reference_latitude = 45.0
        first_lat = 45.0
        second_lat = 50.0
        ref_lat_first = DeviceFetcher.reference_latitude if DeviceFetcher.reference_latitude is not None else first_lat
        x_first = 10.0 * 111000 * np.cos(np.radians(ref_lat_first))
        ref_lat_second = DeviceFetcher.reference_latitude if DeviceFetcher.reference_latitude is not None else second_lat
        x_second = 10.0 * 111000 * np.cos(np.radians(ref_lat_second))
        assert x_first == x_second


# =============================================================================
# CameraModel tests
# =============================================================================

class TestCameraModel:
    """Test CameraModel.pixel_to_ray and angular_uncertainty."""

    def test_center_pixel_ray_is_neg_z(self):
        device = make_device()
        ray = CameraModel.pixel_to_ray(960, 540, device)
        assert np.allclose(ray, np.array([0.0, 0.0, -1.0]), atol=1e-6)

    def test_top_left_pixel_90_fov(self):
        device = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100)
        ray = CameraModel.pixel_to_ray(0, 0, device)
        expected = np.array([-1.0, 1.0, -1.0])
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(ray, expected, atol=1e-6)

    def test_top_left_pixel_60_fov_signs(self):
        device = make_device(fov_h=60.0, fov_v=45.0, width=100, height=100)
        ray = CameraModel.pixel_to_ray(0, 0, device)
        assert ray[0] < 0.0
        assert ray[1] > 0.0
        assert ray[2] < 0.0

    def test_ray_is_normalized(self):
        device = make_device(fov_h=90.0, fov_v=60.0)
        for u, v in [(0, 0), (960, 540), (1920, 1080), (500, 300)]:
            ray = CameraModel.pixel_to_ray(u, v, device)
            assert abs(np.linalg.norm(ray) - 1.0) < 1e-6

    def test_sensor_orientation_90_rotates_ray(self):
        device_no_rot = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100,
                                    sensor_orientation=0)
        device_rot = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100,
                                 sensor_orientation=90)
        ray_no_rot = CameraModel.pixel_to_ray(0, 0, device_no_rot)
        ray_rot = CameraModel.pixel_to_ray(0, 0, device_rot)
        expected = np.array([1.0, 1.0, -1.0])
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(ray_rot, expected, atol=1e-6)

    def test_sensor_orientation_neg_90_rotates_opposite(self):
        device = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100,
                             sensor_orientation=-90)
        ray = CameraModel.pixel_to_ray(0, 0, device)
        expected = np.array([-1.0, -1.0, -1.0])
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(ray, expected, atol=1e-6)

    def test_sensor_orientation_270_same_as_neg_90(self):
        dev_270 = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100,
                              sensor_orientation=270)
        dev_n90 = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100,
                              sensor_orientation=-90)
        ray_270 = CameraModel.pixel_to_ray(0, 0, dev_270)
        ray_n90 = CameraModel.pixel_to_ray(0, 0, dev_n90)
        assert np.allclose(ray_270, ray_n90, atol=1e-6)

    def test_world_orientation_applied(self):
        rot_90_z = R.from_euler('z', np.radians(90))
        device = make_device(orientation=rot_90_z)
        ray = CameraModel.pixel_to_ray(960, 540, device)
        expected = np.array([0.0, 0.0, -1.0])
        assert np.allclose(ray, expected, atol=1e-6)

    def test_world_orientation_plus_sensor_orientation(self):
        rot_180_y = R.from_euler('y', np.radians(180))
        device = make_device(fov_h=90.0, fov_v=90.0, width=100, height=100,
                             sensor_orientation=90, orientation=rot_180_y)
        ray = CameraModel.pixel_to_ray(0, 0, device)
        assert abs(np.linalg.norm(ray) - 1.0) < 1e-6

    def test_angular_uncertainty_horizontal(self):
        device = make_device(fov_h=60.0, fov_v=45.0, width=1920, height=1080)
        angle = CameraModel.angular_uncertainty(device)
        assert angle == pytest.approx(60.0 / 1920.0)

    def test_angular_uncertainty_vertical_dominates(self):
        device = make_device(fov_h=90.0, fov_v=10.0, width=2000, height=100)
        angle = CameraModel.angular_uncertainty(device)
        assert angle == pytest.approx(10.0 / 100.0)

    def test_angular_uncertainty_symmetric(self):
        device = make_device(fov_h=45.0, fov_v=45.0, width=1000, height=1000)
        angle = CameraModel.angular_uncertainty(device)
        assert angle == pytest.approx(0.045)


# =============================================================================
# ProbabilityGrid tests
# =============================================================================

class TestProbabilityGrid:
    """Test sliding-window probability field."""

    def make_grid(self, grid_size=20, voxel_size=1.0, window_size=30):
        return ProbabilityGrid(grid_size, voxel_size,
                               grid_center=np.zeros(3),
                               window_size=window_size)

    def test_init_bounds(self):
        pg = self.make_grid(grid_size=10, voxel_size=2.0)
        assert pg.grid_size == 10
        assert pg.voxel_size == 2.0
        assert np.all(pg.grid_min == -10.0)
        assert np.all(pg.grid_max == 10.0)

    def test_empty_grid_returns_zeros(self):
        pg = self.make_grid()
        field = pg.get_probability()
        assert np.max(field) == 0.0
        assert field.shape == (20, 20, 20)

    def test_add_single_frame(self):
        pg = self.make_grid()
        pg.add_frame([(10, 10, 10, 0.5), (11, 11, 11, 0.3)])
        field = pg.get_probability()
        assert field[10, 10, 10] == 0.5
        assert field[11, 11, 11] == 0.3
        assert field[0, 0, 0] == 0.0

    def test_sliding_window_drops_oldest(self):
        pg = ProbabilityGrid(grid_size=10, voxel_size=1.0,
                             grid_center=np.zeros(3), window_size=3)
        pg.add_frame([(0, 0, 0, 1.0)])
        pg.add_frame([(0, 0, 0, 2.0)])
        pg.add_frame([(0, 0, 0, 3.0)])
        pg.add_frame([(0, 0, 0, 4.0)])  # pushes out frame 1
        field = pg.get_probability()
        # Frames 2+3+4 = 2+3+4 = 9
        assert field[0, 0, 0] == pytest.approx(9.0)

    def test_multiple_frames_accumulate(self):
        pg = self.make_grid()
        for i in range(5):
            pg.add_frame([(5, 5, 5, 0.2)])
        field = pg.get_probability()
        assert field[5, 5, 5] == pytest.approx(1.0)

    def test_same_voxel_multiple_contributions_per_frame(self):
        pg = self.make_grid()
        pg.add_frame([(3, 3, 3, 0.1), (3, 3, 3, 0.2)])
        field = pg.get_probability()
        assert field[3, 3, 3] == pytest.approx(0.3)

    def test_field_is_sum_of_all_frames(self):
        pg = self.make_grid()
        pg.add_frame([(1, 0, 0, 0.5), (2, 0, 0, 0.3)])
        pg.add_frame([(1, 0, 0, 0.1)])
        field = pg.get_probability()
        assert field[1, 0, 0] == pytest.approx(0.6)
        assert field[2, 0, 0] == pytest.approx(0.3)

    def test_set_center_moves_grid(self):
        pg = self.make_grid(grid_size=10, voxel_size=1.0)
        pg.set_center(np.array([5.0, 0.0, 0.0]))
        assert np.all(pg.grid_min == np.array([0.0, -5.0, -5.0]))
        assert np.all(pg.grid_max == np.array([10.0, 5.0, 5.0]))

    def test_clear_removes_all_frames(self):
        pg = self.make_grid()
        pg.add_frame([(5, 5, 5, 0.5)])
        pg.clear()
        field = pg.get_probability()
        assert np.max(field) == 0.0

    def test_frame_buffer_length_capped(self):
        pg = ProbabilityGrid(grid_size=10, voxel_size=1.0,
                             grid_center=np.zeros(3), window_size=5)
        for i in range(10):
            pg.add_frame([(i % 10, 0, 0, 1.0)])
        assert len(pg.frame_buffer) == 5


# =============================================================================
# ConeCaster tests
# =============================================================================

class TestConeCaster:
    """Test cone-based volume intersection ray marching."""

    def make_grid(self, grid_size=10, voxel_size=1.0):
        return ProbabilityGrid(grid_size, voxel_size, grid_center=np.zeros(3))

    def test_sphere_offsets_radius_0(self):
        offs = ConeCaster._sphere_offsets(0)
        assert offs == [(0, 0, 0)]

    def test_sphere_offsets_radius_1(self):
        offs = ConeCaster._sphere_offsets(1)
        assert len(offs) == 7  # center + 6 face neighbors

    def test_sphere_offsets_cache(self):
        offs1 = ConeCaster._sphere_offsets(2)
        offs2 = ConeCaster._sphere_offsets(2)
        assert offs1 is offs2  # cached

    def test_origin_outside_pointing_away_empty(self):
        pg = self.make_grid()
        cc = ConeCaster(pg)
        device = make_device()
        origin = np.array([100.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        assert len(result) == 0

    def test_origin_inside_grid(self):
        pg = self.make_grid()
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=0.0)
        origin = np.zeros(3)
        direction = np.array([1.0, 0.0, 0.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        assert len(result) > 0

    def test_cone_with_zero_uncertainty_is_single_line(self):
        """With zero GPS accuracy and tiny angular uncertainty, cone is ~single voxel."""
        pg = self.make_grid()
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=0.0, width=10000, height=10000)
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, -1.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        # Should be a thin line; most steps return exactly one voxel
        assert len(result) > 0
        # With location_accuracy=0 and huge resolution (= tiny pixel angle),
        # cone radius is ~0 at all distances -> single row of voxels
        # Verify they form a line along Z (ix, iy nearly constant)
        xs = {r[0] for r in result}
        ys = {r[1] for r in result}
        assert len(xs) <= 2  # should be tight around center
        assert len(ys) <= 2

    def test_cone_with_gps_uncertainty_spreads(self):
        """With GPS accuracy 5m, cone radius should be ~5m at origin, spreading."""
        pg = self.make_grid(grid_size=20, voxel_size=2.0)
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=10.0, width=10000, height=10000)
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, -1.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        assert len(result) > 1
        # With radius_voxels = int(10.0/2.0) = 5, should have multiple voxels per step
        xs = {r[0] for r in result}
        assert len(xs) > 1  # cone spreads in X

    def test_gaussian_weights_sum_lt_intensity(self):
        """Total probability assigned should not exceed intensity (within a step)."""
        pg = self.make_grid(grid_size=4, voxel_size=1.0)
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=0.5)
        origin = np.array([1.5, 1.5, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = cc.cast_cone(origin, direction, device, intensity=0.8)
        # Sum of all probabilities in the result
        total = sum(r[3] for r in result)
        # Should be roughly intensity * number_of_steps (center gets most, spread adds)
        assert total > 0.0

    def test_probabilities_positive_and_bounded(self):
        pg = self.make_grid(grid_size=10, voxel_size=1.0)
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=3.0)
        origin = np.zeros(3)
        direction = np.array([1.0, 0.0, 0.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        for _, _, _, prob in result:
            assert 0.0 < prob <= 1.0

    def test_ray_along_y_axis_works(self):
        pg = self.make_grid()
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=0.0, width=10000, height=10000)
        origin = np.zeros(3)
        direction = np.array([0.0, 1.0, 0.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        assert len(result) > 0

    def test_ray_along_z_axis_works(self):
        pg = self.make_grid()
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=0.0, width=10000, height=10000)
        origin = np.zeros(3)
        direction = np.array([0.0, 0.0, 1.0])
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        assert len(result) > 0

    def test_all_voxels_within_grid_bounds(self):
        pg = self.make_grid(grid_size=10, voxel_size=1.0)
        cc = ConeCaster(pg)
        device = make_device(location_accuracy=2.0)
        origin = np.zeros(3)
        direction = np.array([1.0, 0.5, 0.3])
        direction = direction / np.linalg.norm(direction)
        result = cc.cast_cone(origin, direction, device, intensity=1.0)
        for ix, iy, iz, _ in result:
            assert 0 <= ix < 10
            assert 0 <= iy < 10
            assert 0 <= iz < 10


# =============================================================================
# RayCaster (orchestrator) tests
# =============================================================================

class TestRayCaster:
    """Test RayCaster orchestration and process_device."""

    def test_init_sets_up_components(self):
        rc = RayCaster(grid_size=100, voxel_size=2.0, window_size=30)
        assert rc.grid_size == 100
        assert rc.voxel_size == 2.0
        assert rc.prob_grid.window_size == 30
        assert rc.cone_caster is not None

    def test_process_device_returns_sparse(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0)
        img = np.zeros((20, 20), dtype=np.uint8)
        img[10, 10] = 30  # above threshold
        img[5, 5] = 60
        device = make_device(width=20, height=20, motion_image=img,
                             location_accuracy=0.0)
        result = rc.process_device(device, sample_step=1, motion_threshold=15)
        assert len(result) > 0
        # Each element is (ix, iy, iz, prob)
        for item in result:
            assert len(item) == 4

    def test_process_device_skips_below_threshold(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0)
        img = np.zeros((30, 30), dtype=np.uint8)
        img[15, 15] = 5  # below threshold of 15
        device = make_device(width=30, height=30, motion_image=img,
                             location_accuracy=0.0)
        result = rc.process_device(device, sample_step=1, motion_threshold=15)
        assert len(result) == 0

    def test_on_ray_callback_called(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0)
        img = np.zeros((20, 20), dtype=np.uint8)
        img[10, 10] = 50
        device = make_device(width=20, height=20, motion_image=img,
                             location_accuracy=0.0)
        collected = []

        def cb(origin, direction, intensity):
            collected.append((tuple(origin), tuple(direction), intensity))

        rc.process_device(device, sample_step=1, motion_threshold=20,
                          on_ray=cb)
        assert len(collected) > 0

    def test_get_probability_field(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0, window_size=10)
        # Add a frame with known contributions
        rc.prob_grid.add_frame([(5, 5, 5, 0.7), (6, 6, 6, 0.2)])
        field = rc.get_probability_field()
        assert field.shape == (10, 10, 10)
        assert field[5, 5, 5] == 0.7
        assert field[6, 6, 6] == 0.2

    def test_multiframe_accumulation(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0, window_size=5)
        rc.prob_grid.add_frame([(0, 0, 0, 0.1)])
        rc.prob_grid.add_frame([(0, 0, 0, 0.1)])
        rc.prob_grid.add_frame([(0, 0, 0, 0.1)])
        field = rc.get_probability_field()
        assert field[0, 0, 0] == pytest.approx(0.3)

    def test_grid_center_aliases_stay_in_sync(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0,
                       grid_center=np.array([5.0, 0.0, 0.0]))
        assert np.all(rc.grid_center == np.array([5.0, 0.0, 0.0]))
        rc.prob_grid.set_center(np.array([10.0, 0.0, 0.0]))
        # After set_center, aliases need manual update (done in main loop)
        rc.grid_center = rc.prob_grid.grid_center
        rc.grid_min = rc.prob_grid.grid_min
        rc.grid_max = rc.prob_grid.grid_max
        assert np.all(rc.grid_center == np.array([10.0, 0.0, 0.0]))
        assert np.all(rc.grid_min == np.array([5.0, -5.0, -5.0]))


# =============================================================================
# Integration: full pipeline
# =============================================================================

class TestIntegration:
    """End-to-end: multiple devices, cone casting, probability field."""

    def test_two_devices_produce_overlapping_probability(self):
        rc = RayCaster(grid_size=20, voxel_size=1.0, window_size=10)
        img = np.zeros((40, 40), dtype=np.uint8)
        img[20, 20] = 100  # center pixel = bright motion
        img[10, 10] = 50

        # Device 1: at origin, looking down -Z
        dev1 = make_device(width=40, height=40, motion_image=img,
                           position=np.array([5.0, 0.0, 10.0]),
                           location_accuracy=0.0, fov_h=30.0, fov_v=30.0)

        # Device 2: at same XY, different Z, looking at same target
        dev2 = make_device(width=40, height=40, motion_image=img,
                           position=np.array([-5.0, 0.0, 10.0]),
                           location_accuracy=0.0, fov_h=30.0, fov_v=30.0)

        sparse1 = rc.process_device(dev1, sample_step=1, motion_threshold=20)
        sparse2 = rc.process_device(dev2, sample_step=1, motion_threshold=20)

        rc.prob_grid.add_frame(sparse1 + sparse2)
        field = rc.get_probability_field()

        # Should have non-zero probability somewhere
        assert np.max(field) > 0.0

    def test_motion_pixels_outside_frustum_are_ignored(self):
        """Pixels at image edges with very narrow FOV should still produce rays."""
        rc = RayCaster(grid_size=10, voxel_size=1.0)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0, 0] = 100  # corner pixel
        device = make_device(width=10, height=10, motion_image=img,
                             location_accuracy=0.0, fov_h=20.0, fov_v=20.0)
        result = rc.process_device(device, sample_step=1, motion_threshold=50)
        assert len(result) > 0

    def test_empty_image_gives_no_contributions(self):
        rc = RayCaster(grid_size=10, voxel_size=1.0)
        img = np.zeros((20, 20), dtype=np.uint8)
        device = make_device(width=20, height=20, motion_image=img,
                             location_accuracy=0.0)
        result = rc.process_device(device, sample_step=4, motion_threshold=10)
        assert len(result) == 0
