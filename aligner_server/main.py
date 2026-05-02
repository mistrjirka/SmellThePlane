#!/usr/bin/env python3
"""
3D Ray Intersection Visualizer

Connects to multiple phone devices, fetches motion detection data and odometry,
visualizes devices in 3D space, and performs ray intersection to detect objects.

Dependencies: numpy, scipy, requests, OpenCV, and VisPy (with a supported GUI
backend such as PyQt6).
"""

import pyray as pr
import requests
import struct
import json
import numpy as np
import cv2
import time
import sys
import threading
from dataclasses import dataclass
from collections import deque
from typing import List, Dict, Optional, Tuple, Callable
from scipy.spatial.transform import Rotation

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DeviceData:
    device_id: str
    timestamp: int
    position: np.ndarray  # [lat, lon, alt] -> converted to local XYZ
    orientation: Rotation
    fov_horizontal: float
    fov_vertical: float
    image_width: int
    image_height: int
    motion_image: np.ndarray
    sensor_orientation: int = 0  # Camera sensor mounting rotation (degrees), typically 90 or 270
    location_accuracy: float = 0.0  # GPS accuracy (meters)

# =============================================================================
# Camera Model
# =============================================================================

class CameraModel:
    """Camera geometry: pixel-to-ray projection and angular uncertainty."""
    
    @staticmethod
    def pixel_to_ray(u: int, v: int, device: DeviceData) -> np.ndarray:
        """Convert pixel coordinates to a unit ray direction in world space."""
        cx = device.image_width / 2.0
        cy = device.image_height / 2.0
        fx = cx / np.tan(np.radians(device.fov_horizontal / 2))
        fy = cy / np.tan(np.radians(device.fov_vertical / 2))
        
        x = (u - cx) / fx
        y = -(v - cy) / fy  # Flip Y
        z = -1.0
        
        ray_cam = np.array([x, y, z])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Compensate for camera sensor physical mounting offset
        if device.sensor_orientation != 0:
            sensor_rot = Rotation.from_euler('z', -np.radians(device.sensor_orientation))
            ray_cam = sensor_rot.apply(ray_cam)
        
        # Transform to world space
        return device.orientation.apply(ray_cam)
    
    @staticmethod
    def angular_uncertainty(device: DeviceData) -> float:
        """Angular width of one pixel in degrees (used for ray cone spread)."""
        pixel_angle_h = device.fov_horizontal / device.image_width
        pixel_angle_v = device.fov_vertical / device.image_height
        return max(pixel_angle_h, pixel_angle_v)

# =============================================================================
# Device Fetcher
# =============================================================================

class DeviceFetcher:
    """Fetches data from a phone device."""
    
    reference_latitude: Optional[float] = None
    
    def __init__(self, ip: str, port: int = 8080):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"
        self.last_data: Optional[DeviceData] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def fetch_once(self) -> Optional[DeviceData]:
        """Fetch data from the device once."""
        try:
            response = requests.get(f"{self.base_url}/data", timeout=2)
            if response.status_code == 204:
                return None
            if response.status_code != 200:
                print(f"[{self.ip}] Error: Status code {response.status_code}")
                return None
            
            data = response.content
            if len(data) < 8:
                print(f"[{self.ip}] Error: Data missing (len={len(data)})")
                return None
            
            # Parse binary protocol: Magic(4) + MetadataLen(4) + Metadata + Image
            magic = data[:4]
            if magic != b'STP1':
                print(f"[{self.ip}] Error: Invalid magic {magic}")
                return None
            
            metadata_len = struct.unpack('>I', data[4:8])[0]
            metadata_bytes = data[8:8+metadata_len]
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            image_bytes = data[8+metadata_len:]
            
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            motion_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if motion_image is None:
                return None
            
            # Parse metadata
            loc = metadata.get('location', {})
            orient = metadata.get('orientation', {})
            cam = metadata.get('camera', {})
            quat = orient.get('quaternion', {})
            
            # Convert GPS to local coordinates (simple flat earth approximation)
            # Reference point: first device position
            lat = loc.get('latitude', 0)
            lon = loc.get('longitude', 0)
            alt = loc.get('altitude', 0)
            accuracy = loc.get('accuracy', 0.0)
            
            # Use a shared reference latitude so all devices share the same
            # longitudinal scale factor, keeping the coordinate frame Euclidean.
            if DeviceFetcher.reference_latitude is None and lat != 0:
                DeviceFetcher.reference_latitude = lat
            
            ref_lat = DeviceFetcher.reference_latitude if DeviceFetcher.reference_latitude is not None else lat
            
            # Simple conversion (meters from reference)
            # 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
            x = lon * 111000 * np.cos(np.radians(ref_lat))
            y = lat * 111000
            z = alt
            
            # Quaternion to rotation
            q = [quat.get('x', 0), quat.get('y', 0), quat.get('z', 0), quat.get('w', 1)]
            rotation = Rotation.from_quat(q)
            
            return DeviceData(
                device_id=metadata.get('deviceId', 'unknown'),
                timestamp=metadata.get('timestamp', 0),
                position=np.array([x, y, z]),
                orientation=rotation,
                fov_horizontal=cam.get('fovHorizontal', 60),
                fov_vertical=cam.get('fovVertical', 45),
                image_width=cam.get('imageWidth', 1920),
                image_height=cam.get('imageHeight', 1080),
                motion_image=motion_image,
                sensor_orientation=cam.get('sensorOrientation', 0),
                location_accuracy=accuracy
            )
        except Exception as e:
            print(f"Error fetching from {self.ip}: {e}")
            return None
    
    def start_continuous(self):
        """Start continuous fetching in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop continuous fetching."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _fetch_loop(self):
        while self.running:
            data = self.fetch_once()
            if data:
                self.last_data = data
            time.sleep(0.05)  # ~20 Hz

# =============================================================================
# Probability Grid — sliding-window volume accumulation
# =============================================================================

class ProbabilityGrid:
    """Sliding-window probability field for volume intersection.
    
    Accumulates per-frame sparse voxel contributions into a fixed-size deque.
    The probability field is the sum of all active frames in the window.
    """
    
    def __init__(self, grid_size: int = 100, voxel_size: float = 1.0,
                 grid_center: np.ndarray = None, window_size: int = 30):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.grid_center = grid_center if grid_center is not None else np.zeros(3)
        self.window_size = window_size
        self.frame_buffer: deque = deque(maxlen=window_size)
        self._field = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        half_size = 0.5 * grid_size * voxel_size
        self.grid_min = self.grid_center - half_size
        self.grid_max = self.grid_center + half_size
    
    def add_frame(self, sparse_voxels: List[Tuple[int, int, int, float]]):
        """Push a new frame's sparse voxel contributions onto the sliding window."""
        self.frame_buffer.append(sparse_voxels)
    
    def get_probability(self) -> np.ndarray:
        """Return the summed probability field from all frames in the window."""
        self._field.fill(0)
        for frame in self.frame_buffer:
            for ix, iy, iz, prob in frame:
                self._field[ix, iy, iz] += prob
        return self._field
    
    def set_center(self, center: np.ndarray):
        """Move the grid to a new center."""
        self.grid_center = center.copy()
        half_size = 0.5 * self.grid_size * self.voxel_size
        self.grid_min = self.grid_center - half_size
        self.grid_max = self.grid_center + half_size
    
    def clear(self):
        """Clear all frames from the sliding window."""
        self.frame_buffer.clear()

# =============================================================================
# Cone Caster — volume intersection ray marching
# =============================================================================

class ConeCaster:
    """Casts probability cones through a voxel grid.
    
    At each DDA step along the ray, a cone cross-section grows with distance.
    GPS accuracy defines the base radius, angular pixel uncertainty adds spread
    with distance. Voxels within the cone receive Gaussian-weighted probability.
    """
    
    # Precomputed sphere-offset cache: radius -> list of (dx, dy, dz)
    _offset_cache: Dict[int, List[Tuple[int, int, int]]] = {}
    
    def __init__(self, prob_grid: ProbabilityGrid):
        self.grid = prob_grid
    
    @classmethod
    def _sphere_offsets(cls, radius: int) -> List[Tuple[int, int, int]]:
        if radius not in cls._offset_cache:
            offsets = []
            r2 = radius * radius
            for dx in range(-radius, radius + 1):
                dx2 = dx * dx
                for dy in range(-radius, radius + 1):
                    dy2 = dy * dy
                    d2_xy = dx2 + dy2
                    if d2_xy > r2:
                        continue
                    for dz in range(-radius, radius + 1):
                        if d2_xy + dz * dz <= r2:
                            offsets.append((dx, dy, dz))
            cls._offset_cache[radius] = offsets
        return cls._offset_cache[radius]
    
    def cast_cone(self, origin: np.ndarray, direction: np.ndarray,
                  device: DeviceData, intensity: float = 1.0) -> List[Tuple[int, int, int, float]]:
        """Cast a probability cone from origin along direction.
        
        Returns list of (ix, iy, iz, probability) for voxels within the cone.
        """
        sparse = []
        gps_uncertainty = device.location_accuracy  # meters at origin
        angular_uncertainty = np.radians(CameraModel.angular_uncertainty(device))
        
        o = origin
        d = direction
        gmin = self.grid.grid_min
        gmax = self.grid.grid_max
        vs = self.grid.voxel_size
        gs = self.grid.grid_size
        max_cone_radius_voxels = 5  # cap spiral to keep perf reasonable
        
        # ---- ray-box intersection ----
        t_min = 0.0
        t_max = float('inf')
        for axis in range(3):
            if abs(d[axis]) < 1e-12:
                if o[axis] < gmin[axis] or o[axis] > gmax[axis]:
                    return sparse
            else:
                t1 = (gmin[axis] - o[axis]) / d[axis]
                t2 = (gmax[axis] - o[axis]) / d[axis]
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)
                if t_min > t_max:
                    return sparse
        
        if t_min < 0:
            t_min = 0
        
        # ---- start voxel ----
        start = o + t_min * d
        fx = (start[0] - gmin[0]) / vs
        fy = (start[1] - gmin[1]) / vs
        fz = (start[2] - gmin[2]) / vs
        ix, iy, iz = int(fx), int(fy), int(fz)
        if not (0 <= ix < gs and 0 <= iy < gs and 0 <= iz < gs):
            return sparse
        
        # ---- step direction ----
        step_x = 1 if d[0] >= 0 else -1
        step_y = 1 if d[1] >= 0 else -1
        step_z = 1 if d[2] >= 0 else -1
        
        # ---- t_delta ----
        inv_dx = 1.0 / abs(d[0]) if abs(d[0]) > 1e-12 else float('inf')
        inv_dy = 1.0 / abs(d[1]) if abs(d[1]) > 1e-12 else float('inf')
        inv_dz = 1.0 / abs(d[2]) if abs(d[2]) > 1e-12 else float('inf')
        t_delta_x = vs * inv_dx
        t_delta_y = vs * inv_dy
        t_delta_z = vs * inv_dz
        
        # ---- t_max ----
        next_x = (ix + (1 if step_x > 0 else 0)) * vs + gmin[0]
        next_y = (iy + (1 if step_y > 0 else 0)) * vs + gmin[1]
        next_z = (iz + (1 if step_z > 0 else 0)) * vs + gmin[2]
        t_max_x = (next_x - o[0]) / d[0] if abs(d[0]) > 1e-12 else float('inf')
        t_max_y = (next_y - o[1]) / d[1] if abs(d[1]) > 1e-12 else float('inf')
        t_max_z = (next_z - o[2]) / d[2] if abs(d[2]) > 1e-12 else float('inf')
        
        t_current = t_min
        max_steps = gs * 3
        
        for _ in range(max_steps):
            if not (0 <= ix < gs and 0 <= iy < gs and 0 <= iz < gs):
                break
            
            # ---- cone radius at this step ----
            distance = t_current
            cone_radius = gps_uncertainty + distance * np.tan(angular_uncertainty)
            radius_voxels = int(cone_radius / vs)
            if radius_voxels > max_cone_radius_voxels:
                radius_voxels = max_cone_radius_voxels
            
            # ---- Gaussian-weighted probability to cone cross-section ----
            sigma = max(1.0, cone_radius / 2.0) * vs  # in voxel-unit space
            sigma_sq_2 = 2.0 * sigma * sigma
            
            if radius_voxels == 0:
                sparse.append((ix, iy, iz, intensity))
            else:
                offsets = self._sphere_offsets(radius_voxels)
                for dx, dy, dz in offsets:
                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                    if not (0 <= nx < gs and 0 <= ny < gs and 0 <= nz < gs):
                        continue
                    d2 = float(dx * dx + dy * dy + dz * dz) * vs * vs
                    weight = intensity * np.exp(-d2 / sigma_sq_2)
                    if weight > 0.001:
                        sparse.append((nx, ny, nz, weight))
            
            # ---- DDA step ----
            if t_max_x < t_max_y and t_max_x < t_max_z:
                ix += step_x
                t_current = t_max_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_z:
                iy += step_y
                t_current = t_max_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_current = t_max_z
                t_max_z += t_delta_z
            
            if t_current > t_max:
                break
        
        return sparse

# =============================================================================
# Ray Caster — orchestrates camera, cone casting, and probability
# =============================================================================

class RayCaster:
    """Orchestrates volume intersection: camera model, cone casting, probability grid."""
    
    def __init__(self, grid_size: int = 100, voxel_size: float = 1.0,
                 grid_center: np.ndarray = None, window_size: int = 30):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.prob_grid = ProbabilityGrid(grid_size, voxel_size, grid_center, window_size)
        self.cone_caster = ConeCaster(self.prob_grid)
        # Backwards-compatible aliases for Visualizer
        self.grid_center = self.prob_grid.grid_center
        self.grid_min = self.prob_grid.grid_min
        self.grid_max = self.prob_grid.grid_max
    
    def reset(self):
        """Reset probability for a new frame (clear raw accumulator, not the window)."""
        pass  # per-frame sparse list managed by caller
    
    def process_device(self, device: DeviceData, sample_step: int = 10,
                       motion_threshold: int = 10,
                       on_ray: Optional[Callable[[np.ndarray, np.ndarray, float], None]] = None
                       ) -> List[Tuple[int, int, int, float]]:
        """Process motion image, cast cones for each motion pixel.
        
        Returns sparse voxel list for this device's frame contribution.
        """
        sparse = []
        h, w = device.motion_image.shape
        
        for v in range(0, h, sample_step):
            for u in range(0, w, sample_step):
                intensity = device.motion_image[v, u]
                if intensity < motion_threshold:
                    continue
                
                ray_dir = CameraModel.pixel_to_ray(u, v, device)
                cone = self.cone_caster.cast_cone(device.position, ray_dir,
                                                   device, intensity / 255.0)
                sparse.extend(cone)
                if on_ray:
                    on_ray(device.position, ray_dir, intensity / 255.0)
        
        return sparse
    
    def get_probability_field(self) -> np.ndarray:
        """Return the current summed probability field from the sliding window."""
        return self.prob_grid.get_probability()

# =============================================================================
# 3D Visualizer (Raylib)
# =============================================================================

class Visualizer:
    """3D visualization using Raylib."""

    def __init__(self):
        pr.init_window(1280, 720, "3D Ray Intersection Visualizer (Raylib)")
        pr.set_target_fps(60)
        
        # Initialize Camera
        self.camera = pr.Camera3D()
        self.camera.position = pr.Vector3(100.0, 100.0, 100.0)
        self.camera.target = pr.Vector3(0.0, 0.0, 0.0)
        self.camera.up = pr.Vector3(0.0, 0.0, 1.0) # Z-up
        self.camera.fovy = 45.0
        self.camera.projection = pr.CAMERA_PERSPECTIVE
        
        self.device_positions: Dict[str, pr.Vector3] = {}
        self.device_colors: Dict[str, pr.Color] = {}
        self.device_rays: List[Tuple[pr.Vector3, pr.Vector3, pr.Color]] = []
        self.voxels: List[Tuple[pr.Vector3, float, pr.Color]] = [] # pos, size, color
        self.render_origin = np.zeros(3)

    def set_origin(self, origin: np.ndarray):
        """Set the world origin for rendering relative coordinates."""
        self.render_origin = origin.copy()
        # Reset camera to look at origin (now 0,0,0 relative)
        self.camera.target = pr.Vector3(0.0, 0.0, 0.0)
        self.camera.position = pr.Vector3(100.0, 100.0, 100.0)

    def _to_raylib_color(self, color_float: List[float], alpha: float = 1.0) -> pr.Color:
        r = int(color_float[0] * 255)
        g = int(color_float[1] * 255)
        b = int(color_float[2] * 255)
        a = int(alpha * 255)
        return pr.Color(r, g, b, a)

    def update_device(self, device: DeviceData, color: List[float] = [1, 0, 0]):
        # Normalize position relative to render origin
        rel_pos = device.position - self.render_origin
        pos = pr.Vector3(float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2]))
        
        self.device_positions[device.device_id] = pos
        self.device_colors[device.device_id] = self._to_raylib_color(color)
        
        # Add a short direction indicator (optional, done in draw loop for simplicity if needed)

    def add_ray(self, start: np.ndarray, end: np.ndarray, color: List[float]):
        s_np = start - self.render_origin
        e_np = end - self.render_origin
        
        s = pr.Vector3(float(s_np[0]), float(s_np[1]), float(s_np[2]))
        e = pr.Vector3(float(e_np[0]), float(e_np[1]), float(e_np[2]))
        c = self._to_raylib_color(color, alpha=0.5)
        self.device_rays.append((s, e, c))

    def update_voxels(self, prob_field: np.ndarray, ray_caster: RayCaster,
                       threshold_percentile: float = 90):
        self.voxels = []
        max_val = np.max(prob_field)
        if max_val <= 0:
            return

        positive = prob_field[prob_field > 0]
        if positive.size == 0:
            return

        threshold = np.percentile(positive, threshold_percentile)
        indices = np.where(prob_field > threshold)
        if len(indices[0]) == 0:
            return

        # Limit voxels to avoid lag
        count = 0
        max_voxels = 2000

        for i, j, k in zip(*indices):
            if count > max_voxels:
                break

            x = ray_caster.grid_min[0] + (i + 0.5) * ray_caster.voxel_size
            y = ray_caster.grid_min[1] + (j + 0.5) * ray_caster.voxel_size
            z = ray_caster.grid_min[2] + (k + 0.5) * ray_caster.voxel_size

            pos_np = np.array([x, y, z]) - self.render_origin
            pos = pr.Vector3(float(pos_np[0]), float(pos_np[1]), float(pos_np[2]))

            intensity = prob_field[i, j, k] / max_val
            c = pr.Color(int(intensity * 255), 0, int((1.0 - intensity) * 255), 200)

            self.voxels.append((pos, ray_caster.voxel_size * 0.8, c))
            count += 1

    def should_close(self) -> bool:
        return pr.window_should_close()
    
    def close(self):
        pr.close_window()
        
    def begin_frame(self):
        pr.update_camera(self.camera, pr.CAMERA_FREE)
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.begin_mode_3d(self.camera)
        
        # Draw Grid and Axes
        pr.draw_grid(100, 10.0)
        pr.draw_line_3d(pr.Vector3(0,0,0), pr.Vector3(10,0,0), pr.RED)
        pr.draw_line_3d(pr.Vector3(0,0,0), pr.Vector3(0,10,0), pr.GREEN)
        pr.draw_line_3d(pr.Vector3(0,0,0), pr.Vector3(0,0,10), pr.BLUE)

    def draw_state(self):
        # Draw devices
        for dev_id, pos in self.device_positions.items():
            color = self.device_colors.get(dev_id, pr.RED)
            pr.draw_sphere(pos, 2.0, color)
            # Label could be added here
            
        # Draw rays
        for s, e, c in self.device_rays:
             pr.draw_line_3d(s, e, c)
        
        # Draw voxels
        for pos, size, c in self.voxels:
             pr.draw_cube(pos, size, size, size, c)
             
    def end_frame(self):
        pr.end_mode_3d()
        pr.draw_fps(10, 10)
        pr.draw_text("Controls: Mouse to rotate/zoom", 10, 30, 20, pr.WHITE)
        pr.end_drawing()
        
    def clear_rays(self):
        self.device_rays = []

# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <phone_ip1> [phone_ip2] ...")
        print("Example: python main.py 192.168.1.100 192.168.1.101")
        return
    
    phone_ips = sys.argv[1:]
    print(f"Connecting to {len(phone_ips)} device(s)...")
    
    # Create fetchers
    fetchers = [DeviceFetcher(ip) for ip in phone_ips]
    for fetcher in fetchers:
        fetcher.start_continuous()
    
    # Create ray caster centered around expected device positions
    # For now, use origin as center with 200m cube
    ray_caster = RayCaster(grid_size=100, voxel_size=2.0, grid_center=np.zeros(3))
    
    # Create visualizer
    vis = Visualizer()
    
    # Device colors
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    print("\nControls:")
    print("  Close window to exit")
    print("\nWaiting for data from devices...")
    
    try:
        frame_count = 0
        origin_set = False
        while not vis.should_close():
            vis.begin_frame()

            # Collect all sparse voxel contributions for this frame
            all_sparse: List[Tuple[int, int, int, float]] = []
            
            for i, fetcher in enumerate(fetchers):
                if fetcher.last_data:
                    device = fetcher.last_data
                    
                    if frame_count % 60 == 0:
                        print(f"Device {i} active: {device.device_id} pos={device.position}")
                    
                    # Set origin on first data received
                    if not origin_set:
                        ray_caster.prob_grid.set_center(device.position.copy())
                        # Update backwards-compat aliases
                        ray_caster.grid_center = ray_caster.prob_grid.grid_center
                        ray_caster.grid_min = ray_caster.prob_grid.grid_min
                        ray_caster.grid_max = ray_caster.prob_grid.grid_max
                        vis.set_origin(ray_caster.grid_center)
                        origin_set = True
                        print(f"Origin set to {ray_caster.grid_center}")
                    
                    # Process motion image: cast cones and collect rays
                    vis.clear_rays()
                    def collect_ray(origin, direction, intensity):
                        end = origin + direction * 50
                        vis.add_ray(origin, end, colors[i % len(colors)])
                    sparse = ray_caster.process_device(device, sample_step=20,
                                                        motion_threshold=15,
                                                        on_ray=collect_ray)
                    all_sparse.extend(sparse)
                    
                    vis.update_device(device, colors[i % len(colors)])
            
            # Commit this frame's contributions to the sliding window
            ray_caster.prob_grid.add_frame(all_sparse)
            
            # Update voxel visualization every 10 frames
            if frame_count % 10 == 0:
                prob_field = ray_caster.get_probability_field()
                vis.update_voxels(prob_field, ray_caster, threshold_percentile=80)
            
            vis.draw_state()
            vis.end_frame()
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for fetcher in fetchers:
            fetcher.stop()
        vis.close()

if __name__ == "__main__":
    main()
