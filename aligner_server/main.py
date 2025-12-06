#!/usr/bin/env python3
"""
3D Ray Intersection Visualizer

Connects to multiple phone devices, fetches motion detection data and odometry,
visualizes devices in 3D space, and performs ray intersection to detect objects.

Dependencies: numpy, scipy, requests, OpenCV, and VisPy (with a supported GUI
backend such as PyQt6).
"""

import requests
import struct
import json
import numpy as np
import cv2
import time
import sys
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.spatial.transform import Rotation
from vispy import app, scene  # Requires a GUI backend (e.g., PyQt6)

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

# =============================================================================
# Device Fetcher
# =============================================================================

class DeviceFetcher:
    """Fetches data from a phone device."""
    
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
            if response.status_code != 200:
                return None
            
            data = response.content
            if len(data) < 8:
                return None
            
            # Parse binary protocol: Magic(4) + MetadataLen(4) + Metadata + Image
            magic = data[:4]
            if magic != b'STP1':
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
            
            # Simple conversion (meters from reference)
            # 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
            x = lon * 111000 * np.cos(np.radians(lat))
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
                motion_image=motion_image
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
# Ray Caster (ported from ray_voxel.cpp)
# =============================================================================

class RayCaster:
    """Casts rays into a 3D voxel grid."""
    
    def __init__(self, grid_size: int = 100, voxel_size: float = 1.0, grid_center: np.ndarray = None):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.grid_center = grid_center if grid_center is not None else np.zeros(3)
        self.voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        
        # Grid bounds
        half_size = 0.5 * grid_size * voxel_size
        self.grid_min = self.grid_center - half_size
        self.grid_max = self.grid_center + half_size
    
    def reset(self):
        """Reset the voxel grid."""
        self.voxel_grid.fill(0)
    
    def pixel_to_ray(self, u: int, v: int, device: DeviceData) -> np.ndarray:
        """
        Convert pixel coordinates to ray direction in world space.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            device: Device data with camera parameters
            
        Returns:
            Normalized ray direction in world space
        """
        # Camera intrinsics from FOV
        cx = device.image_width / 2.0
        cy = device.image_height / 2.0
        fx = cx / np.tan(np.radians(device.fov_horizontal / 2))
        fy = cy / np.tan(np.radians(device.fov_vertical / 2))
        
        # Pixel to camera space (camera looks down -Z)
        x = (u - cx) / fx
        y = -(v - cy) / fy  # Flip Y
        z = -1.0
        
        ray_cam = np.array([x, y, z])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world space
        ray_world = device.orientation.apply(ray_cam)
        return ray_world
    
    def cast_ray(self, origin: np.ndarray, direction: np.ndarray, intensity: float = 1.0) -> List[Tuple[int, int, int]]:
        """
        Cast a ray using DDA algorithm and accumulate into voxel grid.
        
        Returns list of voxel indices the ray passes through.
        """
        voxels = []
        
        # Ray-box intersection
        t_min = 0.0
        t_max = float('inf')
        
        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < self.grid_min[i] or origin[i] > self.grid_max[i]:
                    return voxels
            else:
                t1 = (self.grid_min[i] - origin[i]) / direction[i]
                t2 = (self.grid_max[i] - origin[i]) / direction[i]
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)
                if t_min > t_max:
                    return voxels
        
        if t_min < 0:
            t_min = 0
        
        # Start position
        start = origin + t_min * direction
        fx = (start[0] - self.grid_min[0]) / self.voxel_size
        fy = (start[1] - self.grid_min[1]) / self.voxel_size
        fz = (start[2] - self.grid_min[2]) / self.voxel_size
        
        ix, iy, iz = int(fx), int(fy), int(fz)
        if not (0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size):
            return voxels
        
        # Step direction
        step_x = 1 if direction[0] >= 0 else -1
        step_y = 1 if direction[1] >= 0 else -1
        step_z = 1 if direction[2] >= 0 else -1
        
        # t_delta: how far along ray to cross one voxel
        t_delta_x = self.voxel_size / abs(direction[0]) if abs(direction[0]) > 1e-12 else float('inf')
        t_delta_y = self.voxel_size / abs(direction[1]) if abs(direction[1]) > 1e-12 else float('inf')
        t_delta_z = self.voxel_size / abs(direction[2]) if abs(direction[2]) > 1e-12 else float('inf')
        
        # t_max: distance to next voxel boundary
        next_x = (ix + (1 if step_x > 0 else 0)) * self.voxel_size + self.grid_min[0]
        next_y = (iy + (1 if step_y > 0 else 0)) * self.voxel_size + self.grid_min[1]
        next_z = (iz + (1 if step_z > 0 else 0)) * self.voxel_size + self.grid_min[2]
        
        t_max_x = (next_x - origin[0]) / direction[0] if abs(direction[0]) > 1e-12 else float('inf')
        t_max_y = (next_y - origin[1]) / direction[1] if abs(direction[1]) > 1e-12 else float('inf')
        t_max_z = (next_z - origin[2]) / direction[2] if abs(direction[2]) > 1e-12 else float('inf')
        
        t_current = t_min
        max_steps = self.grid_size * 3  # Limit iterations
        
        for _ in range(max_steps):
            if not (0 <= ix < self.grid_size and 0 <= iy < self.grid_size and 0 <= iz < self.grid_size):
                break
            
            voxels.append((ix, iy, iz))
            self.voxel_grid[ix, iy, iz] += intensity
            
            # Move to next voxel
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
        
        return voxels

    def process_device(self, device: DeviceData, sample_step: int = 10, motion_threshold: int = 10):
        """
        Process motion image from a device, casting rays for motion pixels.
        
        Args:
            device: Device data with motion image
            sample_step: Sample every N pixels (for performance)
            motion_threshold: Minimum brightness to consider as motion
        """
        h, w = device.motion_image.shape
        
        for v in range(0, h, sample_step):
            for u in range(0, w, sample_step):
                intensity = device.motion_image[v, u]
                if intensity < motion_threshold:
                    continue
                
                ray_dir = self.pixel_to_ray(u, v, device)
                self.cast_ray(device.position, ray_dir, intensity / 255.0)

# =============================================================================
# 3D Visualizer
# =============================================================================

class Visualizer:
    """3D visualization using VisPy."""

    def __init__(self):
        self.canvas = scene.SceneCanvas(
            title="3D Ray Intersection Visualizer",
            keys="interactive",
            size=(1280, 720),
            show=True,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(up="z", fov=60)
        scene.visuals.XYZAxis(parent=self.view.scene, width=5)

        self.device_positions: Dict[str, np.ndarray] = {}
        self.device_colors: Dict[str, np.ndarray] = {}
        self.device_dirs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.device_visual = scene.visuals.Markers(parent=self.view.scene)

        self.ray_visual = scene.visuals.Line(parent=self.view.scene, method="gl")
        self.ray_visual.visible = False

        self.voxel_visual = scene.visuals.Markers(parent=self.view.scene)
        self.voxel_visual.visible = False

        self.direction_visual = scene.visuals.Line(parent=self.view.scene, color="white", width=3)
        self.direction_visual.visible = False

        self.cardinal_length = 50.0
        self._init_cardinal_guides()

        self._closed = False
        self.canvas.events.close.connect(self._on_close)

    @staticmethod
    def _rgba(color: List[float]) -> np.ndarray:
        rgb = np.clip(np.array(color, dtype=np.float32), 0.0, 1.0)
        return np.concatenate([rgb, [1.0]])

    def _on_close(self, _event):
        self._closed = True

    def _init_cardinal_guides(self):
        origins = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        directions = np.array(
            [
                [0.0, self.cardinal_length, 0.0],  # North (+Y)
                [self.cardinal_length, 0.0, 0.0],  # East (+X)
                [0.0, -self.cardinal_length, 0.0],  # South (-Y)
                [-self.cardinal_length, 0.0, 0.0],  # West (-X)
            ],
            dtype=np.float32,
        )
        points = np.empty((origins.shape[0] * 2, 3), dtype=np.float32)
        colors = []
        connect = []
        labels = ["N", "E", "S", "W"]
        label_positions = []
        label_colors = ["white", "white", "white", "white"]
        for idx, (start, delta) in enumerate(zip(origins, directions)):
            points[idx * 2] = start
            points[idx * 2 + 1] = start + delta
            connect.append([idx * 2, idx * 2 + 1])
            c = [1.0, 1.0, 1.0, 0.5]
            colors.append(c)
            colors.append(c)
            label_positions.append(start + delta * 1.05)

        self.cardinal_visual = scene.visuals.Line(parent=self.view.scene, width=2.0, color="white")
        self.cardinal_visual.set_data(
            pos=points,
            connect=np.array(connect, dtype=np.uint32),
            color=np.array(colors, dtype=np.float32),
        )
        self.cardinal_labels = []
        for text, pos, color in zip(labels, label_positions, label_colors):
            label = scene.visuals.Text(
                text,
                color=color,
                parent=self.view.scene,
                font_size=12,
                anchor_x="center",
                anchor_y="center",
                pos=pos,
            )
            self.cardinal_labels.append(label)

    def _refresh_devices(self):
        if not self.device_positions:
            self.device_visual.visible = False
            return

        positions = np.stack(list(self.device_positions.values())).astype(np.float32)
        colors = np.stack(list(self.device_colors.values())).astype(np.float32)
        self.device_visual.set_data(positions, face_color=colors, size=15)
        self.device_visual.visible = True
        x_range = (float(np.min(positions[:, 0])), float(np.max(positions[:, 0])))
        y_range = (float(np.min(positions[:, 1])), float(np.max(positions[:, 1])))
        z_range = (float(np.min(positions[:, 2])), float(np.max(positions[:, 2])))
        self.view.camera.set_range(x=x_range, y=y_range, z=z_range)
        self._refresh_camera_dirs()

    def _refresh_camera_dirs(self):
        if not self.device_dirs:
            self.direction_visual.visible = False
            return

        points = []
        connect = []
        colors = []
        idx = 0
        for (origin, direction) in self.device_dirs.values():
            start = origin
            end = origin + direction
            points.append(start)
            points.append(end)
            connect.append([idx * 2, idx * 2 + 1])
            colors.append([1.0, 0.5, 0.0, 1.0])
            colors.append([1.0, 0.5, 0.0, 1.0])
            idx += 1

        pos = np.asarray(points, dtype=np.float32)
        self.direction_visual.set_data(
            pos=pos,
            connect=np.asarray(connect, dtype=np.uint32),
            color=np.asarray(colors, dtype=np.float32),
        )
        self.direction_visual.visible = True

    def update_device(self, device: DeviceData, color: List[float] = [1, 0, 0]):
        self.device_positions[device.device_id] = device.position.copy()
        self.device_colors[device.device_id] = self._rgba(color)
        look_dir = device.orientation.apply(np.array([0.0, 0.0, -1.0]))
        look_dir = look_dir / np.linalg.norm(look_dir)
        self.device_dirs[device.device_id] = (
            device.position.copy(),
            look_dir * 20.0,
        )
        self._refresh_devices()

    def draw_rays(self, device: DeviceData, ray_caster: RayCaster, max_rays: int = 100, ray_length: float = 50):
        points: List[np.ndarray] = []
        lines: List[List[int]] = []

        h, w = device.motion_image.shape
        step = max(1, int(np.sqrt(h * w / max_rays)))

        line_idx = 0
        for v in range(0, h, step):
            for u in range(0, w, step):
                intensity = device.motion_image[v, u]
                if intensity < 10:
                    continue

                ray_dir = ray_caster.pixel_to_ray(u, v, device)
                start = device.position
                end = start + ray_dir * ray_length

                points.append(start)
                points.append(end)
                lines.append([line_idx * 2, line_idx * 2 + 1])
                line_idx += 1

        if not points:
            self.ray_visual.visible = False
            return

        pos = np.asarray(points, dtype=np.float32)
        connect = np.asarray(lines, dtype=np.uint32)
        colors = np.tile(np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float32), (pos.shape[0], 1))
        self.ray_visual.set_data(pos=pos, connect=connect, color=colors, width=2.0)
        self.ray_visual.visible = True

    def update_voxels(self, ray_caster: RayCaster, threshold_percentile: float = 90):
        if np.max(ray_caster.voxel_grid) <= 0:
            self.voxel_visual.visible = False
            return

        positive = ray_caster.voxel_grid[ray_caster.voxel_grid > 0]
        if positive.size == 0:
            self.voxel_visual.visible = False
            return

        threshold = np.percentile(positive, threshold_percentile)
        indices = np.where(ray_caster.voxel_grid > threshold)
        if len(indices[0]) == 0:
            self.voxel_visual.visible = False
            return

        points = []
        colors = []
        max_val = np.max(ray_caster.voxel_grid)
        for i, j, k in zip(*indices):
            x = ray_caster.grid_min[0] + (i + 0.5) * ray_caster.voxel_size
            y = ray_caster.grid_min[1] + (j + 0.5) * ray_caster.voxel_size
            z = ray_caster.grid_min[2] + (k + 0.5) * ray_caster.voxel_size
            points.append([x, y, z])

            intensity = ray_caster.voxel_grid[i, j, k] / max_val if max_val > 0 else 0
            colors.append([intensity, 0.0, 1.0 - intensity, 1.0])

        self.voxel_visual.set_data(
            np.asarray(points, dtype=np.float32),
            face_color=np.asarray(colors, dtype=np.float32),
            size=5,
        )
        self.voxel_visual.visible = True

    def poll_events(self) -> bool:
        app.process_events()
        self.canvas.update()
        return not self._closed

    def close(self):
        self.canvas.close()

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
    print("  Press 'R' to reset voxel grid")
    print("\nWaiting for data from devices...")
    
    try:
        frame_count = 0
        while True:
            # Process each device
            for i, fetcher in enumerate(fetchers):
                if fetcher.last_data:
                    device = fetcher.last_data
                    
                    # Update grid center based on first device position
                    if frame_count == 0:
                        ray_caster.grid_center = device.position.copy()
                        ray_caster.grid_min = ray_caster.grid_center - 0.5 * ray_caster.grid_size * ray_caster.voxel_size
                        ray_caster.grid_max = ray_caster.grid_center + 0.5 * ray_caster.grid_size * ray_caster.voxel_size
                    
                    # Process motion image
                    ray_caster.process_device(device, sample_step=20, motion_threshold=15)
                    
                    # Update visualization
                    vis.update_device(device, colors[i % len(colors)])
                    vis.draw_rays(device, ray_caster, max_rays=50)
            
            # Update voxel visualization every 10 frames
            if frame_count % 10 == 0:
                vis.update_voxels(ray_caster, threshold_percentile=80)
            
            if not vis.poll_events():
                break
            
            frame_count += 1
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for fetcher in fetchers:
            fetcher.stop()
        vis.close()

if __name__ == "__main__":
    main()
