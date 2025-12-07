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
from typing import List, Dict, Optional, Tuple
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
        """Move pixel_to_ray here (code omitted for brevity, logic preserved)"""
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
        """Same Cast Ray method"""
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
        """Process motion image from a device, casting rays for motion pixels."""
        h, w = device.motion_image.shape
        
        for v in range(0, h, sample_step):
            for u in range(0, w, sample_step):
                intensity = device.motion_image[v, u]
                if intensity < motion_threshold:
                    continue
                
                ray_dir = self.pixel_to_ray(u, v, device)
                self.cast_ray(device.position, ray_dir, intensity / 255.0)

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

    def draw_rays(self, device: DeviceData, ray_caster: RayCaster, max_rays: int = 100, ray_length: float = 50):
        # We collect rays to draw in the frame
        # Clear old rays for this device? Or just accumulate for one frame?
        # Better: clear all rays every frame in main loop before update, 
        # or have a list here that is cleared.
        # Let's clear rays if we are updating fresh.
        pass # We'll handle ray collection differently or just draw immediately if we were in drawing mode.
        # Raylib is immediate mode. We need to store data to draw.
        
    def add_ray(self, start: np.ndarray, end: np.ndarray, color: List[float]):
        s_np = start - self.render_origin
        e_np = end - self.render_origin
        
        s = pr.Vector3(float(s_np[0]), float(s_np[1]), float(s_np[2]))
        e = pr.Vector3(float(e_np[0]), float(e_np[1]), float(e_np[2]))
        c = self._to_raylib_color(color, alpha=0.5)
        self.device_rays.append((s, e, c))

    def update_voxels(self, ray_caster: RayCaster, threshold_percentile: float = 90):
        self.voxels = []
        if np.max(ray_caster.voxel_grid) <= 0:
            return

        positive = ray_caster.voxel_grid[ray_caster.voxel_grid > 0]
        if positive.size == 0:
            return

        threshold = np.percentile(positive, threshold_percentile)
        indices = np.where(ray_caster.voxel_grid > threshold)
        if len(indices[0]) == 0:
            return

        max_val = np.max(ray_caster.voxel_grid)
        
        # Limit voxels to avoid lag
        count = 0
        max_voxels = 2000 
        
        for i, j, k in zip(*indices):
            if count > max_voxels: break
            
            x = ray_caster.grid_min[0] + (i + 0.5) * ray_caster.voxel_size
            y = ray_caster.grid_min[1] + (j + 0.5) * ray_caster.voxel_size
            z = ray_caster.grid_min[2] + (k + 0.5) * ray_caster.voxel_size
            
            # Normalize
            pos_np = np.array([x, y, z]) - self.render_origin
            pos = pr.Vector3(float(pos_np[0]), float(pos_np[1]), float(pos_np[2]))
            
            intensity = ray_caster.voxel_grid[i, j, k] / max_val if max_val > 0 else 0
            # Heatmap color: Blue -> Red
            c = pr.Color(int(intensity*255), 0, int((1.0-intensity)*255), 200)
            
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

            # Process each device
            vis.clear_rays()
            for i, fetcher in enumerate(fetchers):
                if fetcher.last_data:
                    # debug print (once per device per second)
                    if frame_count % 60 == 0:
                        print(f"Device {i} active: {fetcher.last_data.device_id} pos={fetcher.last_data.position}")
                    device = fetcher.last_data
                    
                    # Set origin on first data received
                    if not origin_set:
                        ray_caster.grid_center = device.position.copy()
                        ray_caster.grid_min = ray_caster.grid_center - 0.5 * ray_caster.grid_size * ray_caster.voxel_size
                        ray_caster.grid_max = ray_caster.grid_center + 0.5 * ray_caster.grid_size * ray_caster.voxel_size
                        
                        # Use Raylib method to center camera
                        vis.set_origin(ray_caster.grid_center)
                        origin_set = True
                        print(f"Origin set to {ray_caster.grid_center}")
                    
                    # Process motion image
                    ray_caster.process_device(device, sample_step=20, motion_threshold=15)
                    
                    # Update visualization
                    vis.update_device(device, colors[i % len(colors)])
                    # vis.draw_rays(device, ray_caster, max_rays=50) # Need to re-implement drawing rays per frame
                    # Collect rays for drawing
                    h, w = device.motion_image.shape
                    step = max(1, int(np.sqrt(h * w / 50))) # 50 max rays
                    for v in range(0, h, step):
                        for u in range(0, w, step):
                            intensity = device.motion_image[v, u]
                            if intensity < 15: continue
                            ray_dir = ray_caster.pixel_to_ray(u, v, device)
                            start = device.position
                            end = start + ray_dir * 50
                            vis.add_ray(start, end, colors[i % len(colors)])
            
            # Update voxel visualization every 10 frames
            if frame_count % 10 == 0:
                vis.update_voxels(ray_caster, threshold_percentile=80)
            
            vis.draw_state()
            vis.end_frame()
            
            frame_count += 1
            # time.sleep(0.01) # Raylib handles timing via SetTargetFPS
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for fetcher in fetchers:
            fetcher.stop()
        vis.close()

if __name__ == "__main__":
    main()
