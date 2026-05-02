# SmellThePlane - OpenCode Instructions

## System Architecture

**Distributed 3D object tracking system**: Multiple phones detect motion in their camera view with GPS position, server triangulates object locations in 3D space (e.g., airplanes).

- **`app/`** - Android sensor node (Kotlin/Compose) - captures motion + GPS, serves data via HTTP
- **`aligner_server/`** - Python server - aggregates multi-phone data, ray intersection for 3D triangulation
- **`Pixeltovoxelprojector/`** - Legacy/reference voxel projection code (not actively maintained)

## Android App (`app/`)

**Tech Stack:** Kotlin, Jetpack Compose, CameraX, Ktor Server (embedded)

**Key Files:**
- `app/src/main/java/.../MainActivity.kt` - Main activity, sensor/camera setup
- `app/src/main/java/.../MotionServer.kt` - Ktor HTTP server (port 8080)
- `app/src/main/java/.../MotionData.kt` - Data classes for motion data

**Build Commands:**
```bash
cd app
./gradlew assembleDebug    # Build debug APK
./gradlew test             # Run unit tests
```

**Server Endpoints:**
- `GET /data` - Returns binary motion data (magic "STP1" + metadata JSON + JPEG)
- `GET /settings` - Get current motion settings
- `POST /settings` - Update threshold, gain, jpegQuality
- `POST /resolution` - Change camera resolution

**MotionSettings Object:** Shared state accessible from HTTP API
- `threshold` (default: 50) - Motion detection threshold
- `gain` (default: 2) - Difference amplification
- `jpegQuality` (default: 90)
- Resolution change triggers camera rebinding via `onResolutionChanged` callback

## Aligner Server (`aligner_server/`)

**Tech Stack:** Python 3, PyRay (raylib), NumPy, SciPy, OpenCV

**Run:**
```bash
cd aligner_server
python main.py <phone_ip1> [phone_ip2] ...
# Example: python main.py 192.168.1.100 192.168.1.101
```

**Binary Protocol:** `STP1` (4 bytes) + MetadataLen (4 bytes, big-endian) + Metadata JSON + JPEG image

**Key Classes:**
- `DeviceFetcher` - Fetches data from phone devices
- `RayCaster` - 3D ray intersection (voxel grid)
- `Visualizer` - PyRay 3D visualization

## Pixeltovoxelprojector (`Pixeltovoxelprojector/`)

Legacy voxel projection code. Not actively maintained.

**Scripts:**
- `blenderrenderscript.py` - Blender rendering automation
- `spacevoxelviewer.py` - Space voxel visualization
- `voxelmotionviewer.py` - Motion voxel viewer
- `ray_voxel.cpp` / `process_image.cpp` - C++ ray casting

## Git Workflow

- Root repo contains all components as subdirectories
- Each subdirectory (`app/`, `Pixeltovoxelprojector/`) may have its own git history
