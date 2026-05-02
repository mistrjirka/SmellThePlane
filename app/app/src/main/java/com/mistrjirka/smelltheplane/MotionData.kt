package com.mistrjirka.smelltheplane

data class MotionData(
    val deviceId: String,
    val timestamp: Long,
    val location: LocationData,
    val orientation: OrientationData,
    val camera: CameraData
)

data class LocationData(
    val latitude: Double,
    val longitude: Double,
    val altitude: Double,
    val accuracy: Float
)

data class OrientationData(
    val quaternion: Quaternion,
    val euler: EulerAngles
)

data class Quaternion(val x: Float, val y: Float, val z: Float, val w: Float)
data class EulerAngles(val azimuth: Float, val pitch: Float, val roll: Float)

data class CameraData(
    val fovHorizontal: Float,  // Horizontal FOV in degrees
    val fovVertical: Float,    // Vertical FOV in degrees
    val imageWidth: Int,
    val imageHeight: Int,
    val sensorOrientation: Int = 0  // Camera sensor mounting rotation (degrees), typically 90 or 270
)
