package com.mistrjirka.smelltheplane

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.mistrjirka.smelltheplane.ui.theme.SmellThePlaneTheme
import java.io.ByteArrayOutputStream
import java.util.Arrays
import java.util.UUID
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan

class MainActivity : ComponentActivity(), SensorEventListener, LocationListener {

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private val motionServer = MotionServer
    private val deviceId = UUID.randomUUID().toString()

    // Sensor Data
    @Volatile private var currentOrientation = OrientationData(Quaternion(0f, 0f, 0f, 1f), EulerAngles(0f, 0f, 0f))
    
    // Location Data
    @Volatile private var currentLocation = LocationData(0.0, 0.0, 0.0, 0f)

    // Camera Executor
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    
    // Camera references for rebinding
    private var cameraProvider: ProcessCameraProvider? = null
    private var previewView: PreviewView? = null
    
    // Camera FOV (calculated from Camera2 API)
    private var cameraFovHorizontal: Float = 60f
    private var cameraFovVertical: Float = 45f
    private var cameraSensorOrientation: Int = 0  // SENSOR_ORIENTATION from Camera2 API

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        // Start Server
        motionServer.start()
        
        // Register resolution change callback
        MotionSettings.onResolutionChanged = {
            runOnUiThread {
                cameraProvider?.let { provider ->
                    previewView?.let { preview ->
                        startCamera(provider, preview)
                    }
                }
            }
        }

        // Request Permissions
        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (permissions.all { it.value }) {
                startSensors()
                startLocation()
            }
        }

        if (allPermissionsGranted()) {
            startSensors()
            startLocation()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        setContent {
            SmellThePlaneTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    MainScreen(
                        serverIp = motionServer.getIpAddress(),
                        onCameraProvider = { provider, pv ->
                            cameraProvider = provider
                            previewView = pv
                            queryAvailableResolutions(provider)
                            startCamera(provider, pv)
                        }
                    )
                }
            }
        }
    }

    private fun startSensors() {
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)?.also { rotationVector ->
            sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    private fun startLocation() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 1f, this)
        }
    }
    
    private fun queryAvailableResolutions(cameraProvider: ProcessCameraProvider) {
        // CameraX doesn't provide a simple API to query all supported resolutions
        // Use common resolutions - the actual resolution will be the closest supported one
        MotionSettings.availableResolutions = listOf(
            android.util.Size(3840, 2160),  // 4K
            android.util.Size(2560, 1440),  // QHD
            android.util.Size(1920, 1080),  // Full HD
            android.util.Size(1280, 720),   // HD
            android.util.Size(960, 540),    // qHD
            android.util.Size(640, 480)     // VGA
        )
    }

    private fun calculateCameraFov() {
        try {
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            // Get back camera ID
            val cameraId = cameraManager.cameraIdList.firstOrNull { id ->
                val characteristics = cameraManager.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                facing == CameraCharacteristics.LENS_FACING_BACK
            } ?: return
            
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            
            // Get focal lengths and sensor size
            val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            
            if (focalLengths != null && focalLengths.isNotEmpty() && sensorSize != null) {
                val focalLength = focalLengths[0]  // Use first available focal length
                
                // Calculate FOV: FOV = 2 * atan(sensorSize / (2 * focalLength))
                val fovHorizontalRad = 2.0 * atan((sensorSize.width / (2.0 * focalLength)).toDouble())
                val fovVerticalRad = 2.0 * atan((sensorSize.height / (2.0 * focalLength)).toDouble())
                
                cameraFovHorizontal = Math.toDegrees(fovHorizontalRad).toFloat()
                cameraFovVertical = Math.toDegrees(fovVerticalRad).toFloat()
            }
            
            // Read sensor orientation (physical mounting rotation of the camera sensor)
            cameraSensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
            } catch (e: Exception) {
            Log.w("SmellThePlane", "Failed to calculate camera FOV, using defaults", e)
        }
    }

    private fun startCamera(cameraProvider: ProcessCameraProvider, previewView: PreviewView) {
        // Calculate FOV before setting up camera
        calculateCameraFov()
        
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        // Build resolution selector based on target settings
        val resolutionSelector = if (MotionSettings.targetWidth > 0 && MotionSettings.targetHeight > 0) {
            val targetSize = android.util.Size(MotionSettings.targetWidth, MotionSettings.targetHeight)
            androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                .setResolutionStrategy(
                    androidx.camera.core.resolutionselector.ResolutionStrategy(
                        targetSize,
                        androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                    )
                )
                .build()
        } else {
            androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                .setResolutionStrategy(androidx.camera.core.resolutionselector.ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                .build()
        }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, MotionAnalyzer { imageBytes, width, height ->
                    MotionSettings.currentResolution = android.util.Size(width, height)
                    updateServerData(imageBytes, width, height)
                })
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
        } catch (exc: Exception) {
            Log.e("SmellThePlane", "Failed to start camera", exc)
        }
    }

    private fun updateServerData(imageBytes: ByteArray, width: Int, height: Int) {
        val data = MotionData(
            deviceId = deviceId,
            timestamp = System.currentTimeMillis(),
            location = currentLocation,
            orientation = currentOrientation,
            camera = CameraData(
                fovHorizontal = cameraFovHorizontal,
                fovVertical = cameraFovVertical,
                imageWidth = width,
                imageHeight = height,
                sensorOrientation = cameraSensorOrientation
            )
        )
        motionServer.updateData(data, imageBytes)
    }
        override fun onDestroy() {
            super.onDestroy()
            if (!isChangingConfigurations) {
                motionServer.stop()
            }
            cameraExecutor.shutdown()
        }

    // SensorEventListener
    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            val rotationMatrix = FloatArray(9)
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            val orientation = FloatArray(3)
            SensorManager.getOrientation(rotationMatrix, orientation)

            // Quaternion from rotation vector
            val q = FloatArray(4)
            SensorManager.getQuaternionFromVector(q, event.values)

            currentOrientation = OrientationData(
                Quaternion(q[1], q[2], q[3], q[0]), // Android returns w, x, y, z
                EulerAngles(
                    Math.toDegrees(orientation[0].toDouble()).toFloat(),
                    Math.toDegrees(orientation[1].toDouble()).toFloat(),
                    Math.toDegrees(orientation[2].toDouble()).toFloat()
                )
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // LocationListener
    override fun onLocationChanged(location: Location) {
        currentLocation = LocationData(
            location.latitude,
            location.longitude,
            location.altitude,
            location.accuracy
        )
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
    }
}

@Composable
fun MainScreen(serverIp: String, onCameraProvider: (ProcessCameraProvider, PreviewView) -> Unit) {
    val context = LocalContext.current
    var previewView by remember { mutableStateOf<PreviewView?>(null) }

    LaunchedEffect(previewView) {
        if (previewView != null) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                onCameraProvider(cameraProvider, previewView!!)
            }, ContextCompat.getMainExecutor(context))
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Text(
            text = "Server running at: http://$serverIp:8080/data",
            modifier = Modifier.padding(16.dp)
        )
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).also {
                    previewView = it
                }
            },
            modifier = Modifier.fillMaxSize()
        )
    }
}

class MotionAnalyzer(private val onMotionDetected: (ByteArray, Int, Int) -> Unit) : ImageAnalysis.Analyzer {
    private var previousFrame: ByteArray? = null

    override fun analyze(image: ImageProxy) {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val currentY = extractLumaPlane(image)
        val diffBuffer = ByteArray(ySize)

        // Read settings from shared object
        val threshold = MotionSettings.threshold
        val gain = MotionSettings.gain
        val jpegQuality = MotionSettings.jpegQuality

        val lastFrame = previousFrame
        if (lastFrame != null && lastFrame.size == ySize) {
            for (i in 0 until ySize) {
                val curr = currentY[i].toInt() and 0xFF
                val prev = lastFrame[i].toInt() and 0xFF
                val rawDiff = abs(curr - prev)
                val amplified = minOf(255, rawDiff * gain)
                // Zero-out low values so only real motion remains
                diffBuffer[i] = if (amplified >= threshold) amplified.toByte() else 0
            }
        } else {
            Arrays.fill(diffBuffer, 0.toByte())
        }

        previousFrame = currentY

        val nv21 = ByteArray(ySize + (ySize / 2))
        System.arraycopy(diffBuffer, 0, nv21, 0, ySize)
        Arrays.fill(nv21, ySize, nv21.size, 0x80.toByte())

        val jpegBytes = ByteArrayOutputStream().use { out ->
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            yuvImage.compressToJpeg(Rect(0, 0, width, height), jpegQuality, out)
            out.toByteArray()
        }

        onMotionDetected(jpegBytes, width, height)
        image.close()
    }

    private fun extractLumaPlane(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val buffer = yPlane.buffer
        val rowStride = yPlane.rowStride
        val ySize = width * height
        val luma = ByteArray(ySize)

        buffer.rewind()
        if (rowStride == width) {
            buffer.get(luma, 0, ySize)
        } else {
            for (row in 0 until height) {
                buffer.position(row * rowStride)
                buffer.get(luma, row * width, width)
            }
        }
        return luma
    }
}