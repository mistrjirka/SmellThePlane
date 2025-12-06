package com.mistrjirka.smelltheplane

import android.util.Size
import com.google.gson.Gson
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondBytes
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import java.net.Inet4Address
import java.net.NetworkInterface
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Shared settings accessible from both the camera pipeline and the HTTP API
object MotionSettings {
    @Volatile var threshold: Int = 50
    @Volatile var gain: Int = 2
    @Volatile var jpegQuality: Int = 90

    @Volatile var targetWidth: Int = 0
    @Volatile var targetHeight: Int = 0
    @Volatile var availableResolutions: List<Size> = emptyList()
    @Volatile var currentResolution: Size? = null

    var onResolutionChanged: (() -> Unit)? = null
}

data class ResolutionRequest(val width: Int, val height: Int)

data class MotionSettingsRequest(
    val threshold: Int? = null,
    val gain: Int? = null,
    val jpegQuality: Int? = null
)

object MotionServer {

    private const val PORT = 8080

    private val gson = Gson()
    @Volatile private var server: ApplicationEngine? = null
    @Volatile private var currentMetadata: MotionData? = null
    @Volatile private var currentImageBytes: ByteArray? = null

    fun start() {
        synchronized(this) {
            if (server != null) return

            val engine = embeddedServer(Netty, port = PORT) {
                routing {
                    get("/data") {
                        val metadata = currentMetadata
                        val image = currentImageBytes

                        if (metadata != null && image != null) {
                            val metadataJson = gson.toJson(metadata).toByteArray(Charsets.UTF_8)
                            val magicBytes = "STP1".toByteArray(Charsets.UTF_8)
                            val metadataLength = ByteBuffer.allocate(4)
                                .order(ByteOrder.BIG_ENDIAN)
                                .putInt(metadataJson.size)
                                .array()

                            val totalSize = magicBytes.size + metadataLength.size + metadataJson.size + image.size
                            val buffer = ByteBuffer.allocate(totalSize)
                            buffer.put(magicBytes)
                            buffer.put(metadataLength)
                            buffer.put(metadataJson)
                            buffer.put(image)

                            call.respondBytes(buffer.array())
                        } else {
                            call.respondBytes(
                                bytes = "WAIT".toByteArray(),
                                status = HttpStatusCode.NoContent
                            )
                        }
                    }

                    get("/settings") {
                        call.respondText(gson.toJson(buildSettingsPayload()), ContentType.Application.Json)
                    }

                    post("/settings") {
                        try {
                            val body = call.receiveText()
                            val request = gson.fromJson(body, MotionSettingsRequest::class.java)

                            request.threshold?.let { MotionSettings.threshold = it }
                            request.gain?.let { MotionSettings.gain = it }
                            request.jpegQuality?.let { MotionSettings.jpegQuality = it }

                            val response = buildSettingsPayload().toMutableMap()
                            response["status"] = "ok"
                            call.respondText(gson.toJson(response), ContentType.Application.Json)
                        } catch (e: Exception) {
                            val response = mapOf("error" to (e.message ?: "Unknown error"))
                            call.respondText(
                                gson.toJson(response),
                                ContentType.Application.Json,
                                HttpStatusCode.BadRequest
                            )
                        }
                    }

                    post("/resolution") {
                        try {
                            val body = call.receiveText()
                            val request = gson.fromJson(body, ResolutionRequest::class.java)

                            val isValid = MotionSettings.availableResolutions.any {
                                it.width == request.width && it.height == request.height
                            }

                            if (!isValid) {
                                val response = mapOf(
                                    "status" to "error",
                                    "message" to "Resolution ${request.width}x${request.height} is not available"
                                )
                                call.respondText(
                                    gson.toJson(response),
                                    ContentType.Application.Json,
                                    HttpStatusCode.BadRequest
                                )
                                return@post
                            }

                            MotionSettings.targetWidth = request.width
                            MotionSettings.targetHeight = request.height
                            MotionSettings.onResolutionChanged?.invoke()

                            val response = buildSettingsPayload().toMutableMap()
                            response["status"] = "ok"
                            response["message"] = "Resolution change requested. Camera will restart."
                            call.respondText(gson.toJson(response), ContentType.Application.Json)
                        } catch (e: Exception) {
                            val response = mapOf("error" to (e.message ?: "Unknown error"))
                            call.respondText(
                                gson.toJson(response),
                                ContentType.Application.Json,
                                HttpStatusCode.BadRequest
                            )
                        }
                    }

                    get("/") {
                        call.respondText("Running: ${getIpAddress()}")
                    }
                }
            }

            engine.start(wait = false)
            server = engine
        }
    }

    fun stop() {
        synchronized(this) {
            server?.stop(1000, 2000)
            server = null
        }
    }

    fun updateData(metadata: MotionData, imageBytes: ByteArray) {
        currentMetadata = metadata
        currentImageBytes = imageBytes
    }

    fun getIpAddress(): String {
        try {
            val en = NetworkInterface.getNetworkInterfaces()
            while (en.hasMoreElements()) {
                val intf = en.nextElement()
                val enumIpAddr = intf.inetAddresses
                while (enumIpAddr.hasMoreElements()) {
                    val inetAddress = enumIpAddr.nextElement()
                    if (!inetAddress.isLoopbackAddress && inetAddress is Inet4Address) {
                        return inetAddress.hostAddress ?: "Unknown"
                    }
                }
            }
        } catch (ex: Exception) {
            ex.printStackTrace()
        }
        return "Unknown"
    }
}

private fun buildSettingsPayload(): MutableMap<String, Any?> {
    val available = MotionSettings.availableResolutions.map {
        mapOf("width" to it.width, "height" to it.height)
    }
    val current = MotionSettings.currentResolution?.let {
        mapOf("width" to it.width, "height" to it.height)
    }
    val target = if (MotionSettings.targetWidth > 0 && MotionSettings.targetHeight > 0) {
        mapOf("width" to MotionSettings.targetWidth, "height" to MotionSettings.targetHeight)
    } else {
        "highest"
    }
    return mutableMapOf(
        "threshold" to MotionSettings.threshold,
        "gain" to MotionSettings.gain,
        "jpegQuality" to MotionSettings.jpegQuality,
        "current" to current,
        "target" to target,
        "available" to available
    )
}
