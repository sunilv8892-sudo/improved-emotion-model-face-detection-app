package com.example.yolo_app.yolo_app

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.BufferedOutputStream

class MainActivity : FlutterActivity() {
	private val CHANNEL = "com.coad.faceattendance/save"

	override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
		super.configureFlutterEngine(flutterEngine)

		MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
			if (call.method == "saveToDownloads") {
				val filename = call.argument<String>("filename") ?: "export.dat"
				val dataBase64 = call.argument<String>("dataBase64") ?: ""
				val subFolder = call.argument<String>("subFolder") ?: ""
				try {
					val bytes = Base64.decode(dataBase64, Base64.DEFAULT)
					val saved = saveToDownloads(this, filename, bytes, subFolder)
					result.success(saved)
				} catch (e: Exception) {
					result.error("save_error", e.message, null)
				}
			} else {
				result.notImplemented()
			}
		}
	}

	private fun saveToDownloads(context: Context, filename: String, bytes: ByteArray, subFolder: String): Boolean {
		return try {
			val mime = when (filename.substringAfterLast('.', "").lowercase()) {
				"pdf" -> "application/pdf"
				"csv" -> "text/csv"
				"txt" -> "text/plain"
				else -> "application/octet-stream"
			}

			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
				val values = ContentValues().apply {
					put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
					put(MediaStore.MediaColumns.MIME_TYPE, mime)
					put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + if (subFolder.isNotEmpty()) "/$subFolder" else "")
				}
				val resolver = context.contentResolver
				val uri = resolver.insert(MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
				uri?.let {
					resolver.openOutputStream(it)?.use { out ->
						BufferedOutputStream(out).use { bos ->
							bos.write(bytes)
							bos.flush()
						}
					}
					return true
				}
				false
			} else {
				// Pre-Q: write directly to external public Downloads directory
				val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
				val folder = if (subFolder.isNotEmpty()) java.io.File(downloads, subFolder) else downloads
				if (!folder.exists()) folder.mkdirs()
				val file = java.io.File(folder, filename)
				file.outputStream().use { it.write(bytes) }
				true
			}
		} catch (e: Exception) {
			false
		}
	}
}
