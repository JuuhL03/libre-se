package com.margelo.nitro.libreslandmarker

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.margelo.nitro.NitroModules
import com.margelo.nitro.core.ArrayBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min


/**
 * Detector de maos e corpo do Libre-se no Android (MediaPipe Tasks), com os
 * mesmos ajustes da versao web (libre-se/web/app.js): modo IMAGE, confianca
 * 0.5, imagem espelhada e com no maximo 640 de largura.
 */
@DoNotStrip
@Keep
class HybridLibresLandmarker : HybridLibresLandmarkerSpec() {
  private val trava = Any()
  private var maos: HandLandmarker? = null
  private var pose: PoseLandmarker? = null
  private var numMaos = 2

  // reaproveitados entre quadros
  private var bruto: Bitmap? = null
  private var linha: ByteArray = ByteArray(0)
  private var px: IntArray = IntArray(0)

  private fun contexto() =
    NitroModules.applicationContext ?: throw IllegalStateException("contexto do app indisponivel")

  private fun criarMaos(n: Int): HandLandmarker {
    val opcoes = HandLandmarker.HandLandmarkerOptions.builder()
      .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
      .setRunningMode(RunningMode.IMAGE)
      .setNumHands(n)
      .setMinHandDetectionConfidence(0.5f)
      .setMinHandPresenceConfidence(0.5f)
      .setMinTrackingConfidence(0.5f)
      .build()
    return HandLandmarker.createFromOptions(contexto(), opcoes)
  }

  private fun criarPose(): PoseLandmarker {
    val opcoes = PoseLandmarker.PoseLandmarkerOptions.builder()
      .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build())
      .setRunningMode(RunningMode.IMAGE)
      .setNumPoses(1)
      .setMinPoseDetectionConfidence(0.5f)
      .setMinPosePresenceConfidence(0.5f)
      .setOutputSegmentationMasks(false)
      .build()
    return PoseLandmarker.createFromOptions(contexto(), opcoes)
  }

  override fun configurar(numMaos: Double) {
    val n = max(1, min(2, numMaos.toInt()))
    synchronized(trava) {
      if (maos == null || n != this.numMaos) {
        maos?.close()
        maos = criarMaos(n)
        this.numMaos = n
      }
    }
  }

  /** Quadro da camera -> Bitmap de pe, espelhado (se pedido) e com no maximo 640 de largura. */
  private fun prepararImagem(
    pixels: ArrayBuffer, w: Int, h: Int, bpr: Int, formato: String, orientacao: String, espelhar: Boolean,
  ): Bitmap {
    val buf: ByteBuffer = pixels.getBuffer(false).order(ByteOrder.nativeOrder())
    // a ultima linha costuma vir sem o preenchimento (Image.Plane do Android): basta bpr*(h-1) + w*4
    require(w > 0 && h > 0 && bpr >= w * 4 && buf.capacity() >= bpr.toLong() * (h - 1) + w * 4L) {
      "quadro invalido (${w}x$h, $bpr bytes por linha, ${buf.capacity()} bytes)"
    }
    val bgra = formato.contains("bgra")
    require(bgra || formato.contains("rgba") || formato.contains("rgb-rgb")) {
      "formato $formato nao suportado: use pixelFormat 'rgb'"
    }

    var bmp = bruto
    if (bmp == null || bmp.width != w || bmp.height != h) {
      bmp?.recycle()
      bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
      bruto = bmp
    }
    if (!bgra && bpr == w * 4) {
      // RGBA sem preenchimento: copia direto (ARGB_8888 guarda RGBA na memoria)
      buf.rewind()
      bmp!!.copyPixelsFromBuffer(buf)
    } else {
      // por linha: tira o preenchimento e, em BGRA, troca R e B
      if (px.size != w * h) px = IntArray(w * h)
      if (linha.size != w * 4) linha = ByteArray(w * 4)
      for (y in 0 until h) {
        buf.position(y * bpr)
        buf.get(linha, 0, w * 4)
        var i = 0
        val base = y * w
        for (x in 0 until w) {
          val c0 = linha[i].toInt() and 0xff
          val c1 = linha[i + 1].toInt() and 0xff
          val c2 = linha[i + 2].toInt() and 0xff
          val r = if (bgra) c2 else c0
          val b = if (bgra) c0 else c2
          px[base + x] = (0xff shl 24) or (r shl 16) or (c1 shl 8) or b
          i += 4
        }
      }
      bmp!!.setPixels(px, 0, w, 0, 0, w, h)
    }

    // 'right' = os pixels estao girados +90 graus em relacao ao desejado
    val graus = when (orientacao) {
      "right" -> -90f
      "left" -> 90f
      "down" -> 180f
      else -> 0f
    }
    val lw = if (graus == 90f || graus == -90f) h else w
    val escala = min(1f, 640f / lw)
    if (graus == 0f && !espelhar && escala >= 1f) return bmp

    val m = Matrix()
    if (graus != 0f) m.postRotate(graus)
    if (espelhar) m.postScale(-1f, 1f)        // como o cv2.flip do app de computador e o canvas da web
    if (escala < 1f) m.postScale(escala, escala)
    return Bitmap.createBitmap(bmp, 0, 0, w, h, m, true)
  }

  private fun r4(v: Float): Double = Math.round(v.toDouble() * 10000.0) / 10000.0

  override fun detectar(
    pixels: ArrayBuffer, largura: Double, altura: Double, bytesPorLinha: Double, formato: String,
    orientacao: String, espelhar: Boolean, comPose: Boolean,
  ): PontosQuadro {
    synchronized(trava) {
      val t0 = SystemClock.elapsedRealtimeNanos()
      val detMaos = maos ?: criarMaos(numMaos).also { maos = it }
      val detPose = if (comPose) (pose ?: criarPose().also { pose = it }) else null

      val bmp = prepararImagem(
        pixels, largura.toInt(), altura.toInt(), bytesPorLinha.toInt(), formato, orientacao, espelhar,
      )
      val imagem = BitmapImageBuilder(bmp).build()

      val pontos = ArrayList<Double>(126)
      val lados = ArrayList<Double>(2)
      val r = detMaos.detect(imagem)
      r.landmarks().take(2).forEachIndexed { k, mao ->
        for (p in mao) {
          pontos.add(r4(p.x())); pontos.add(r4(p.y())); pontos.add(r4(p.z()))
        }
        val nome = r.handedness().getOrNull(k)?.firstOrNull()?.categoryName() ?: ""
        lados.add(if (nome.isEmpty()) -1.0 else if (nome.lowercase().startsWith("r")) 1.0 else 0.0)
      }

      val corpo = ArrayList<Double>(132)
      if (detPose != null) {
        val pts = detPose.detect(imagem).landmarks().firstOrNull()
        pts?.forEach { q ->
          corpo.add(r4(q.x())); corpo.add(r4(q.y())); corpo.add(r4(q.z()))
          corpo.add(r4(q.visibility().orElse(1.0f)))
        }
      }

      val res = PontosQuadro(
        pontos.toDoubleArray(), lados.toDoubleArray(), corpo.toDoubleArray(),
        bmp.width.toDouble(), bmp.height.toDouble(),
        (SystemClock.elapsedRealtimeNanos() - t0) / 1e6,
      )
      if (bmp !== bruto) bmp.recycle()
      return res
    }
  }
}
