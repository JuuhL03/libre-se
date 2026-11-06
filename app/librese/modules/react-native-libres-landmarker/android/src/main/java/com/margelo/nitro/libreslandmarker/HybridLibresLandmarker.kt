package com.margelo.nitro.libreslandmarker

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.margelo.nitro.NitroModules
import com.margelo.nitro.core.ArrayBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


/**
 * Detector de maos e corpo do Libre-se no Android (MediaPipe Tasks), com os
 * mesmos ajustes da versao web (libre-se/web/app.js): modo IMAGE, confianca
 * 0.5, imagem espelhada e com no maximo 640 de largura.
 *
 * Roda na GPU do celular (como a web faz quando a CPU e lenta: os pontos mudam
 * ~0,005, do tamanho do tremor normal do MediaPipe); se o aparelho nao suportar,
 * volta sozinho para a CPU. Os detectores sao criados na thread da camera, a
 * mesma que detecta (o configurar() so anota o numero de maos).
 */
@DoNotStrip
@Keep
class HybridLibresLandmarker : HybridLibresLandmarkerSpec() {
  private val trava = Any()
  private var maos: HandLandmarker? = null
  private var pose: PoseLandmarker? = null
  private var numMaos = 2          // o do detector atual
  private var maosPedidas = 2      // o que o app pediu (1 no alfabeto, 2 nas palavras)
  private var gpu = true           // tenta a GPU; na falha, CPU de vez

  // reaproveitados entre quadros (sem alocar ~1 MB por quadro)
  private var bruto: Bitmap? = null
  private var saida: Bitmap? = null
  private var compacto: ByteBuffer? = null
  private var linha: ByteArray = ByteArray(0)
  private var px: IntArray = IntArray(0)
  private val pintura = Paint(Paint.FILTER_BITMAP_FLAG)

  private fun contexto() =
    NitroModules.applicationContext ?: throw IllegalStateException("contexto do app indisponivel")

  private fun base(modelo: String): BaseOptions =
    BaseOptions.builder()
      .setModelAssetPath(modelo)
      .setDelegate(if (gpu) Delegate.GPU else Delegate.CPU)
      .build()

  private fun criarMaos(n: Int): HandLandmarker {
    val opcoes = HandLandmarker.HandLandmarkerOptions.builder()
      .setBaseOptions(base("hand_landmarker.task"))
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
      .setBaseOptions(base("pose_landmarker_lite.task"))
      .setRunningMode(RunningMode.IMAGE)
      .setNumPoses(1)
      .setMinPoseDetectionConfidence(0.5f)
      .setMinPosePresenceConfidence(0.5f)
      .setOutputSegmentationMasks(false)
      .build()
    return PoseLandmarker.createFromOptions(contexto(), opcoes)
  }

  /** A GPU falhou (criar ou detectar): fecha tudo e segue na CPU. */
  private fun paraCpu(erro: Throwable) {
    Log.w(TAG, "GPU indisponivel neste aparelho; usando a CPU", erro)
    gpu = false
    maos?.close(); maos = null
    pose?.close(); pose = null
  }

  /** Roda [bloco]; se falhar na GPU, passa para a CPU e tenta de novo uma vez. */
  private fun <T> comFallback(bloco: () -> T): T =
    try {
      bloco()
    } catch (e: Throwable) {
      if (!gpu) throw e
      paraCpu(e)
      bloco()
    }

  private fun detectorMaos(): HandLandmarker {
    val atual = maos
    if (atual != null && numMaos == maosPedidas) return atual
    atual?.close()
    maos = null
    val novo = criarMaos(maosPedidas)
    maos = novo
    numMaos = maosPedidas
    return novo
  }

  private fun detectorPose(): PoseLandmarker = pose ?: criarPose().also { pose = it }

  override fun configurar(numMaos: Double) {
    val n = max(1, min(2, numMaos.toInt()))
    synchronized(trava) { maosPedidas = n }   // o detector muda no proximo quadro
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
    } else if (!bgra) {
      // RGBA com preenchimento no fim de cada linha: tira o preenchimento copiando
      // linha a linha em bloco (antes era pixel a pixel, ~300 mil por quadro)
      val tam = w * 4 * h
      var c = compacto
      if (c == null || c.capacity() != tam) {
        c = ByteBuffer.allocateDirect(tam)
        compacto = c
      }
      c!!.clear()
      val d = buf.duplicate()
      for (y in 0 until h) {
        val ini = y * bpr
        d.limit(ini + w * 4)   // o limite primeiro: a posicao atual (fim da linha anterior) cabe nele
        d.position(ini)
        c.put(d)
      }
      c.rewind()
      bmp!!.copyPixelsFromBuffer(c)
    } else {
      // BGRA: troca R e B pixel a pixel
      if (px.size != w * h) px = IntArray(w * h)
      if (linha.size != w * 4) linha = ByteArray(w * 4)
      for (y in 0 until h) {
        buf.position(y * bpr)
        buf.get(linha, 0, w * 4)
        var i = 0
        val base = y * w
        for (x in 0 until w) {
          val b = linha[i].toInt() and 0xff
          val g = linha[i + 1].toInt() and 0xff
          val r = linha[i + 2].toInt() and 0xff
          px[base + x] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
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
    val deitado = graus == 90f || graus == -90f
    val lw = if (deitado) h else w
    val lh = if (deitado) w else h
    val escala = min(1f, 640f / lw)
    if (graus == 0f && !espelhar && escala >= 1f) return bmp

    // gira, espelha e reduz desenhando num Bitmap reaproveitado (antes: um Bitmap novo por quadro)
    val fw = max(1, (lw * escala).roundToInt())
    val fh = max(1, (lh * escala).roundToInt())
    var out = saida
    if (out == null || out.width != fw || out.height != fh) {
      out?.recycle()
      out = Bitmap.createBitmap(fw, fh, Bitmap.Config.ARGB_8888)
      saida = out
    }
    val m = Matrix()
    m.postTranslate(-w / 2f, -h / 2f)               // centro da imagem na origem
    if (graus != 0f) m.postRotate(graus)
    if (espelhar) m.postScale(-1f, 1f)              // como o cv2.flip do app de computador e o canvas da web
    if (escala < 1f) m.postScale(escala, escala)
    m.postTranslate(fw / 2f, fh / 2f)               // e de volta no centro da saida
    Canvas(out!!).drawBitmap(bmp!!, m, pintura)
    return out
  }

  private fun r4(v: Float): Double = Math.round(v.toDouble() * 10000.0) / 10000.0

  private fun agoraMs(): Double = SystemClock.elapsedRealtimeNanos() / 1e6

  override fun detectar(
    pixels: ArrayBuffer, largura: Double, altura: Double, bytesPorLinha: Double, formato: String,
    orientacao: String, espelhar: Boolean, comPose: Boolean,
  ): PontosQuadro {
    synchronized(trava) {
      val t0 = agoraMs()
      val bmp = prepararImagem(
        pixels, largura.toInt(), altura.toInt(), bytesPorLinha.toInt(), formato, orientacao, espelhar,
      )
      val imagem: MPImage = BitmapImageBuilder(bmp).build()
      val t1 = agoraMs()

      val pontos = ArrayList<Double>(126)
      val lados = ArrayList<Double>(2)
      val r: HandLandmarkerResult = comFallback { detectorMaos().detect(imagem) }
      r.landmarks().take(2).forEachIndexed { k, mao ->
        for (p in mao) {
          pontos.add(r4(p.x())); pontos.add(r4(p.y())); pontos.add(r4(p.z()))
        }
        val nome = r.handedness().getOrNull(k)?.firstOrNull()?.categoryName() ?: ""
        lados.add(if (nome.isEmpty()) -1.0 else if (nome.lowercase().startsWith("r")) 1.0 else 0.0)
      }
      val t2 = agoraMs()

      val corpo = ArrayList<Double>(132)
      if (comPose) {
        val pts = comFallback { detectorPose().detect(imagem) }.landmarks().firstOrNull()
        pts?.forEach { q ->
          corpo.add(r4(q.x())); corpo.add(r4(q.y())); corpo.add(r4(q.z()))
          corpo.add(r4(q.visibility().orElse(1.0f)))
        }
      }
      val t3 = agoraMs()

      return PontosQuadro(
        pontos.toDoubleArray(), lados.toDoubleArray(), corpo.toDoubleArray(),
        bmp.width.toDouble(), bmp.height.toDouble(),
        t3 - t0, t1 - t0, t2 - t1, t3 - t2,
        if (gpu) "GPU" else "CPU",
      )
    }
  }

  companion object {
    private const val TAG = "LibresLandmarker"
  }
}
