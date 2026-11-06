//
// HybridLibresLandmarker.swift
//
// Detector de maos e corpo do Libre-se no iPhone (MediaPipe Tasks), com os
// mesmos ajustes da versao web (libre-se/web/app.js): modo IMAGE, confianca
// 0.5, imagem espelhada e com no maximo 640 de largura.
//

import Accelerate
import CoreVideo
import Foundation
import MediaPipeTasksVision
import NitroModules

/// A mesma largura da extracao dos videos de treino (LARGURA_DETECCAO na web).
private let larguraDeteccao = 640

enum ErroLandmarker: Error, CustomStringConvertible {
  case modelo(String)
  case imagem(String)

  var description: String {
    switch self {
    case .modelo(let m): return "modelo: \(m)"
    case .imagem(let m): return "imagem: \(m)"
    }
  }
}

class HybridLibresLandmarker: HybridLibresLandmarkerSpec {
  private let trava = NSLock()
  private var maos: HandLandmarker?
  private var pose: PoseLandmarker?
  private var numMaos = 2

  // buffers reaproveitados entre quadros (evita alocar 1 MB a cada quadro)
  private var bufA: UnsafeMutableRawPointer?
  private var bufB: UnsafeMutableRawPointer?
  private var capacidade = 0

  deinit {
    bufA?.deallocate()
    bufB?.deallocate()
  }

  // MARK: - modelos

  private static func caminho(_ nome: String) throws -> String {
    let candidatos = [Bundle.main, Bundle(for: HybridLibresLandmarker.self)]
    for b in candidatos {
      if let p = b.path(forResource: nome, ofType: "task") { return p }
      if let url = b.url(forResource: "LibresLandmarkerModelos", withExtension: "bundle"),
        let rb = Bundle(url: url), let p = rb.path(forResource: nome, ofType: "task")
      {
        return p
      }
    }
    throw ErroLandmarker.modelo("\(nome).task nao esta no app")
  }

  private func criarMaos(_ n: Int) throws -> HandLandmarker {
    let o = HandLandmarkerOptions()
    o.baseOptions.modelAssetPath = try Self.caminho("hand_landmarker")
    o.runningMode = .image
    o.numHands = n
    o.minHandDetectionConfidence = 0.5
    o.minHandPresenceConfidence = 0.5
    o.minTrackingConfidence = 0.5
    return try HandLandmarker(options: o)
  }

  private func criarPose() throws -> PoseLandmarker {
    let o = PoseLandmarkerOptions()
    o.baseOptions.modelAssetPath = try Self.caminho("pose_landmarker_lite")
    o.runningMode = .image
    o.numPoses = 1
    o.minPoseDetectionConfidence = 0.5
    o.minPosePresenceConfidence = 0.5
    o.shouldOutputSegmentationMasks = false
    return try PoseLandmarker(options: o)
  }

  func configurar(numMaos: Double) throws {
    let n = max(1, min(2, Int(numMaos)))
    trava.lock()
    defer { trava.unlock() }
    if maos == nil || n != self.numMaos {
      maos = try criarMaos(n)
      self.numMaos = n
    }
  }

  // MARK: - imagem

  private func garantirCapacidade(_ bytes: Int) {
    if bytes <= capacidade { return }
    bufA?.deallocate()
    bufB?.deallocate()
    bufA = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
    bufB = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
    capacidade = bytes
  }

  /// Quadro da camera -> CVPixelBuffer BGRA de pe, espelhado (se pedido) e com no maximo 640 de largura.
  private func prepararImagem(
    _ pixels: ArrayBuffer, _ w: Int, _ h: Int, _ bpr: Int, _ formato: String,
    _ orientacao: String, _ espelhar: Bool
  ) throws -> CVPixelBuffer {
    guard w > 0, h > 0, bpr >= w * 4, pixels.size >= bpr * h else {
      throw ErroLandmarker.imagem("quadro invalido (\(w)x\(h), \(bpr) bytes por linha, \(pixels.size) bytes)")
    }
    let rgba = formato.contains("rgba")
    guard rgba || formato.contains("bgra") else {
      throw ErroLandmarker.imagem("formato \(formato) nao suportado: use pixelFormat 'rgb'")
    }
    let girar = orientacao == "left" || orientacao == "right"
    let (lw, lh) = girar ? (h, w) : (w, h)          // depois de girar
    let escala = min(1.0, Double(larguraDeteccao) / Double(lw))
    let fw = max(1, Int((Double(lw) * escala).rounded()))
    let fh = max(1, Int((Double(lh) * escala).rounded()))
    garantirCapacidade(max(w * h, lw * lh) * 4)

    var origem = vImage_Buffer(data: UnsafeMutableRawPointer(pixels.data), height: vImagePixelCount(h),
                               width: vImagePixelCount(w), rowBytes: bpr)
    var a = vImage_Buffer(data: bufA, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)

    // 1. canais em BGRA (o formato que o MediaPipe le no iOS)
    if rgba {
      let mapa: [UInt8] = [2, 1, 0, 3]
      vImagePermuteChannels_ARGB8888(&origem, &a, mapa, vImage_Flags(kvImageNoFlags))
    } else {
      vImageCopyBuffer(&origem, &a, 4, vImage_Flags(kvImageNoFlags))
    }

    // 2. de pe: 'right' = os pixels estao girados +90 graus em relacao ao desejado
    var atual = a
    if orientacao != "up" && !orientacao.isEmpty {
      var b = vImage_Buffer(data: bufB, height: vImagePixelCount(lh), width: vImagePixelCount(lw), rowBytes: lw * 4)
      let rot: UInt8
      switch orientacao {
      case "right": rot = UInt8(kRotate90DegreesCounterClockwise)
      case "left": rot = UInt8(kRotate90DegreesClockwise)
      default: rot = UInt8(kRotate180DegreesClockwise)
      }
      var fundo: [UInt8] = [0, 0, 0, 255]
      vImageRotate90_ARGB8888(&atual, &b, rot, &fundo, vImage_Flags(kvImageNoFlags))
      atual = b
    }

    // 3. espelhada, como o cv2.flip do app de computador e o canvas da web
    if espelhar {
      var destino = vImage_Buffer(data: atual.data == bufA ? bufB : bufA, height: atual.height,
                                  width: atual.width, rowBytes: atual.rowBytes)
      vImageHorizontalReflect_ARGB8888(&atual, &destino, vImage_Flags(kvImageNoFlags))
      atual = destino
    }

    // 4. no maximo 640 de largura, direto no CVPixelBuffer
    var saida: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
    guard CVPixelBufferCreate(kCFAllocatorDefault, fw, fh, kCVPixelFormatType_32BGRA, attrs, &saida) == kCVReturnSuccess,
      let pb = saida
    else {
      throw ErroLandmarker.imagem("nao criou o CVPixelBuffer \(fw)x\(fh)")
    }
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    var final = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(fh),
                              width: vImagePixelCount(fw), rowBytes: CVPixelBufferGetBytesPerRow(pb))
    if fw == Int(atual.width) && fh == Int(atual.height) {
      vImageCopyBuffer(&atual, &final, 4, vImage_Flags(kvImageNoFlags))
    } else {
      vImageScale_ARGB8888(&atual, &final, nil, vImage_Flags(kvImageHighQualityResampling))
    }
    return pb
  }

  // MARK: - deteccao

  private static func r4(_ v: Float) -> Double { (Double(v) * 10000).rounded() / 10000 }

  func detectar(
    pixels: ArrayBuffer, largura: Double, altura: Double, bytesPorLinha: Double, formato: String,
    orientacao: String, espelhar: Bool, comPose: Bool
  ) throws -> PontosQuadro {
    trava.lock()
    defer { trava.unlock() }
    let t0 = CFAbsoluteTimeGetCurrent()
    if maos == nil { maos = try criarMaos(numMaos) }
    if comPose && pose == nil { pose = try criarPose() }

    let pb = try prepararImagem(pixels, Int(largura), Int(altura), Int(bytesPorLinha), formato, orientacao, espelhar)
    let imagem = try MPImage(pixelBuffer: pb)
    let t1 = CFAbsoluteTimeGetCurrent()

    var pontos: [Double] = []
    var lados: [Double] = []
    if let r = try maos?.detect(image: imagem) {
      for (k, mao) in r.landmarks.prefix(2).enumerated() {
        for p in mao { pontos += [Self.r4(p.x), Self.r4(p.y), Self.r4(p.z)] }
        let nome = (k < r.handedness.count ? r.handedness[k].first?.categoryName : nil) ?? ""
        lados.append(nome.isEmpty ? -1 : (nome.lowercased().hasPrefix("r") ? 1 : 0))
      }
    }
    let t2 = CFAbsoluteTimeGetCurrent()

    var corpo: [Double] = []
    if comPose, let rp = try pose?.detect(image: imagem), let pts = rp.landmarks.first {
      for q in pts {
        corpo += [Self.r4(q.x), Self.r4(q.y), Self.r4(q.z), Self.r4(q.visibility?.floatValue ?? 1.0)]
      }
    }

    let t3 = CFAbsoluteTimeGetCurrent()

    return PontosQuadro(
      maos: pontos, lados: lados, pose: corpo,
      largura: Double(CVPixelBufferGetWidth(pb)), altura: Double(CVPixelBufferGetHeight(pb)),
      ms: (t3 - t0) * 1000, msImagem: (t1 - t0) * 1000, msMaos: (t2 - t1) * 1000,
      msCorpo: (t3 - t2) * 1000, processamento: "CPU")
  }
}
