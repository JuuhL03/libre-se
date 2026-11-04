import type { HybridObject } from 'react-native-nitro-modules'

/**
 * Pontos de um quadro, no mesmo formato que a versao web manda ao servidor
 * (libre-se/web/app.js, funcao detectar): coordenadas normalizadas (0..1) da
 * imagem ja de pe e espelhada, como o cv2.flip do app de computador.
 */
export interface PontosQuadro {
  /** 21 pontos x 3 (x, y, z) por mao detectada: 0, 63 ou 126 numeros. */
  maos: number[]
  /** Lado de cada mao detectada: 1 = direita, 0 = esquerda, -1 = sem rotulo. */
  lados: number[]
  /** 33 pontos x 4 (x, y, z, visibilidade) do corpo, ou vazio se nao pediu/nao achou. */
  pose: number[]
  /** Largura e altura da imagem analisada (depois de girar). */
  largura: number
  altura: number
  /** Tempo da deteccao, em milissegundos. */
  ms: number
}

export interface LibresLandmarker
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  /**
   * Carrega os modelos do MediaPipe (hand_landmarker.task e
   * pose_landmarker_lite.task, empacotados no app). Pode ser chamado de novo
   * para trocar o numero de maos (1 no alfabeto, 2 nas palavras).
   */
  configurar(numMaos: number): void

  /**
   * Detecta maos (e o corpo, se `comPose`) num quadro RGB da camera.
   * @param pixels    pixels intercalados de 4 bytes (frame.getPixelBuffer())
   * @param formato   'rgb-bgra-8-bit' ou 'rgb-rgba-8-bit' (frame.pixelFormat)
   * @param orientacao 'up' | 'right' | 'left' | 'down' (frame.orientation)
   * @param espelhar  espelha a imagem antes de detectar (vista de selfie)
   */
  detectar(
    pixels: ArrayBuffer,
    largura: number,
    altura: number,
    bytesPorLinha: number,
    formato: string,
    orientacao: string,
    espelhar: boolean,
    comPose: boolean
  ): PontosQuadro
}
