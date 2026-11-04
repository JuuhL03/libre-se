import { NitroModules } from 'react-native-nitro-modules'
import type { LibresLandmarker, PontosQuadro } from './specs/LibresLandmarker.nitro'

export type { LibresLandmarker, PontosQuadro }

/** Cria o detector nativo (MediaPipe Tasks) de maos e corpo. */
export function criarLandmarker(): LibresLandmarker {
  return NitroModules.createHybridObject<LibresLandmarker>('LibresLandmarker')
}
