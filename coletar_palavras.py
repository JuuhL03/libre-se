#!/usr/bin/env python3
"""
Coleta de PALAVRAS pela webcam (dataset proprio do grupo).

Cada repeticao e cortada sozinha, com o MESMO segmentador do app de camera:
comece com a mao fora do quadro, faca o sinal e ABAIXE a mao. Cada take e
salvo em dois formatos:
  coleta_v3/<palavra>/PessoaNN_<palavra>_<n>.npz      duas maos + corpo (treino_v3.py)
  coletados/<palavra>/SinalizadorNN_<palavra>_<n>.npy  so a mao principal (train_rotulado.py)
O numero da pessoa permite a avaliacao por pessoa (avaliar_v3.py / eval_loso.py):
o modelo e testado em alguem que ele nunca viu. Na v3 o nome usa "Pessoa",
para a pessoa do grupo nao se misturar com os sinalizadores do MINDS.
Sem o modelo de pose do MediaPipe (ou com --so-uma-mao), grava so o formato antigo.

Uso:
  python coletar_palavras.py --sinalizador 1
  python coletar_palavras.py --sinalizador 2 --palavras oi,obrigado,tchau
  python coletar_palavras.py --sinalizador 1 --takes 15

Teclas:
  N / P      = proxima / palavra anterior
  D          = apaga o ultimo take gravado (se errou o sinal)
  Q          = sair
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from libres_features import FEAT_DIM
from libres_realtime import LeitorCamera, SignSegmenter, SignSegmenterV3, abrir_camera

PROJECT_DIR = Path(__file__).parent
COLETADOS_DIR = PROJECT_DIR / "coletados"
COLETA_V3_DIR = PROJECT_DIR / "coleta_v3"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"

PALAVRAS_PADRAO = [
    "oi", "tchau", "obrigado", "por_favor", "desculpa",
    "sim", "nao", "ajuda", "banheiro", "agua",
]

ap = argparse.ArgumentParser(description="Coleta de palavras em Libras")
ap.add_argument("--sinalizador", type=int, required=True,
                help="numero de quem esta sinalizando (1, 2, 3...). "
                     "Cada pessoa do grupo usa sempre o mesmo numero.")
ap.add_argument("--palavras", type=str, default=",".join(PALAVRAS_PADRAO),
                help="lista separada por virgula (sem acento, use _ no lugar de espaco)")
ap.add_argument("--takes", type=int, default=10, help="repeticoes por palavra (meta)")
ap.add_argument("--so-uma-mao", action="store_true",
                help="grava so o formato antigo (uma mao, sem o corpo)")
ap.add_argument("--camera", type=int, default=None, metavar="N",
                help="numero da camera (0, 1, 2...); sem isso, escolhe sozinho e pula "
                     "cameras de imagem preta ou parada (OBS...)")
args = ap.parse_args()

palavras = [p.strip().lower().replace(" ", "_") for p in args.palavras.split(",") if p.strip()]
prefixo = f"Sinalizador{args.sinalizador:02d}"
prefixo_v3 = f"Pessoa{args.sinalizador:02d}"

# detectores: duas maos + pose (v3), os mesmos do app; se nao der, uma mao (v2)
det_v3 = None
if not args.so_uma_mao:
    try:
        from extrair_minds_v3 import DetectorV3
        det_v3 = DetectorV3(HAND_MODEL)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        print(f"[aviso] sem duas maos + corpo ({e}); gravando so a mao principal")


def takes_salvos(palavra: str) -> list[Path]:
    if det_v3 is not None:
        return sorted((COLETA_V3_DIR / palavra).glob(f"{prefixo_v3}_{palavra}_*.npz"))
    return sorted((COLETADOS_DIR / palavra).glob(f"{prefixo}_{palavra}_*.npy"))


def proximo_nome(palavra: str) -> tuple[Path, Path]:
    """(arquivo v2 .npy, arquivo v3 .npz) com o mesmo numero de take."""
    pasta, pasta3 = COLETADOS_DIR / palavra, COLETA_V3_DIR / palavra
    pasta.mkdir(parents=True, exist_ok=True)
    if det_v3 is not None:
        pasta3.mkdir(parents=True, exist_ok=True)
    n = 0
    while ((pasta / f"{prefixo}_{palavra}_{n:03d}.npy").exists()
           or (pasta3 / f"{prefixo_v3}_{palavra}_{n:03d}.npz").exists()):
        n += 1
    return pasta / f"{prefixo}_{palavra}_{n:03d}.npy", pasta3 / f"{prefixo_v3}_{palavra}_{n:03d}.npz"


landmarker = None
if det_v3 is None:
    BaseOptions = mp.tasks.BaseOptions
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
            num_hands=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))

cap = abrir_camera(indice=args.camera)
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
             "(Teams, Discord, navegador, OBS) ou escolha outra: --camera 1")
leitor = LeitorCamera(cap)

segmenter = SignSegmenterV3() if det_v3 is not None else SignSegmenter()
idx = 0
ultimos_salvos: list[Path] = []
aviso, aviso_ate = "", 0.0
tempos: list[float] = []            # para guardar o fps do take


def salvar(palavra: str, sinal, aspecto: float) -> tuple[Path, int] | None:
    """Grava o take nos dois formatos. Retorna (arquivo principal, quadros)."""
    global ultimos_salvos
    f2, f3 = proximo_nome(palavra)
    if det_v3 is None:
        np.save(f2, sinal.astype(np.float32))
        ultimos_salvos = [f2]
        return f2, len(sinal)
    from libres_v3 import mao_dominante_crua
    n = len(sinal["maos"])
    dt = np.diff(tempos[-n:]) if len(tempos) >= 2 else np.array([1 / 30])
    fps = float(1 / np.median(dt)) if len(dt) and np.median(dt) > 0 else 30.0
    np.savez_compressed(f3, maos=sinal["maos"].astype(np.float32), lado=sinal["lado"].astype(np.int8),
                        conf_lado=np.zeros((n, 2), np.float32), pose=sinal["pose"].astype(np.float32),
                        aspecto=np.float32(aspecto), fps=np.float32(fps))
    ultimos_salvos = [f3]
    cru = mao_dominante_crua(sinal["maos"], sinal["lado"], aspecto)
    if cru is not None:                     # formato antigo: so a mao que mais se move
        np.save(f2, cru.astype(np.float32))
        ultimos_salvos.append(f2)
    return f3, n


print(f"\n{prefixo_v3 if det_v3 is not None else prefixo} | palavras: {', '.join(palavras)} "
      f"| meta: {args.takes} takes cada | {'duas maos + corpo' if det_v3 is not None else 'uma mao'}")
print("Faca o sinal e ABAIXE a mao. N/P = trocar palavra, D = apagar ultimo, Q = sair\n")

while True:
    ret, frame = leitor.read()
    if not ret:
        break
    t = time.time()
    tempos.append(t)
    del tempos[:-600]
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    palavra = palavras[idx]

    if det_v3 is not None:
        maos, lado, pose = det_v3.detectar(rgb, t)
        presentes = [k for k in range(2) if not np.isnan(maos[k, 0, 0])]
        for k in presentes:
            cor = (0, 255, 0) if k == presentes[0] else (255, 200, 0)
            for x, y_, _ in maos[k]:
                cv2.circle(frame, (int(x * w), int(y_ * h)), 4, cor, -1)
        quadro = {"maos": maos, "lado": lado, "pose": pose} if presentes else None
        sinal = segmenter.update(quadro, t)
    else:
        res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))
        lm = None
        if res and res.hand_landmarks:
            hand = res.hand_landmarks[0]
            lm = np.array([c for p in hand for c in (p.x, p.y, p.z)], dtype=np.float32)
            for p in hand:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
        sinal = segmenter.update(lm if lm is not None and len(lm) == FEAT_DIM else None, t)

    if sinal is not None:
        principal, nq = salvar(palavra, sinal, w / h)
        n = len(takes_salvos(palavra))
        print(f"  salvo {principal.name} ({nq} frames) -> {n}/{args.takes}")
        aviso, aviso_ate = f"SALVO! {n}/{args.takes}", time.time() + 1.0
        if n >= args.takes and idx < len(palavras) - 1:
            idx += 1
            segmenter.reset()
            print(f"  meta atingida -> proxima palavra: {palavras[idx]}")

    n = len(takes_salvos(palavra))
    cor_meta = (0, 255, 0) if n >= args.takes else (255, 255, 255)
    cv2.putText(frame, f"{prefixo_v3 if det_v3 is not None else prefixo}  [{idx+1}/{len(palavras)}]", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, palavra.upper(), (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.putText(frame, f"takes: {n}/{args.takes}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_meta, 2)
    if segmenter.recording:
        cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (0, 0, 255), -1)
        cv2.putText(frame, f"gravando {segmenter.n_frames}", (frame.shape[1] - 260, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if time.time() < aviso_ate:
        cv2.putText(frame, aviso, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(frame, "N/P = palavra  D = apagar ultimo  Q = sair",
                (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Libre-se - Coleta de palavras", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        idx = min(idx + 1, len(palavras) - 1)
        segmenter.reset()
    elif key == ord("p"):
        idx = max(idx - 1, 0)
        segmenter.reset()
    elif key == ord("d") and any(f.exists() for f in ultimos_salvos):
        for f in ultimos_salvos:
            if f.exists():
                f.unlink()
        print(f"  apagado {ultimos_salvos[0].name}")
        aviso, aviso_ate = "ultimo take apagado", time.time() + 1.0
        ultimos_salvos = []

cap.release()
cv2.destroyAllWindows()
print("\nResumo:")
for p in palavras:
    print(f"  {p:12s}: {len(takes_salvos(p))} takes")
