#!/usr/bin/env python3
"""
Coleta de PALAVRAS pela webcam (dataset proprio do grupo).

Cada repeticao e cortada sozinha, com o MESMO segmentador do app de camera:
comece com a mao fora do quadro, faca o sinal e ABAIXE a mao -> o take e
salvo em coletados/<palavra>/SinalizadorNN_<palavra>_<n>.npy (landmarks
crus, sinal inteiro). O numero do sinalizador permite a avaliacao por
pessoa (eval_loso.py): o modelo e testado em alguem que ele nunca viu.

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
from libres_realtime import LeitorCamera, SignSegmenter, abrir_camera

PROJECT_DIR = Path(__file__).parent
COLETADOS_DIR = PROJECT_DIR / "coletados"
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
args = ap.parse_args()

palavras = [p.strip().lower().replace(" ", "_") for p in args.palavras.split(",") if p.strip()]
prefixo = f"Sinalizador{args.sinalizador:02d}"


def takes_salvos(palavra: str) -> list[Path]:
    return sorted((COLETADOS_DIR / palavra).glob(f"{prefixo}_{palavra}_*.npy"))


def proximo_nome(palavra: str) -> Path:
    pasta = COLETADOS_DIR / palavra
    pasta.mkdir(parents=True, exist_ok=True)
    n = 0
    while (pasta / f"{prefixo}_{palavra}_{n:03d}.npy").exists():
        n += 1
    return pasta / f"{prefixo}_{palavra}_{n:03d}.npy"


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

cap = abrir_camera()
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
         "(Teams, Discord, navegador, OBS) ou rode com LIBRES_CAMERA=1.")
leitor = LeitorCamera(cap)

segmenter = SignSegmenter()
idx = 0
ultimo_salvo: Path | None = None
aviso, aviso_ate = "", 0.0

print(f"\n{prefixo} | palavras: {', '.join(palavras)} | meta: {args.takes} takes cada")
print("Faca o sinal e ABAIXE a mao. N/P = trocar palavra, D = apagar ultimo, Q = sair\n")

while True:
    ret, frame = leitor.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))

    lm = None
    if res and res.hand_landmarks:
        hand = res.hand_landmarks[0]
        lm = np.array([c for p in hand for c in (p.x, p.y, p.z)], dtype=np.float32)
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

    palavra = palavras[idx]
    sinal = segmenter.update(lm if lm is not None and len(lm) == FEAT_DIM else None)
    if sinal is not None:
        ultimo_salvo = proximo_nome(palavra)
        np.save(ultimo_salvo, sinal.astype(np.float32))
        n = len(takes_salvos(palavra))
        print(f"  salvo {ultimo_salvo.name} ({len(sinal)} frames) -> {n}/{args.takes}")
        aviso, aviso_ate = f"SALVO! {n}/{args.takes}", time.time() + 1.0
        if n >= args.takes and idx < len(palavras) - 1:
            idx += 1
            segmenter.reset()
            print(f"  meta atingida -> proxima palavra: {palavras[idx]}")

    n = len(takes_salvos(palavra))
    cor_meta = (0, 255, 0) if n >= args.takes else (255, 255, 255)
    cv2.putText(frame, f"{prefixo}  [{idx+1}/{len(palavras)}]", (20, 35),
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
    elif key == ord("d") and ultimo_salvo is not None and ultimo_salvo.exists():
        ultimo_salvo.unlink()
        print(f"  apagado {ultimo_salvo.name}")
        aviso, aviso_ate = "ultimo take apagado", time.time() + 1.0
        ultimo_salvo = None

cap.release()
cv2.destroyAllWindows()
print("\nResumo:")
for p in palavras:
    print(f"  {p:12s}: {len(takes_salvos(p))} takes")
