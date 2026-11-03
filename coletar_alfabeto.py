#!/usr/bin/env python3
"""
Coleta do ALFABETO pela webcam (poses paradas), para treinar o modelo com
maos reais do grupo. Cada pessoa usa sempre o mesmo numero.

Uso:
  python coletar_alfabeto.py --pessoa 1
  python coletar_alfabeto.py --pessoa 2 --takes 3
  python coletar_alfabeto.py --pessoa 1 --letras A,B,C

Como funciona: a tela mostra a letra. Faca a letra com a mao parada e aperte
ESPACO. Depois de uma contagem curta, grava ~1,5 s (45 quadros) e passa para o
proximo take/letra sozinho. Mexa um pouco a mao entre os takes (distancia,
angulo), isso ajuda o modelo a generalizar. J e Z tem movimento: durante a
gravacao, repita o movimento da letra sem parar (~2 s).

Teclas:  ESPACO = gravar  |  N / P = proxima / anterior  |  D = apaga o ultimo take
         Q = sair
Depois:  python train_alfabeto.py
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from libres_features import FEAT_DIM
from libres_realtime import LeitorCamera, abrir_camera

PROJECT_DIR = Path(__file__).parent
COLETA_DIR = PROJECT_DIR / "coleta_alfabeto"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"

LETRAS_PADRAO = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
QUADROS_POR_TAKE = 45
QUADROS_DINAMICA = 60          # J e Z: grava o movimento inteiro
DINAMICAS = {"J", "Z"}
CONTAGEM_S = 1.0

ap = argparse.ArgumentParser(description="Coleta do alfabeto em Libras")
ap.add_argument("--pessoa", type=int, required=True, help="numero fixo de quem esta sinalizando")
ap.add_argument("--letras", type=str, default="".join(LETRAS_PADRAO),
                help="letras a coletar, ex.: A,B,C (padrao: A a Z)")
ap.add_argument("--takes", type=int, default=3, help="takes por letra (padrao 3)")
ap.add_argument("--camera", type=int, default=None, metavar="N",
                help="numero da camera (0, 1, 2...); sem isso, escolhe sozinho e pula "
                     "cameras de imagem preta ou parada (OBS...)")
args = ap.parse_args()

letras = [c.strip().upper() for c in args.letras.replace(",", "") if c.strip()]
prefixo = f"Pessoa{args.pessoa:02d}"


def takes(letra: str) -> list[Path]:
    return sorted((COLETA_DIR / letra).glob(f"{prefixo}_{letra}_*.npy"))


def proximo_arquivo(letra: str) -> Path:
    pasta = COLETA_DIR / letra
    pasta.mkdir(parents=True, exist_ok=True)
    n = 0
    while (pasta / f"{prefixo}_{letra}_{n:03d}.npy").exists():
        n += 1
    return pasta / f"{prefixo}_{letra}_{n:03d}.npy"


landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
        num_hands=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

cap = abrir_camera(indice=args.camera)
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
             "ou escolha outra: --camera 1")
leitor = LeitorCamera(cap)

idx = 0
estado = "parado"          # parado | contagem | gravando
t_contagem = 0.0
buffer: list[np.ndarray] = []
ultimo: Path | None = None
aviso, aviso_ate = "", 0.0

print(f"\n{prefixo} | letras: {''.join(letras)} | {args.takes} takes por letra")
print("Faca a letra e aperte ESPACO. N/P = trocar letra, D = apagar ultimo, Q = sair\n")

while True:
    ok, frame = leitor.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    lm = None
    if res and res.hand_landmarks:
        hand = res.hand_landmarks[0]
        lm = np.array([c for p in hand for c in (p.x, p.y, p.z)], dtype=np.float32)
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

    letra = letras[idx]
    n = len(takes(letra))

    if estado == "contagem" and time.time() - t_contagem >= CONTAGEM_S:
        estado, buffer = "gravando", []
    if estado == "gravando":
        if lm is not None and len(lm) == FEAT_DIM:
            buffer.append(lm)
        if len(buffer) >= (QUADROS_DINAMICA if letra in DINAMICAS else QUADROS_POR_TAKE):
            ultimo = proximo_arquivo(letra)
            np.save(ultimo, np.array(buffer))
            n = len(takes(letra))
            print(f"  salvo {ultimo.name} -> {n}/{args.takes}")
            aviso, aviso_ate = f"SALVO! {n}/{args.takes}", time.time() + 1.0
            estado = "parado"
            if n >= args.takes and idx < len(letras) - 1:
                idx += 1

    # ---------------- HUD
    cv2.putText(frame, f"{prefixo}  [{idx+1}/{len(letras)}]", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, letras[idx], (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 8)
    n = len(takes(letras[idx]))
    cv2.putText(frame, f"takes: {n}/{args.takes}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if n >= args.takes else (255, 255, 255), 2)
    if estado == "contagem":
        cv2.putText(frame, "prepare...", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 3)
    elif estado == "gravando":
        cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (0, 0, 255), -1)
        meta = QUADROS_DINAMICA if letras[idx] in DINAMICAS else QUADROS_POR_TAKE
        cv2.putText(frame, f"gravando {len(buffer)}/{meta}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if letras[idx] in DINAMICAS:
            cv2.putText(frame, "repita o MOVIMENTO da letra sem parar", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif lm is None:
        cv2.putText(frame, "mao nao detectada", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "ESPACO = gravar", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if time.time() < aviso_ate:
        cv2.putText(frame, aviso, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(frame, "ESPACO gravar  N/P letra  D apagar  Q sair", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Libre-se - Coleta do alfabeto", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" ") and estado == "parado":
        estado, t_contagem = "contagem", time.time()
    elif key == ord("n"):
        idx, estado = min(idx + 1, len(letras) - 1), "parado"
    elif key == ord("p"):
        idx, estado = max(idx - 1, 0), "parado"
    elif key == ord("d") and ultimo is not None and ultimo.exists():
        ultimo.unlink()
        print(f"  apagado {ultimo.name}")
        aviso, aviso_ate, ultimo = "ultimo take apagado", time.time() + 1.0, None

cap.release()
cv2.destroyAllWindows()
faltam = [l for l in letras if len(takes(l)) < args.takes]
print(f"\nColeta de {prefixo}: {len(letras) - len(faltam)}/{len(letras)} letras completas.")
if faltam:
    print("Faltam:", " ".join(faltam))
print("Proximo passo:  python train_alfabeto.py")
