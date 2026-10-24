#!/usr/bin/env python3
"""
Libre-se - camera unica para ALFABETO e PALAVRAS.

Uso: python libras_camera.py

Teclas:
  M          = alterna modo ALFABETO <-> PALAVRAS
  ESPACO     = limpa o texto montado
  BACKSPACE  = apaga o ultimo item
  Q          = sair

Modo ALFABETO: segure a letra parada; ela entra no texto apos alguns
frames estaveis (mesmo modelo/logica do alphabet_camera.py).

Modo PALAVRAS: comece com a mao fora do quadro (ou abaixada), faca o sinal
e ABAIXE a mao. O sinal tambem termina se a mao ficar parada no ar por ~1 s
depois de se mexer. O app mostra as 3 palavras mais provaveis do ultimo
sinal, para facilitar os testes.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from libres_features import FEAT_DIM
from libres_realtime import (LeitorCamera, LetterRecognizer, SignSegmenter,
                             Soletrador, WordRecognizer, abrir_camera, accept_word)

PROJECT_DIR = Path(__file__).parent
WORD_MODEL = PROJECT_DIR / "modelo_rotulado_xgb.pkl"
LETTER_MODEL = PROJECT_DIR / "modelo_alfabeto.pkl"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"


# ---------------------------------------------------------------- modelos
letters = words = None
try:
    letters = LetterRecognizer(LETTER_MODEL)
    print(f"Alfabeto: {LETTER_MODEL.name} carregado")
except Exception as e:  # noqa: BLE001
    print(f"[aviso] alfabeto indisponivel: {e}")
try:
    words = WordRecognizer(WORD_MODEL)
    print(f"Palavras: {len(words.classes)} classes -> {', '.join(words.classes)}")
except Exception as e:  # noqa: BLE001
    print(f"[aviso] palavras indisponiveis: {e}")
if letters is None and words is None:
    sys.exit("Nenhum modelo carregado.")

mode = "ALFABETO" if letters is not None else "PALAVRAS"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))

# ---------------------------------------------------------------- camera
cap = abrir_camera()
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
         "(Teams, Discord, navegador, OBS) ou rode com LIBRES_CAMERA=1.")
leitor = LeitorCamera(cap)

segmenter = SignSegmenter()
texto: list[str] = []          # itens ja reconhecidos (letras e palavras)
letra_atual: list[str] = []    # letras sendo soletradas
soletrador = Soletrador(letters) if letters is not None else None
ultimo_top: list[tuple[str, float]] = []
ultimo_ok: str | None = None
t_prev = time.time()
fps = 0.0


def texto_completo() -> str:
    partes = texto + (["".join(letra_atual)] if letra_atual else [])
    return " ".join(partes)


def fechar_soletracao() -> None:
    if letra_atual:
        texto.append("".join(letra_atual))
        letra_atual.clear()


print("\nM = modo | ESPACO = limpar | BACKSPACE = apagar | Q = sair\n")

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
        lm = np.array([c for p in hand for c in (p.x, p.y, p.z)], dtype=np.float64)
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

    # ------------------------------------------------------- ALFABETO
    if mode == "ALFABETO":
        info = soletrador.update(lm if lm is not None and len(lm) == FEAT_DIM else None)
        if info["acao"] == "troca" and letra_atual:
            letra_atual[-1] = info["letra"]
        elif info["acao"] == "add":
            letra_atual.append(info["letra"])
        if info["letra"]:
            # palpite sempre visivel: verde = aceito, cinza = confianca baixa
            cor = (0, 255, 0) if info["aceita"] else (170, 170, 170)
            marca = "" if info["aceita"] else "?"
            mov = "  (movimento)" if info["movendo"] else ""
            cv2.putText(frame, f"{info['letra']}{marca} ({info['conf']*100:.0f}%){mov}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)
            cv2.putText(frame, "  ".join(f"{l} {c*100:.0f}%" for l, c in info["top"]),
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # ------------------------------------------------------- PALAVRAS
    else:
        sinal = segmenter.update(lm)
        if sinal is not None:
            ultimo_top = words.predict(sinal)
            ultimo_ok = accept_word(ultimo_top)
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in ultimo_top)
            print(f"  sinal de {len(sinal)} frames -> {resumo}"
                  f"{'' if ultimo_ok else '  (incerto)'}")
            if ultimo_ok:
                fechar_soletracao()
                texto.append(ultimo_ok.upper())

        if segmenter.recording:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (0, 0, 255), -1)
            cv2.putText(frame, f"gravando {segmenter.n_frames}",
                        (frame.shape[1] - 260, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for k, (pal, conf) in enumerate(ultimo_top):
            cor = (0, 255, 0) if (k == 0 and ultimo_ok) else (200, 200, 200)
            cv2.putText(frame, f"{k+1}. {pal} {conf*100:.0f}%", (20, 140 + 32 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    # ------------------------------------------------------- HUD
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now
    cv2.putText(frame, f"[{mode}]  {fps:4.1f} fps   M=modo Q=sair", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, texto_completo()[-40:], (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("Libre-se", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("m"):
        alvo = "PALAVRAS" if mode == "ALFABETO" else "ALFABETO"
        if (alvo == "PALAVRAS" and words is None) or (alvo == "ALFABETO" and letters is None):
            print(f"  modo {alvo} indisponivel (modelo nao carregado)")
        else:
            mode = alvo
            fechar_soletracao()
            segmenter.reset()
            if soletrador is not None:
                soletrador.reset()
            print(f"  modo: {mode}")
    elif key == ord(" "):
        texto.clear()
        letra_atual.clear()
        ultimo_top, ultimo_ok = [], None
    elif key == 8:
        if letra_atual:
            letra_atual.pop()
        elif texto:
            texto.pop()

cap.release()
cv2.destroyAllWindows()
print(f"Texto final: {texto_completo() or '(vazio)'}")
