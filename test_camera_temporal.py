#!/usr/bin/env python3
"""
Teste em tempo real com camera - so PALAVRAS (XGBoost v2).

Faca o sinal e ABAIXE a mao (ou segure parado ~1 s): o sinal inteiro e
classificado de uma vez. Para alfabeto + palavras juntos use libras_camera.py.
"""

import sys
from pathlib import Path
from typing import Optional
import mediapipe as mp
import cv2
import numpy as np

from libres_features import FEAT_DIM
from libres_realtime import (LeitorCamera, SignSegmenter, WordRecognizer,
                             abrir_camera, accept_word)

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "modelo_rotulado_xgb.pkl"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
CONFIDENCE_THRESHOLD: float = 0.45
MARGIN_THRESHOLD: float = 0.15

print("=" * 50)
print("PREDICAO EM TEMPO REAL - MODELO OTIMIZADO")
print("=" * 50)

print("Carregando modelo...")
try:
    recognizer = WordRecognizer(MODEL_PATH)
except RuntimeError as e:
    sys.exit(str(e))

print(f"Classes: {recognizer.classes}")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(hand_options)


print("Inicializando camera...")
cap = abrir_camera()
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
         "(Teams, Discord, navegador, OBS) ou rode com LIBRES_CAMERA=1.")
leitor = LeitorCamera(cap)

segmenter = SignSegmenter()
top: list[tuple[str, float]] = []
prediction: str = ""

print("\n>>> FACA UM SINAL EM LIBRAS E ABAIXE A MAO <<<")
print("Pressione 'q' para sair")
print("-" * 40)

while True:
    ret, frame = leitor.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    lm = None
    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        lm = np.array([c for p in hand for c in (p.x, p.y, p.z)])
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, (0, 255, 0), -1)

    sinal = segmenter.update(lm if lm is not None and len(lm) == FEAT_DIM else None)
    if sinal is not None:
        top = recognizer.predict(sinal)
        ok = accept_word(top, CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD)
        prediction = ok.upper() if ok else "DESCONHECIDO"
        print(f"  {len(sinal)} frames -> " +
              ", ".join(f"{p} {c*100:.0f}%" for p, c in top))

    if segmenter.recording:
        cv2.putText(frame, f"gravando... {segmenter.n_frames}", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if prediction:
        cor = (0, 0, 255) if prediction == "DESCONHECIDO" else (0, 255, 0)
        conf = f" ({top[0][1]*100:.0f}%)" if prediction != "DESCONHECIDO" else ""
        cv2.putText(frame, f">>> {prediction}{conf}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, cor, 4)

    cv2.putText(frame, "Q = Sair", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Libras - Reconhecimento de Palavras', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Encerrado!")
