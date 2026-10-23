"""
libres_realtime.py
------------------
Pecas de inferencia em tempo real compartilhadas pelos apps de camera.

- SignSegmenter: decide ONDE comeca e termina um sinal (palavra) no fluxo
  da camera. Substitui a antiga janela deslizante de 30 frames, que quase
  nunca coincidia com o sinal inteiro e nao batia com o que o modelo viu
  no treino (o sinal completo, do levantar ao abaixar da mao).
- WordRecognizer: classifica um sinal segmentado (modelo_rotulado_xgb.pkl).
- LetterRecognizer: classifica uma pose estatica (modelo_alfabeto.pkl).

Regra do segmentador (a mesma definicao de "sinal" usada no treino, onde
cada video e reduzido aos frames em que a mao aparece):
  * o sinal comeca quando a mao aparece;
  * termina quando a mao sai do quadro (abaixar a mao) por GAP_FRAMES
    frames seguidos, OU quando a mao, depois de se mexer, fica parada por
    STILL_FRAMES frames, OU quando passa de MAX_FRAMES.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from libres_features import (FEAT_DIM, FEATURE_VERSION, N_COORDS, N_LANDMARKS,
                             compute_enhanced_features, normalize_landmarks,
                             normalize_sequence)


class SignSegmenter:
    def __init__(self,
                 min_frames: int = 10,
                 max_frames: int = 120,
                 gap_frames: int = 5,
                 still_frames: int = 15,
                 still_speed: float = 0.04,
                 min_motion: float = 1.0,
                 rearm_speed: float = 0.12) -> None:
        """
        min_frames  : sinais mais curtos que isso sao descartados (ruido)
        max_frames  : forca o fim do sinal (evita buffer infinito)
        gap_frames  : frames seguidos sem mao para considerar "mao abaixada"
        still_frames: frames parados para encerrar o sinal com a mao no ar
        still_speed : velocidade do punho (em tamanhos de mao por frame)
                      abaixo da qual a mao conta como parada
        min_motion  : caminho minimo percorrido (tamanhos de mao) para que
                      a parada conte como fim de sinal
        rearm_speed : depois de encerrar por parada, so comeca um novo sinal
                      quando a mao voltar a se mexer acima desta velocidade
        """
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.gap_frames = gap_frames
        self.still_frames = still_frames
        self.still_speed = still_speed
        self.min_motion = min_motion
        self.rearm_speed = rearm_speed
        self.reset()

    def reset(self) -> None:
        self.buf: list[np.ndarray] = []
        self.gap = 0
        self.still = 0
        self.path = 0.0
        self.waiting = False      # True = sinal ja emitido, mao ainda no ar
        self._last: Optional[np.ndarray] = None

    @staticmethod
    def _speed(a: np.ndarray, b: np.ndarray) -> float:
        ra = a.reshape(N_LANDMARKS, N_COORDS)
        rb = b.reshape(N_LANDMARKS, N_COORDS)
        size = max(float(np.linalg.norm(rb[9, :2] - rb[0, :2])), 1e-6)
        return float(np.linalg.norm(rb[0, :2] - ra[0, :2])) / size

    @property
    def recording(self) -> bool:
        return bool(self.buf) and not self.waiting

    @property
    def n_frames(self) -> int:
        return len(self.buf)

    def _emit(self) -> Optional[np.ndarray]:
        seq = np.array(self.buf) if len(self.buf) >= self.min_frames else None
        self.buf = []
        self.still = 0
        self.path = 0.0
        return seq

    def update(self, landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Recebe os 63 valores crus do frame (ou None sem mao).

        Retorna a sequencia crua (T, 63) quando um sinal termina; senao None.
        """
        if landmarks is None:
            self.gap += 1
            if self.gap >= self.gap_frames:
                self.waiting = False
                self._last = None
                if self.buf:
                    return self._emit()
            return None

        lm = np.asarray(landmarks, dtype=np.float64).reshape(-1)
        self.gap = 0
        speed = self._speed(self._last, lm) if self._last is not None else 0.0
        self._last = lm

        if self.waiting:
            if speed < self.rearm_speed:
                return None
            self.waiting = False      # mao voltou a se mexer: novo sinal

        self.buf.append(lm)
        self.path += speed
        self.still = self.still + 1 if speed < self.still_speed else 0

        if len(self.buf) >= self.max_frames:
            self.waiting = True
            return self._emit()
        if self.still >= self.still_frames and self.path < self.min_motion:
            # mao parada no ar sem ter feito nada: descarta a espera e
            # mantem so o ultimo frame como inicio do proximo sinal
            self.buf = self.buf[-1:]
            self.path = 0.0
            self.still = 0
            return None
        if self.still >= self.still_frames and self.path >= self.min_motion:
            # corta a cauda parada: ela nao existe nos videos de treino
            self.buf = self.buf[:-self.still + 1] if self.still > 1 else self.buf
            self.waiting = True
            return self._emit()
        return None


class WordRecognizer:
    def __init__(self, model_path: Path) -> None:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        if data.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(
                f"{model_path.name} foi treinado com a versao antiga das features "
                "(sem movimento da mao). Rode `python train_rotulado.py` de novo.")
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.le = data["label_encoder"]
        self.classes = list(self.le.classes_)

    def predict(self, raw_seq: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        seq = normalize_sequence(raw_seq)
        feat = compute_enhanced_features(seq).reshape(1, -1)
        proba = self.model.predict_proba(self.scaler.transform(feat))[0]
        order = np.argsort(proba)[::-1][:top_k]
        return [(str(self.le.inverse_transform([i])[0]), float(proba[i])) for i in order]


class LetterRecognizer:
    def __init__(self, model_path: Path) -> None:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        feat = normalize_landmarks(np.asarray(landmarks, dtype=np.float64).reshape(1, FEAT_DIM))
        proba = self.model.predict_proba(self.scaler.transform(feat))[0]
        top = int(proba.argmax())
        return str(self.model.classes_[top]), float(proba[top])


def accept_word(top: list[tuple[str, float]],
                min_conf: float = 0.45, min_margin: float = 0.15) -> Optional[str]:
    """Aceita a palavra top-1 se tiver confianca e margem sobre a 2a."""
    if not top:
        return None
    p1 = top[0][1]
    p2 = top[1][1] if len(top) > 1 else 0.0
    return top[0][0] if p1 >= min_conf and (p1 - p2) >= min_margin else None
