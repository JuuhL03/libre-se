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
                             alphabet_features, mirror_raw,
                             compute_enhanced_features, normalize_landmarks,
                             normalize_sequence)


class SignSegmenter:
    """Segmenta sinais no fluxo da camera usando TEMPO (segundos), nao numero de
    quadros: funciona igual a 10, 15 ou 30 fps. Velocidades sao medidas em
    "tamanhos de mao por segundo" (punho -> base do dedo medio)."""

    def __init__(self,
                 min_duration: float = 0.45,
                 min_frames: int = 6,
                 max_duration: float = 8.0,
                 gap_time: float = 0.33,
                 still_time: float = 1.0,
                 still_speed: float = 0.6,
                 min_motion: float = 1.0,
                 rearm_speed: float = 1.8) -> None:
        """
        min_duration: sinais mais curtos (s) sao descartados como ruido
        min_frames  : minimo de quadros para classificar
        max_duration: forca o fim do sinal (s)
        gap_time    : tempo sem mao (s) para considerar "mao abaixada"
        still_time  : tempo parado (s) para encerrar o sinal com a mao no ar
        still_speed : velocidade do punho (maos/s) abaixo da qual conta como parada
        min_motion  : caminho minimo (maos) para a parada contar como fim de sinal
        rearm_speed : depois de encerrar por parada, so comeca outro sinal quando a
                      mao voltar a se mexer acima desta velocidade (maos/s)
        """
        self.min_duration = min_duration
        self.min_frames = min_frames
        self.max_duration = max_duration
        self.gap_time = gap_time
        self.still_time = still_time
        self.still_speed = still_speed
        self.min_motion = min_motion
        self.rearm_speed = rearm_speed
        self.reset()

    def reset(self) -> None:
        self.buf: list[np.ndarray] = []
        self.tbuf: list[float] = []
        self.path = 0.0
        self.still_since: Optional[float] = None
        self.waiting = False      # True = sinal ja emitido, mao ainda no ar
        self._last: Optional[np.ndarray] = None
        self._t_last: Optional[float] = None
        self._t_mao: Optional[float] = None

    @staticmethod
    def _desloc(a: np.ndarray, b: np.ndarray) -> float:
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
        dur = (self.tbuf[-1] - self.tbuf[0]) if self.tbuf else 0.0
        ok = len(self.buf) >= self.min_frames and dur >= self.min_duration
        seq = np.array(self.buf) if ok else None
        self.buf, self.tbuf = [], []
        self.path = 0.0
        self.still_since = None
        return seq

    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> Optional[np.ndarray]:
        """Recebe os 63 valores crus do quadro (ou None sem mao) e o instante t
        em segundos (padrao: relogio do sistema).

        Retorna a sequencia crua (T, 63) quando um sinal termina; senao None.
        """
        import time as _time
        t = _time.time() if t is None else float(t)

        if landmarks is None:
            if self._t_mao is not None and t - self._t_mao >= self.gap_time:
                self.waiting = False
                self._last = None
                self._t_mao = None
                if self.buf:
                    return self._emit()
            return None

        lm = np.asarray(landmarks, dtype=np.float64).reshape(-1)
        self._t_mao = t
        if self._last is not None and self._t_last is not None:
            dt = min(max(t - self._t_last, 1 / 120), 0.5)
            desloc = self._desloc(self._last, lm)
            vel = desloc / dt
        else:
            desloc, vel = 0.0, 0.0
        self._last, self._t_last = lm, t

        if self.waiting:
            if vel < self.rearm_speed:
                return None
            self.waiting = False      # mao voltou a se mexer: novo sinal

        self.buf.append(lm)
        self.tbuf.append(t)
        self.path += desloc
        if vel < self.still_speed:
            if self.still_since is None:
                self.still_since = t
        else:
            self.still_since = None
        parado = self.still_since is not None and t - self.still_since >= self.still_time

        if t - self.tbuf[0] >= self.max_duration:
            self.waiting = True
            return self._emit()
        if parado and self.path < self.min_motion:
            # mao parada no ar sem ter feito nada: mantem so o ultimo quadro
            self.buf, self.tbuf = self.buf[-1:], self.tbuf[-1:]
            self.path = 0.0
            self.still_since = t
            return None
        if parado:
            # corta a cauda parada: ela nao existe nos videos de treino
            corte = next(i for i, ti in enumerate(self.tbuf) if ti >= self.still_since)
            self.buf, self.tbuf = self.buf[:max(corte, 1)], self.tbuf[:max(corte, 1)]
            self.waiting = True
            return self._emit()
        return None


class MovimentoMao:
    """Velocidade recente das pontas do indicador e do minimo (maos/s), media
    numa janela curta. Usada para distinguir letras com movimento (J, Z)."""

    def __init__(self, janela: float = 0.4) -> None:
        self.janela = janela
        self.hist: list[tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.hist = []

    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> float:
        import time as _time
        t = _time.time() if t is None else float(t)
        if landmarks is None:
            self.hist = []
            return 0.0
        r = np.asarray(landmarks, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
        size = max(float(np.linalg.norm(r[9, :2] - r[0, :2])), 1e-6)
        self.hist.append((t, r[[8, 20], :2] / size))
        self.hist = [(ti, p) for ti, p in self.hist if t - ti <= self.janela]
        if len(self.hist) < 3:
            return 0.0
        caminho = sum(float(np.linalg.norm(b[1] - a[1], axis=1).max())
                      for a, b in zip(self.hist, self.hist[1:]))
        dur = self.hist[-1][0] - self.hist[0][0]
        return caminho / dur if dur > 0 else 0.0


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
        self.versao = data.get("feature_version", 1)   # 1 = modelo antigo (so coordenadas)
        # inferencia de 1 amostra por quadro: paralelizar so atrasa (~50 ms -> ~12 ms)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.classes = [str(c) for c in self.model.classes_]

    def _features(self, raw: np.ndarray) -> np.ndarray:
        if self.versao >= 2:
            return alphabet_features(raw)
        return normalize_landmarks(raw)

    def predict_top(self, landmarks: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """As k letras mais provaveis. Avalia a mao como esta e espelhada e fica
        com a versao mais confiante: os videos de treino nao sao espelhados e a
        camera e, e canhotos fazem as letras espelhadas."""
        raw = np.asarray(landmarks, dtype=np.float64).reshape(1, FEAT_DIM)
        duas = np.vstack([raw, mirror_raw(raw)])
        probas = self.model.predict_proba(self.scaler.transform(self._features(duas)))
        proba = probas[int(probas.max(axis=1).argmax())]
        ordem = np.argsort(proba)[::-1][:k]
        return [(str(self.model.classes_[i]), float(proba[i])) for i in ordem]

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        return self.predict_top(landmarks, 1)[0]


LETRAS_DINAMICAS = {"J", "Z"}   # letras com movimento no alfabeto manual


class Soletrador:
    """Decide, quadro a quadro, quando uma letra entra no texto.

    - mao parada: vale a melhor letra ESTATICA (todas menos J e Z);
    - mao em movimento: se J ou Z estiver entre os 2 palpites, vale ele
      (o J tem a mesma forma de mao do I: o que o distingue e o movimento);
    - a letra precisa ficar estavel por um tempo minimo (segundos);
    - letra repetida (OSSO) exige tirar a mao do quadro entre as duas;
    - se uma letra acabou de entrar e em seguida vem o movimento de J/Z sem a
      mao sair, ela e trocada (o "I" que vira "J").
    """

    def __init__(self, rec: "LetterRecognizer", conf_min: float = 0.55, conf_din: float = 0.25,
                 tempo_estavel: float = 0.3, tempo_din: float = 0.15,
                 mov_limiar: float = 1.0, janela_troca: float = 1.0) -> None:
        self.rec = rec
        self.conf_min, self.conf_din = conf_min, conf_din
        self.tempo_estavel, self.tempo_din = tempo_estavel, tempo_din
        self.mov_limiar = mov_limiar
        self.janela_troca = janela_troca
        self.tem_dinamicas = bool(LETRAS_DINAMICAS & set(rec.classes))
        self.mov = MovimentoMao()
        self.ultimo = ""
        self.reset()

    def reset(self) -> None:
        self.estavel, self.desde, self.emitido = "", 0.0, False
        self.mao_saiu = True
        self.t_ultima = -1e9
        self.residuo = None          # letra estatica que sobra no fim de um J/Z
        self.mov.reset()

    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> dict:
        import time as _time
        t = _time.time() if t is None else float(t)
        info = {"letra": "", "conf": 0.0, "top": [], "aceita": False, "acao": None,
                "movendo": False}
        if landmarks is None:
            self.estavel, self.emitido = "", False
            self.mao_saiu = True
            self.residuo = None
            self.mov.update(None, t)
            return info

        top = self.rec.predict_top(landmarks, len(self.rec.classes))
        movimento = self.mov.update(landmarks, t)
        movendo = movimento >= self.mov_limiar
        din = [(l, c) for l, c in top[:2] if l in LETRAS_DINAMICAS]
        if movendo and din and din[0][1] >= self.conf_din:
            letra, conf = din[0]
            dinamica = True
        else:
            est = [(l, c) for l, c in top if l not in LETRAS_DINAMICAS] or top
            letra, conf = est[0]
            dinamica = False
        ok = conf >= (self.conf_din if dinamica else self.conf_min)
        info.update(letra=letra, conf=conf, top=top[:3], aceita=ok, movendo=movendo)

        if not ok:
            self.estavel, self.emitido = "", False
            return info
        if letra != self.estavel:
            self.estavel, self.desde, self.emitido = letra, t, False
        espera = self.tempo_din if dinamica else self.tempo_estavel
        if self.emitido or t - self.desde < espera:
            return info

        self.emitido = True
        if not dinamica and self.residuo is not None:
            if letra == self.residuo:
                return info          # final parado de um J/Z: nao conta
            self.residuo = None
        base = None
        if dinamica:   # letra estatica com a mesma forma de mao (ex.: I para o J)
            est = [(l, c) for l, c in top if l not in LETRAS_DINAMICAS]
            base = est[0][0] if est else None
        if dinamica and not self.mao_saiu and t - self.t_ultima <= self.janela_troca \
                and self.ultimo == base:
            info["acao"] = "troca"    # o "I" que acabou de entrar era o comeco do "J"
        elif letra != self.ultimo or self.mao_saiu:
            info["acao"] = "add"
        else:
            return info
        if dinamica:
            self.residuo = base
        self.ultimo, self.t_ultima, self.mao_saiu = letra, t, False
        return info


def accept_word(top: list[tuple[str, float]],
                min_conf: float = 0.45, min_margin: float = 0.15) -> Optional[str]:
    """Aceita a palavra top-1 se tiver confianca e margem sobre a 2a."""
    if not top:
        return None
    p1 = top[0][1]
    p2 = top[1][1] if len(top) > 1 else 0.0
    return top[0][0] if p1 >= min_conf and (p1 - p2) >= min_margin else None


def abrir_camera(largura: int = 1280, altura: int = 720):
    """Abre a webcam testando indices e backends ate conseguir uma IMAGEM de verdade.

    - No Windows o backend padrao (MSMF) as vezes abre mas nao le ("can't grab
      frame", erro -1072875772 = camera em uso por outro programa); o DirectShow
      costuma funcionar.
    - Cameras virtuais (NVIDIA Broadcast, OBS, Snap Camera...) podem abrir e
      entregar so imagem PRETA quando o programa delas nao esta transmitindo:
      essas sao puladas e so usadas como ultimo recurso.
    Defina LIBRES_CAMERA=N para forcar o indice da camera.
    """
    import os
    import sys
    import time
    import cv2

    if sys.platform.startswith("win"):
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "padrao")]
    else:
        backends = [(cv2.CAP_ANY, "padrao")]
    forcado = os.environ.get("LIBRES_CAMERA")
    indices = [int(forcado)] if forcado is not None else [0, 1, 2, 3]

    def tentar(idx, backend):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            return None, "nao abriu"
        # muitas webcams so entregam 10 fps em 1280x720 no formato padrao (YUY2);
        # em MJPG chegam a 30 fps
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        cap.set(cv2.CAP_PROP_FPS, 30)
        leu = False
        t0 = time.time()
        while time.time() - t0 < 2.5:          # tempo para a exposicao automatica
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                leu = True
                if float(frame.mean()) > 8.0:  # tem imagem de verdade
                    return cap, "ok"
            else:
                time.sleep(0.03)
        cap.release()
        return None, ("so imagem preta" if leu else "nao entregou imagem")

    reserva = None
    relatorio = []
    for idx in indices:
        for backend, nome in backends:
            cap, motivo = tentar(idx, backend)
            if cap is not None:
                h, w = cap.read()[1].shape[:2]
                print(f"Camera {idx} aberta via {nome} ({w}x{h})")
                return cap
            relatorio.append(f"  camera {idx} via {nome}: {motivo}")
            if motivo == "so imagem preta" and reserva is None:
                reserva = (idx, backend, nome)
            if motivo != "nao abriu":
                break                          # indice existe; nao repete backends

    print("Nenhuma camera entregou imagem:")
    print("\n".join(relatorio))
    print("Dicas: feche programas que usam a webcam (Teams, Discord, navegador, OBS,\n"
          "NVIDIA Broadcast) ou abra o NVIDIA Broadcast com a webcam como fonte.")
    if reserva is not None:
        idx, backend, nome = reserva
        print(f"Abrindo mesmo assim a camera {idx} ({nome}), que esta mandando imagem preta.")
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    return None


class LeitorCamera:
    """Envolve cap.read() tolerando falhas momentaneas (comuns no Windows)."""

    def __init__(self, cap, max_falhas: int = 60):
        self.cap = cap
        self.max_falhas = max_falhas
        self.falhas = 0

    def read(self):
        import time
        while True:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.falhas = 0
                return True, frame
            self.falhas += 1
            if self.falhas >= self.max_falhas:
                return False, None
            time.sleep(0.02)
