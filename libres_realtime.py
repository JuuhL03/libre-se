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
                             alphabet_features, mirror_raw, eh_y, forma_mao, letra_urv,
                             compute_enhanced_features, normalize_landmarks,
                             normalize_sequence)


class _TamanhoMao:
    """Tamanho da mao (punho -> base do medio) suavizado: o valor de um quadro
    so tambem treme, e dividir posicoes absolutas por ele cria movimento falso."""

    def __init__(self, alfa: float = 0.15) -> None:
        self.alfa, self.valor = alfa, None

    def reset(self) -> None:
        self.valor = None

    def update(self, r: np.ndarray) -> float:
        atual = max(float(np.linalg.norm(r[9, :2] - r[0, :2])), 1e-6)
        self.valor = atual if self.valor is None else (1 - self.alfa) * self.valor + self.alfa * atual
        return self.valor


class _VelocidadeSuave:
    """Velocidade (maos/s) pelo deslocamento LIQUIDO numa janela curta,
    comparando a posicao media do inicio com a do fim da janela.

    Somar o caminho quadro a quadro transforma o tremor do MediaPipe em
    "movimento" (meio pixel de tremor ja dava >1 mao/s a 30 fps); a media
    cancela o tremor e independe da taxa de quadros."""

    def __init__(self, janela: float = 0.25, min_span: float = 0.0) -> None:
        self.janela = janela
        self.min_span = min_span     # historico minimo (s) antes de medir
        self.hist: list[tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self.hist = []

    def update(self, pontos: np.ndarray, t: float) -> float:
        self.hist.append((t, np.asarray(pontos, dtype=np.float64).reshape(-1, 2)))
        dentro = [(ti, p) for ti, p in self.hist if t - ti <= self.janela]
        if len(dentro) < 4:
            # camera lenta (ex.: 10 quadros/s): a janela teria 2-3 quadros e a
            # velocidade ficaria sempre zero; usa os 4 ultimos (ate 0,6 s)
            dentro = [(ti, p) for ti, p in self.hist[-4:] if t - ti <= max(self.janela, 0.6)]
        self.hist = dentro
        n = len(self.hist)
        if n < 4 or self.hist[-1][0] - self.hist[0][0] < self.min_span:
            return 0.0
        k = max(1, n // 3)
        ini, fim = self.hist[:k], self.hist[-k:]
        p0 = np.mean([p for _, p in ini], axis=0)
        p1 = np.mean([p for _, p in fim], axis=0)
        dt = np.mean([ti for ti, _ in fim]) - np.mean([ti for ti, _ in ini])
        if dt <= 0:
            return 0.0
        return float(np.linalg.norm(p1 - p0, axis=1).max() / dt)


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
        self._vel = _VelocidadeSuave()
        self._tam = _TamanhoMao()

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
                self._t_last = None
                self._t_mao = None
                self._vel.reset()
                self._tam.reset()
                if self.buf:
                    return self._emit()
            return None

        lm = np.asarray(landmarks, dtype=np.float64).reshape(-1)
        self._t_mao = t
        r = lm.reshape(N_LANDMARKS, N_COORDS)
        tam = self._tam.update(r)
        vel = self._vel.update(r[0, :2] / tam, t)
        dt = min(max(t - self._t_last, 1 / 120), 0.5) if self._t_last is not None else 0.0
        desloc = vel * dt
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
    """Velocidade das pontas do indicador e do minimo (maos/s), suavizada.
    Usada para separar letras paradas das letras com movimento."""

    def __init__(self, janela: float = 0.3) -> None:
        # so mede depois de ~0,25 s de mao no quadro: com poucos quadros a
        # media nao cancela o tremor e a mao "parada" parecia se mexer
        self._vel = _VelocidadeSuave(janela, min_span=0.8 * janela)
        self.tam = _TamanhoMao()

    def reset(self) -> None:
        self._vel.reset()
        self.tam.reset()

    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> float:
        import time as _time
        t = _time.time() if t is None else float(t)
        if landmarks is None:
            self.reset()
            return 0.0
        r = np.asarray(landmarks, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
        return self._vel.update(r[[8, 20], :2] / self.tam.update(r), t)


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


# Letras com movimento no alfabeto manual da Libras (H, J, K, X, Z).
LETRAS_DINAMICAS = {"H", "J", "K", "X", "Z"}
# letra parada que tem a mesma forma de mao do inicio de cada letra com movimento
BASE_DINAMICA = {"J": {"I"}, "H": {"U", "V", "R"}, "Z": None, "X": None, "K": None}


class _Segmento:
    """Um trecho continuo de movimento da mao (candidato a H, J, K, X ou Z)."""

    def __init__(self, t: float) -> None:
        self.t0 = t
        self.formas: list = []
        self.estaticas: list = []
        self.top2: list = []
        self.ponta: list = []       # ponta do indicador em "maos" (referencial da imagem)

    def add(self, t, forma, estatica, top, raw, tam: float) -> None:
        r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
        self.formas.append(forma)
        self.estaticas.append(estatica)
        self.top2.append({l for l, _ in top[:2]})
        self.ponta.append(r[8, :2] / tam)
        self.t1 = t

    def inversoes_x(self, minimo: float = 0.25) -> int:
        """Quantas vezes a ponta do indicador muda de direcao na horizontal
        (com histerese, para ignorar tremor). Z ~ 2 ou mais; X ~ 0 ou 1."""
        xs = [p[0] for p in self.ponta]
        n, ref, direcao = 0, xs[0] if xs else 0.0, 0
        for x in xs[1:]:
            d = x - ref
            if direcao >= 0 and d < -minimo:
                n += direcao > 0
                direcao, ref = -1, x
            elif direcao <= 0 and d > minimo:
                n += direcao < 0
                direcao, ref = 1, x
            elif (direcao > 0 and x > ref) or (direcao < 0 and x < ref):
                ref = x
        return n

    def predominio(self, fracao: float = 0.6):
        validas = [f for f in self.formas if f is not None]
        if not validas:
            return None
        from collections import Counter
        forma, n = Counter("DOIS" if f in ("U", "R", "V", "DOIS") else f for f in validas).most_common(1)[0]
        return forma if n >= fracao * len(self.formas) else None

    def letra(self) -> Optional[str]:
        dur = self.t1 - self.t0
        if dur < 0.3 or len(self.formas) < 4:
            return None
        forma = self.predominio()
        fim = [("DOIS" if f in ("U", "R", "V", "DOIS") else f) for f in self.formas[-max(2, len(self.formas) // 3):]]
        mantida = forma is not None and fim.count(forma) >= len(fim) / 2   # a forma dura ate o fim
        pts = np.array(self.ponta)
        dx, dy = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        if mantida and forma == "I":
            return "J"
        if mantida and forma == "DOIS" and dx >= dy:
            return "H"
        if mantida and forma == "INDICADOR":
            # Z: vai e volta na horizontal E desce (o traco diagonal); um X
            # sacudido varias vezes tambem inverte a direcao, mas quase nao desce
            return "Z" if self.inversoes_x() >= 2 and dy >= 0.3 * dx else "X"
        if sum("K" in t for t in self.top2) >= 0.5 * len(self.top2):
            return "K"
        return None


class Soletrador:
    """Decide, quadro a quadro, quando uma letra entra no texto.

    - mao PARADA: letra estatica (modelo + regras de geometria para Y e para
      desempatar U/R/V), sobre a media dos ultimos 0,2 s (probabilidades e
      pontos da mao, contra o tremor); precisa ficar estavel ~0,3 s;
    - mao EM MOVIMENTO: nada entra durante o gesto; quando o movimento termina,
      ele e classificado como H, J, K, X ou Z pela forma da mao e pelo caminho
      (J = forma de I; H = U/V com movimento lateral; Z = indicador em
      zigue-zague que desce; X = indicador com movimento curto, sem descer;
      K = indicado pelo modelo);
    - letra repetida (OSSO) exige tirar a mao do quadro entre as duas;
    - se a letra parada que acabou de entrar era o comeco do gesto (o "I" do
      "J"), ela e trocada pela letra com movimento.
    """

    def __init__(self, rec: "LetterRecognizer", conf_min: float = 0.55,
                 tempo_estavel: float = 0.3, mov_limiar: float = 1.5,
                 fim_movimento: float = 0.25, janela_troca: float = 1.5,
                 janela_prob: float = 0.2) -> None:
        self.rec = rec
        self.janela_prob = janela_prob   # media das probabilidades (s): tira o tremor
        self.conf_min = conf_min
        self.tempo_estavel = tempo_estavel
        self.mov_limiar = mov_limiar
        self.fim_movimento = fim_movimento
        self.janela_troca = janela_troca
        self.mov = MovimentoMao()
        self.ultimo = ""
        self.reset()

    def reset(self) -> None:
        self.estavel, self.desde, self.emitido = "", 0.0, False
        self.mao_saiu = True
        self.t_ultima = -1e9
        self.residuo = None
        self.seg: Optional[_Segmento] = None
        self.t_parou: Optional[float] = None
        self.n_mov = 0                   # quadros seguidos em movimento
        self.n_dif = 0                   # quadros seguidos com outra forma de mao
        self.hist_prob: list = []
        self.hist_raw: list = []
        self.forma_ultimo = None         # forma da mao quando a ultima letra parada entrou
        self.mov.reset()

    # ------------------------------------------------------------ estatica
    def _top_suave(self, top: list, t: float) -> list:
        """Media das probabilidades dos ultimos `janela_prob` segundos. Com a mao
        parada, o tremor do MediaPipe faz a confianca oscilar (ex.: L entre 48%
        e 86%) e zerava a contagem; a media estabiliza sem atrasar quase nada."""
        self.hist_prob.append((t, dict(top)))
        self.hist_prob = [(ti, d) for ti, d in self.hist_prob if t - ti <= self.janela_prob]
        if len(self.hist_prob) == 1:
            return top
        n = len(self.hist_prob)
        media = {l: sum(d.get(l, 0.0) for _, d in self.hist_prob) / n for l in self.rec.classes}
        return sorted(media.items(), key=lambda x: -x[1])

    def _raw_suave(self, raw: np.ndarray, t: float) -> np.ndarray:
        """Media dos pontos da mao na mesma janela: o tremor fazia a distancia
        entre as pontas (U x R x V) pular de uma faixa para outra."""
        self.hist_raw.append((t, raw))
        self.hist_raw = [(ti, r) for ti, r in self.hist_raw if t - ti <= self.janela_prob]
        return np.mean([r for _, r in self.hist_raw], axis=0)

    def _estatica(self, raw: np.ndarray, top: list) -> tuple[str, float]:
        est = [(l, c) for l, c in top if l not in LETRAS_DINAMICAS] or top
        letra, conf = est[0]
        if eh_y(raw):
            return "Y", max(conf, 0.95)
        if letra in ("U", "R", "V"):
            fam = sum(c for l, c in est if l in ("U", "R", "V"))
            urv = letra_urv(raw)
            if urv:
                return urv, fam
        return letra, conf

    def _emitir(self, letra: str, t: float, dinamica: bool, base: Optional[set],
                t_inicio: Optional[float] = None) -> Optional[str]:
        # a janela da troca conta a partir do INICIO do gesto (o "I" entrou logo antes do "J")
        ref = t if t_inicio is None else t_inicio
        if dinamica and base and not self.mao_saiu and ref - self.t_ultima <= self.janela_troca \
                and self.ultimo in base:
            acao = "troca"
        elif letra != self.ultimo or self.mao_saiu:
            acao = "add"
        else:
            return None
        self.ultimo, self.t_ultima, self.mao_saiu = letra, t, False
        return acao

    def _fechar_segmento(self, t: float, info: dict) -> None:
        seg, self.seg = self.seg, None
        self.n_dif = 0
        letra = seg.letra() if seg else None
        if not letra:
            return
        base = BASE_DINAMICA.get(letra)
        if letra in ("Z", "X"):
            # so troca a letra anterior se ela foi feita com o indicador esticado
            # (o "D"/"L" que entrou antes do zigue-zague); no JUIZ, o "I" fica
            base = {self.ultimo} if self.forma_ultimo == "INDICADOR" else set()
        elif base is None:              # K: a letra parada do comeco do gesto
            base = set(seg.estaticas[:3])
        acao = self._emitir(letra, t, True, base, seg.t0)
        if acao:
            self.forma_ultimo = None
            info.update(letra=letra, conf=1.0, aceita=True, acao=acao)
            # o que sobra parado no fim do gesto e a letra da forma predominante
            # (ex.: o "I" depois do "J"), nao a dos quadros de transicao
            from collections import Counter
            dom = seg.predominio(0.5)
            norm = lambda f: "DOIS" if f in ("U", "R", "V", "DOIS") else f
            do_gesto = [e for f, e in zip(seg.formas, seg.estaticas) if norm(f) == dom]
            self.residuo = Counter(do_gesto).most_common(1)[0][0] if do_gesto else None
            self.estavel, self.emitido = "", False

    # ------------------------------------------------------------ quadro
    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> dict:
        import time as _time
        t = _time.time() if t is None else float(t)
        info = {"letra": "", "conf": 0.0, "top": [], "aceita": False, "acao": None,
                "movendo": False}
        if landmarks is None:
            if self.seg is not None:
                self._fechar_segmento(t, info)
            self.estavel, self.emitido = "", False
            self.mao_saiu = True
            self.residuo = None
            self.t_parou = None
            self.hist_prob, self.hist_raw = [], []
            self.mov.update(None, t)
            return info

        raw = np.asarray(landmarks, dtype=np.float64).reshape(-1)
        top = self._top_suave(self.rec.predict_top(raw, len(self.rec.classes)), t)
        geo = self._raw_suave(raw, t)    # pose media: regras de geometria sem tremor
        letra, conf = self._estatica(geo, top)
        movendo = self.mov.update(raw, t) >= self.mov_limiar
        self.n_mov = self.n_mov + 1 if movendo else 0
        movendo = self.n_mov >= 2        # ignora picos de 1 quadro
        info.update(letra=letra, conf=conf, top=top[:3], movendo=movendo)

        if movendo:
            self.t_parou = None
            forma = forma_mao(raw)
            if self.seg is not None and len(self.seg.formas) >= 4:
                # a forma da mao mudou no meio do movimento: o gesto anterior acabou
                # (ex.: o "J" emendado na transicao para a proxima letra)
                dom = self.seg.predominio(0.5)
                atual = "DOIS" if forma in ("U", "R", "V", "DOIS") else forma
                # forma None = duvida (tremor), nao conta como troca nem zera a contagem
                if dom is None:
                    self.n_dif = 0
                elif atual is not None:
                    self.n_dif = self.n_dif + 1 if atual != dom else 0
                if self.n_dif >= 2:
                    self.n_dif = 0
                    self._fechar_segmento(t, info)
            if self.seg is None:
                self.seg = _Segmento(t)
            self.seg.add(t, forma, letra, top, raw, self.mov.tam.valor or 1.0)
            self.estavel, self.emitido = "", False
            return info                  # nada entra durante o gesto

        ok = conf >= self.conf_min
        info["aceita"] = ok
        if not ok:
            self.estavel, self.emitido = "", False
        elif letra != self.estavel:
            self.estavel, self.desde, self.emitido = letra, t, False

        if self.seg is not None:         # o gesto terminou? (a letra parada ja vai contando)
            if self.t_parou is None:
                self.t_parou = t
            if t - self.t_parou < self.fim_movimento:
                return info
            self._fechar_segmento(t, info)
            if info["acao"]:
                return info

        if not ok or self.emitido or t - self.desde < self.tempo_estavel:
            return info
        self.emitido = True
        if self.residuo is not None:
            if letra == self.residuo:
                return info              # final parado de um gesto: nao conta
            self.residuo = None
        acao = self._emitir(letra, t, False, None)
        if acao:
            self.forma_ultimo = forma_mao(geo)
            info["acao"] = acao
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
