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

from pathlib import Path
from typing import Optional

import numpy as np

from libres_modelos import carregar_modelo
from libres_features import (FEAT_DIM, FEATURE_VERSION, N_COORDS, N_LANDMARKS,
                             alphabet_features, mirror_raw, eh_y, familia_forma, forma_mao,
                             letra_urv, letra_ase, letra_para_baixo,
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
                 rearm_speed: float = 1.8,
                 cauda_parada: float = 0.0) -> None:
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
        cauda_parada: quanto (s) do trecho parado do fim fica no sinal. 0 = corta
                      tudo (os videos do MINDS nao tem a mao parada no fim); na
                      calibracao, sinais parados (numeros) precisam da mao parada
        """
        self.min_duration = min_duration
        self.min_frames = min_frames
        self.max_duration = max_duration
        self.gap_time = gap_time
        self.still_time = still_time
        self.still_speed = still_speed
        self.min_motion = min_motion
        self.rearm_speed = rearm_speed
        self.cauda_parada = cauda_parada
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
            corte = next((i for i, ti in enumerate(self.tbuf) if ti >= self.still_since + self.cauda_parada),
                         len(self.tbuf))
            self.buf, self.tbuf = self.buf[:max(corte, 1)], self.tbuf[:max(corte, 1)]
            self.waiting = True
            return self._emit()
        return None


def normal_palma(r: np.ndarray) -> np.ndarray:
    """Normal (unitaria) do plano da palma: punho, base do indicador e do minimo."""
    n = np.cross(r[5] - r[0], r[17] - r[0])
    return n / max(float(np.linalg.norm(n)), 1e-9)


def angulo(a: np.ndarray, b: np.ndarray) -> float:
    """Angulo (graus) entre dois vetores."""
    c = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def isotropico(raw: np.ndarray, aspecto: float = 1.0) -> np.ndarray:
    """Pontos do MediaPipe na mesma escala nos tres eixos. x vem dividido pela
    largura da imagem e y pela altura: numa camera 16:9 um dedo apontando para
    cima parece 1,78x maior do que apontando para o lado. z ja usa a escala de x."""
    r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS).copy()
    if aspecto != 1.0:
        r[:, 1] /= aspecto
    return r


def tamanho_mao(r: np.ndarray) -> float:
    """Tamanho da mao em 3D: media das distancias do punho as bases dos quatro
    dedos. Nao muda quando a mao gira (o tamanho na imagem muda) e treme menos
    que uma distancia so."""
    return float(np.mean(np.linalg.norm(r[[5, 9, 13, 17]] - r[0], axis=1)))


# distancias (em 3D, divididas pelo tamanho da mao) que descrevem a configuracao
# da mao sem depender de posicao, tamanho ou giro: ponta de cada dedo ao punho
# (dobrado x esticado), polegar ate a base do indicador, polegar ate as pontas
# do indicador e do medio (O, K) e indicador ate o medio (U x V x R)
_PARES_FORMA = ((4, 0), (8, 0), (12, 0), (16, 0), (20, 0), (4, 5), (4, 8), (4, 12), (8, 12))
_PARES_A = [a for a, _ in _PARES_FORMA]
_PARES_B = [b for _, b in _PARES_FORMA]


def descritor_forma(r: np.ndarray, tam: Optional[float] = None) -> np.ndarray:
    tam = tamanho_mao(r) if tam is None else tam
    return np.linalg.norm(r[_PARES_A] - r[_PARES_B], axis=1) / max(tam, 1e-9)


class MovimentoMao:
    """Intensidade do movimento da mao em "tamanhos de mao por segundo", com tres
    componentes, cada uma comparando a media do comeco e a do fim de uma janela
    curta (o que cancela o tremor da deteccao):

    - translacao das pontas do indicador e do minimo (J, Z, K, X de lado);
    - mudanca de escala: a mao indo para tras ou para a frente (X);
    - giro da palma: o vetor entre as bases do indicador e do minimo muda de
      direcao/comprimento quando a mao gira (H), e a normal da palma (em 3D)
      muda de direcao mesmo quando a mao ja comeca de lado.

    update() devolve a maior delas, ja na escala do limiar (1,5)."""

    ESCALA_REF = 0.25    # log do tamanho por segundo que equivale a 1,5 maos/s
    GIRO_REF = 0.8       # mudanca do vetor da palma (em maos) por segundo idem (giro do H)
    GIRO_ANG_REF = 90.0  # giro da normal da palma (graus por segundo) idem

    def __init__(self, janela: float = 0.3) -> None:
        self.janela = janela
        self.min_span = 0.8 * janela     # so mede depois de ~0,25 s de mao no quadro
        self.tam = _TamanhoMao()
        self.reset()

    def reset(self) -> None:
        self.hist: list = []
        self.tam.reset()
        self.ultimo = {"trans": 0.0, "escala": 0.0, "giro": 0.0}

    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> float:
        import time as _time
        t = _time.time() if t is None else float(t)
        if landmarks is None:
            self.reset()
            return 0.0
        r = np.asarray(landmarks, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
        self.tam.update(r)
        tam = max(tamanho_mao(r), 1e-6)
        # escala pelo raio de giro dos 21 pontos (3D): media de muitos pontos treme
        # ~4x menos que uma distancia so, entao da para ver o X indo para tras
        # mesmo a 15 quadros/s; nao muda quando a mao gira
        raio = max(float(np.sqrt(np.mean(np.sum((r - r.mean(axis=0)) ** 2, axis=1)))), 1e-6)
        self.hist.append((t, r[[8, 20], :2].copy(), tam, (r[5, :2] - r[17, :2]) / tam, normal_palma(r), raio))
        self.hist = [h for h in self.hist if t - h[0] <= max(self.janela, 0.6)]
        dentro = [h for h in self.hist if t - h[0] <= self.janela]
        # camera lenta ou falhas da deteccao no comeco da janela (comum no meio de
        # um gesto rapido): estende a janela para tras, ate 0,6 s, em vez de zerar
        # a velocidade (o que cortava o Z ao meio)
        k = len(self.hist) - len(dentro)
        while k > 0 and (len(dentro) < 4 or dentro[-1][0] - dentro[0][0] < self.min_span):
            k -= 1
            dentro = [self.hist[k]] + dentro
        n = len(dentro)
        self.ultimo = {"trans": 0.0, "escala": 0.0, "giro": 0.0, "giro_ang": 0.0}
        # 3 quadros bastam se cobrem o tempo minimo (rajada de falhas no gesto)
        if n < 3 or dentro[-1][0] - dentro[0][0] < self.min_span:
            return 0.0
        k = max(1, n // 3)
        ini, fim = dentro[:k], dentro[-k:]
        dt = np.mean([h[0] for h in fim]) - np.mean([h[0] for h in ini])
        if dt <= 0:
            return 0.0
        tam_m = float(np.mean([h[2] for h in dentro]))
        media = lambda grupo, j: np.mean([h[j] for h in grupo], axis=0)
        trans = float(np.linalg.norm(media(fim, 1) - media(ini, 1), axis=1).max()) / tam_m / dt
        escala = abs(float(np.mean([np.log(h[5]) for h in fim]) - np.mean([np.log(h[5]) for h in ini]))) / dt
        giro = float(np.linalg.norm(media(fim, 3) - media(ini, 3))) / dt
        giro_ang = angulo(media(ini, 4), media(fim, 4)) / dt
        self.ultimo = {"trans": trans, "escala": escala, "giro": giro, "giro_ang": giro_ang}
        return max(trans, escala / self.ESCALA_REF * 1.5, giro / self.GIRO_REF * 1.5,
                   giro_ang / self.GIRO_ANG_REF * 1.5)


class WordRecognizer:
    def __init__(self, model_path: Path) -> None:
        data = carregar_modelo(model_path)
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
        data = carregar_modelo(model_path)
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
# letras paradas que podem ser o comeco (ou o meio) de um Z ou de um X: o indicador sozinho
BASES_ZX = {"D", "G", "P", "X", "Z", "I"}
BASES_HK = {"H", "K", "U", "V", "R"}      # indicador e medio levantados


class _Segmento:
    """Um trecho continuo de movimento da mao (candidato a H, J, K, X ou Z).

    Os pontos chegam ja isotropicos (ver isotropico())."""

    # variacao maxima do descritor_forma dentro do nucleo do gesto. Trocas de letra
    # mudam o descritor em 0,4 a 1,5; girar a mao 75 graus com o z do MediaPipe
    # subestimado pela metade muda ate ~0,3 (gancho) -> 0,4 separa os dois casos
    TOL_FORMA = 0.4

    def __init__(self, t: float) -> None:
        self.t0 = self.t1 = t
        self.ts: list = []
        self.formas: list = []
        self.estaticas: list = []
        self.top3: list = []
        self.ponta: list = []       # ponta do indicador (imagem)
        self.punho: list = []       # punho (imagem)
        self.tams: list = []        # tamanho da mao (3D)
        self.palma: list = []       # vetor base do indicador -> base do minimo, em maos
        self.normais: list = []     # normal da palma (3D)
        self.desc: list = []        # configuracao da mao (descritor_forma)
        self.nucleo_ij: Optional[tuple] = None

    def add(self, t, forma, estatica, top, raw, tam: float = 1.0) -> None:
        r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
        tq = max(tamanho_mao(r), 1e-6)
        self.ts.append(t)
        self.formas.append(forma)
        self.estaticas.append(estatica)
        self.top3.append({l for l, _ in top[:3]})
        self.ponta.append(r[8, :2].copy())
        self.punho.append(r[0, :2].copy())
        self.tams.append(tq)
        self.palma.append((r[5, :2] - r[17, :2]) / tq)
        self.normais.append(normal_palma(r))
        self.desc.append(descritor_forma(r, tq))
        self.t1 = t

    # ---------------------------------------------------------------- medidas
    @staticmethod
    def _inversoes(xs, minimo: float) -> int:
        """Mudancas de direcao com histerese (ignora tremor)."""
        n, ref, direcao = 0, xs[0] if len(xs) else 0.0, 0
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

    def _janela(self, i: int, j: Optional[int]) -> tuple[int, int]:
        n = len(self.formas)
        return max(0, i), (n - 1 if j is None else min(j, n - 1))

    def predominio(self, fracao: float = 0.6, i: int = 0, j: Optional[int] = None):
        """Familia de forma predominante (DOIS, IND, I...) ou None, nos quadros i..j.

        Conta so os quadros em que a forma foi reconhecida: no meio de um gesto
        rapido a mao borra e varios quadros ficam sem forma (None), o que antes
        derrubava o gesto inteiro. Exige forma reconhecida em 40% dos quadros."""
        i, j = self._janela(i, j)
        formas = self.formas[i:j + 1]
        validas = [familia_forma(f) for f in formas if f is not None]
        if not validas or len(validas) < 0.4 * len(formas):
            return None
        from collections import Counter
        forma, n = Counter(validas).most_common(1)[0]
        return forma if n >= fracao * len(validas) else None

    def _quadros_em(self, segundos: float) -> int:
        """Quadros (impar) que cabem em `segundos`, na taxa deste gesto."""
        dur = max(self.t1 - self.t0, 1e-6)
        k = int(segundos * (len(self.formas) - 1) / dur)
        return max(1, k + (k % 2 == 0 and k > 1))

    def _k_suave(self) -> int:
        """Media movel de ~0,1 s (30 fps -> 3 quadros; 15 fps -> 1): tira o
        tremor do caminho sem apagar o zigue-zague do Z."""
        return self._quadros_em(0.1)

    @staticmethod
    def _suavizar(a: np.ndarray, k: int) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        if k <= 1 or len(a) < 3:
            return a
        pad = k // 2
        ap = np.concatenate([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)])
        ker = np.ones(k) / k
        return np.stack([np.convolve(ap[:, j], ker, mode="valid") for j in range(a.shape[1])], axis=1)

    def nucleo(self) -> tuple[int, int]:
        """O maior trecho em que a CONFIGURACAO da mao fica a mesma.

        Uma letra com movimento mexe a mao inteira sem mudar a forma (o I do J,
        o gancho do X); numa troca de letra (O -> L, K -> I) e a forma que muda.
        O gesto costuma vir grudado na troca para a letra (A -> gancho -> X) ou
        na troca seguinte (J -> U): o movimento que conta e o deste trecho."""
        n = len(self.desc)
        if n < 2:
            return 0, n - 1
        D = self._suavizar(np.array(self.desc), self._quadros_em(0.2))
        melhor = (0, 0)
        for i in range(n):
            if n - 1 - i <= melhor[1] - melhor[0]:
                break
            mn = mx = D[i]
            for j in range(i + 1, n):
                mn, mx = np.minimum(mn, D[j]), np.maximum(mx, D[j])
                if float((mx - mn).max()) > self.TOL_FORMA:
                    break
                if j - i > melhor[1] - melhor[0]:
                    melhor = (i, j)
        return melhor

    def medidas(self, i: int = 0, j: Optional[int] = None) -> dict:
        """Medidas do movimento nos quadros i..j (em tamanhos de mao)."""
        i, j = self._janela(i, j)
        sl = slice(i, j + 1)
        k = self._k_suave()
        un = float(np.median(self.tams[sl]))
        P = self._suavizar(np.array(self.ponta[sl]) / un, k)
        W = self._suavizar(np.array(self.punho[sl]) / un, k)
        palma = self._suavizar(np.array(self.palma[sl]), k)
        normais = self._suavizar(np.array(self.normais[sl]), k)
        logt = self._suavizar(np.log(np.array(self.tams[sl]))[:, None], self._quadros_em(0.15))[:, 0]
        n0 = np.mean(normais[:max(1, k)], axis=0)
        return {"dur": self.ts[j] - self.ts[i], "n": j - i + 1,
                "dx": float(np.ptp(P[:, 0])), "dy": float(np.ptp(P[:, 1])),
                "wdx": float(np.ptp(W[:, 0])), "wdy": float(np.ptp(W[:, 1])),
                "escala": float(np.ptp(logt)),
                "giro": float(np.linalg.norm(palma - palma[0], axis=1).max()),
                "giro_ang": max(angulo(n0, nv) for nv in normais),
                "inv_x": self._inversoes(P[:, 0], 0.25) if len(P) else 0,
                "kh": float(np.mean([bool(s & {"K", "H"}) for s in self.top3[sl]])),
                "gancho": float(np.mean([f == "GANCHO" for f in self.formas[sl]]))}

    def letra(self) -> Optional[str]:
        """Classifica o gesto pela forma da mao e pelo caminho, no nucleo (o
        trecho com a forma constante): J = forma do I andando; Z = indicador
        com zigue-zague que desce; X = indicador (esticado ou em gancho) andando
        ou indo para tras, sem zigue-zague; K e H = indicador e medio levantados
        (mesma forma), K subindo e descendo, H girando a mao ou indo para o lado."""
        if not self.formas:
            return None
        i, j = self.nucleo_ij = self.nucleo()
        m = self.medidas(i, j)
        # gesto curto, mas com movimento de verdade (a 15 quadros/s um X dura 4-5 quadros)
        if m["dur"] < 0.19 or m["n"] < 3:
            return None
        # cada letra exige o seu tipo de movimento da MAO (nao dos dedos)
        anda = max(m["wdx"], m["wdy"]) >= 0.25      # o punho andou (J, Z, X, K, H de lado)
        # a mao foi para tras (X) sem girar: girando, a profundidade estimada pelo
        # MediaPipe erra e o tamanho 3D da mao muda mesmo sem ela se afastar
        recua = m["escala"] >= 0.08 and m["giro_ang"] < 30
        gira = m["giro_ang"] >= 40                   # a palma girou (H)
        if not (anda or recua or gira):
            return None
        forma = self.predominio(0.6, i, j)
        if forma == "I":
            return "J" if anda else None
        # a letra que o modelo parado ve na maior parte do gesto
        from collections import Counter
        est = [e for e in self.estaticas[i:j + 1] if e]
        dom, n_dom = Counter(est).most_common(1)[0] if est else (None, 0)
        dom = dom if n_dom >= 0.5 * max(len(est), 1) else None
        if forma == "IND" and m["kh"] >= 0.5 and (dom is None or dom in BASES_HK | BASES_ZX):
            forma = "DOIS"           # o modelo ve K/H: indicador e medio, e o K subindo
        if forma == "IND":
            if not (anda or recua):
                return None
            # o modelo parado ve, na maior parte do gesto, uma letra que nao e a base do
            # Z nem do X (L, A, S, E, N, K...): e essa mao se mexendo, nao Z nem X. Uma mao
            # fechada as vezes parece "indicador em gancho", e o L, "so o indicador"
            if dom is not None and dom not in BASES_ZX:
                return None
            if m["inv_x"] >= 2 and m["dy"] >= 0.3 * m["dx"] and m["gancho"] < 0.5:
                return "Z"
            return "X"
        if forma == "DOIS" or m["kh"] >= 0.5:
            if m["wdy"] >= 0.35 and m["wdy"] >= 1.2 * m["wdx"]:
                return "K"               # sobe e desce
            if gira or (m["wdx"] >= 0.5 and m["wdx"] >= m["wdy"]):
                return "H"               # gira a mao (ou vai para o lado)
        return None


class Soletrador:
    """Decide, quadro a quadro, quando uma letra entra no texto.

    - mao PARADA: letra estatica (modelo + regras de geometria para Y e para
      desempatar U/R/V), sobre a media dos ultimos 0,2 s (probabilidades e
      pontos da mao, contra o tremor); precisa ficar estavel ~0,3 s;
    - mao EM MOVIMENTO (translacao, ida para tras ou giro; acima de 1,5 mao/s,
      ou de 1,0 mantido por 0,25 s): nada entra durante o gesto; quando ele
      termina, e classificado como H, J, K, X ou Z pela forma da mao e pelo
      movimento, medido so no trecho em que a forma nao muda (ver
      _Segmento.nucleo e _Segmento.letra);
    - falhas da deteccao nao contam como "mao saiu": ate 0,35 s com a mao
      parada e ate 0,6 s no meio de um movimento (a mao borrada some mais);
      o gesto continua e nenhuma letra se repete;
    - letra repetida (OSSO) exige tirar a mao do quadro entre as duas;
    - se a letra parada que acabou de entrar era o comeco do gesto (o "I" do
      "J"), ela e trocada pela letra com movimento.
    """

    def __init__(self, rec: "LetterRecognizer", conf_min: float = 0.55,
                 tempo_estavel: float = 0.3, mov_limiar: float = 1.5,
                 fim_movimento: float = 0.25, janela_troca: float = 4.0,
                 janela_prob: float = 0.2, tempo_ausente: float = 0.35,
                 mov_saida: float = 0.6, aspecto: float = 1.0) -> None:
        self.rec = rec
        self.aspecto = aspecto           # largura / altura da imagem (pontos isotropicos)
        self.janela_prob = janela_prob   # media das probabilidades (s): tira o tremor
        self.conf_min = conf_min
        self.tempo_estavel = tempo_estavel
        self.mov_limiar = mov_limiar
        self.fim_movimento = fim_movimento
        self.janela_troca = janela_troca
        self.tempo_ausente = tempo_ausente
        # no meio de um gesto a deteccao falha mais (mao borrada): so conta como
        # "mao saiu" depois de mais tempo (o Z continua depois de 0,4 s sem mao)
        self.tempo_ausente_gesto = max(tempo_ausente, 0.6)
        self.mov_saida = mov_saida       # histerese: o gesto continua acima de 60% do limiar
        self.mov_lento = 2 / 3           # movimento lento e sustentado (0,25 s) tambem e gesto
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
        self.t_ult_parado: Optional[float] = None   # ultimo quadro VISTO parado (pausa real)
        self.n_mov = 0                   # quadros seguidos em movimento
        self.t_lento: Optional[float] = None   # inicio do movimento lento atual
        self.em_gesto = False
        self.t_q_ant: Optional[float] = None
        self.recentes: list = []         # quadros parados recentes (comeco do gesto)
        self.t_fim_seg = -1e9
        self.n_dif = 0                   # quadros seguidos com outra forma de mao
        self.hist_prob: list = []
        self.hist_raw: list = []
        self.forma_ultimo = None         # familia da forma quando a ultima letra parada entrou
        self.t_mao: Optional[float] = None   # ultimo quadro com mao
        self.fora = True                 # a saida da mao ja foi tratada
        self.info_ant: Optional[dict] = None
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
        if getattr(self.rec, "ultimo_confiante", False):
            return letra, conf     # alfabeto calibrado: o exemplo da propria pessoa decide
        if eh_y(raw):
            return "Y", max(conf, 0.95)
        baixo = letra_para_baixo(raw)          # M, N, Q: dedos apontando para baixo
        if baixo:
            return baixo, max(conf, 0.9)
        if letra in ("U", "R", "V"):
            fam = sum(c for l, c in est if l in ("U", "R", "V"))
            urv = letra_urv(raw)
            if urv:
                return urv, fam
        if letra in ("A", "S", "E"):
            # mao fechada: o modelo confunde A, S e E; quem decide e o polegar
            prob = {l: c for l, c in top}
            fam = sum(prob.get(l, 0.0) for l in ("A", "S", "E"))
            ase = letra_ase(raw, prob.get("S", 0.0), prob.get("E", 0.0))
            if ase:
                return ase, fam
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
        # os quadros parados depois do ultimo quadro do gesto ainda podem ser o
        # comeco do proximo (a mao para um instante entre a troca e o X)
        self.t_fim_seg = seg.t1 if seg is not None else t
        self.n_dif = 0
        letra = seg.letra() if seg else None
        if not letra:
            return
        i, j = seg.nucleo_ij or (0, len(seg.formas) - 1)
        dom = seg.predominio(0.5, i, j)
        # so troca a letra parada anterior se ela tinha a mesma forma de mao do
        # gesto (o "I" antes do "J", o "D" antes do "Z", o "V" antes do "H");
        # no JUIZ, o "I" fica. Sem forma clara (K pelo modelo), vale a letra
        # parada vista no comeco do gesto.
        if letra == "J":
            base = {"I"}
        elif dom is not None and self.forma_ultimo == dom:
            base = {self.ultimo}
        elif dom is None:
            base = set(seg.estaticas[i:i + 3])
        else:
            base = set()
        acao = self._emitir(letra, t, True, base, seg.t0)
        if acao:
            self.forma_ultimo = None
            info.update(letra=letra, conf=1.0, aceita=True, acao=acao)
            # o que sobra parado no fim do gesto e a letra da forma predominante
            # (ex.: o "I" depois do "J"), nao a dos quadros de transicao
            from collections import Counter
            do_gesto = [e for f, e in zip(seg.formas[i:j + 1], seg.estaticas[i:j + 1])
                        if familia_forma(f) == dom]
            self.residuo = Counter(do_gesto).most_common(1)[0][0] if do_gesto else None
            # a letra parada que ja vinha contando continua (se for a da forma do
            # gesto, o residuo impede que ela entre de novo)
            self.emitido = False

    def _limite_ausente(self) -> float:
        # a mao sumiu NO MEIO de um movimento (gesto ou movimento lento comecando):
        # mao borrada, nao "mao saiu". Se ja tinha parado, vale o tempo normal.
        lento = self.t_lento is not None and self.t_mao is not None and self.t_mao - self.t_lento >= 0.1
        return self.tempo_ausente_gesto if (self.em_gesto or lento) else self.tempo_ausente

    def finalizar(self, t: Optional[float] = None) -> dict:
        """Fecha o gesto que estiver aberto (ex.: antes de trocar de modo ou de
        sair), como se a mao tivesse saido do quadro. Devolve o info do update."""
        import time as _time
        t = _time.time() if t is None else float(t)
        info = {"letra": "", "conf": 0.0, "top": [], "aceita": False, "acao": None, "movendo": False}
        if self.seg is not None:
            self._mao_saiu(t, info)
        return info

    def _mao_saiu(self, t: float, info: dict) -> None:
        """A mao ficou fora do quadro tempo suficiente: fecha o gesto e libera a
        repeticao de letras."""
        if self.seg is not None:
            self._fechar_segmento(t, info)
        self.estavel, self.emitido = "", False
        self.mao_saiu = True
        self.residuo = None
        self.t_parou = self.t_ult_parado = None
        self.hist_prob, self.hist_raw = [], []
        self.n_mov = 0
        self.t_lento = None
        self.em_gesto = False
        self.t_q_ant = None
        self.recentes = []
        self.mov.reset()
        self.fora = True

    # ------------------------------------------------------------ quadro
    def update(self, landmarks: Optional[np.ndarray], t: Optional[float] = None) -> dict:
        import time as _time
        t = _time.time() if t is None else float(t)
        info = {"letra": "", "conf": 0.0, "top": [], "aceita": False, "acao": None,
                "movendo": False}
        if landmarks is None:
            if self.fora or self.t_mao is None:
                return info
            if t - self.t_mao >= self._limite_ausente():
                self._mao_saiu(t, info)
                return info
            # falha curta da deteccao: mantem tudo (gesto, contagem) e repete a tela
            if self.info_ant:
                info.update({k: v for k, v in self.info_ant.items() if k != "acao"})
            return info

        if not self.fora and self.t_mao is not None and t - self.t_mao >= self._limite_ausente():
            self._mao_saiu(t, info)          # camera lenta: a saida so aparece agora
            if info["acao"]:
                self.t_mao, self.fora = t, False
                return info
        self.t_mao, self.fora = t, False

        raw = np.asarray(landmarks, dtype=np.float64).reshape(-1)
        top = self._top_suave(self.rec.predict_top(raw, len(self.rec.classes)), t)
        geo = self._raw_suave(raw, t)    # pose media: regras de geometria sem tremor
        letra, conf = self._estatica(geo, top)
        iso = isotropico(raw, self.aspecto)     # medidas de movimento: mesma escala em x, y, z
        vel = self.mov.update(iso, t)
        dt_q = t - self.t_q_ant if self.t_q_ant is not None else 1 / 30
        self.t_q_ant = t
        if self.em_gesto:
            # histerese: dentro de um gesto basta 60% do limiar para continuar;
            # sem isso o K (sobe, para la em cima, desce) virava dois pedacos curtos
            self.em_gesto = vel >= self.mov_saida * self.mov_limiar
            if not self.em_gesto:
                self.n_mov = 0
                self.t_lento = None
        else:
            self.n_mov = self.n_mov + 1 if vel >= self.mov_limiar else 0
            # movimento mais lento, mas sustentado (um J pequeno e devagar): 2/3 do
            # limiar por 0,25 s seguidos tambem conta (o tremor nao se sustenta)
            if vel >= self.mov_lento * self.mov_limiar:
                self.t_lento = t if self.t_lento is None else self.t_lento
            else:
                self.t_lento = None
            # ignora picos de 1 quadro (a 10 quadros/s, 1 quadro ja e 0,1 s)
            self.em_gesto = (self.n_mov >= 2 or (self.n_mov >= 1 and dt_q >= 0.08)
                             or (self.t_lento is not None and t - self.t_lento >= 0.25))
        movendo = self.em_gesto
        info.update(letra=letra, conf=conf, top=top[:3], movendo=movendo)

        forma = forma_mao(raw)
        if movendo:
            # pausa = tempo em que a mao foi VISTA parada (quadros sem mao no meio de
            # um gesto rapido nao sao pausa: antes cortavam o Z em dois)
            pausa = 0.0 if self.t_parou is None or self.t_ult_parado is None else \
                (self.t_ult_parado - self.t_parou) + min(t - self.t_ult_parado, 1 / 15)
            if self.seg is not None and pausa >= 0.15:
                # a mao parou um instante e voltou a mexer: o gesto anterior acabou
                # (ex.: o H seguido da troca para o O, sem tirar a mao)
                self._fechar_segmento(t, info)
                if info["acao"]:
                    self.t_parou = self.t_ult_parado = None
                    self.info_ant = dict(info)
                    return info
            self.t_parou = self.t_ult_parado = None
            if self.seg is not None and len(self.seg.formas) >= 4:
                # a forma da mao mudou no meio do movimento: o gesto anterior acabou
                # (ex.: o "J" emendado na transicao para a proxima letra)
                dom = self.seg.predominio(0.5)
                atual = familia_forma(forma)
                # forma None = duvida (tremor), nao conta como troca nem zera a contagem
                if dom is None:
                    self.n_dif = 0
                elif atual is not None:
                    self.n_dif = self.n_dif + 1 if atual != dom else 0
                if self.n_dif >= 2:
                    self.n_dif = 0
                    self._fechar_segmento(t, info)
            if self.seg is None:
                # o movimento so e detectado alguns quadros depois de comecar (a
                # velocidade e media de 0,3 s): o gesto inclui os ultimos 0,3 s
                # parados, para medir o giro/caminho inteiro (sobretudo a 15 fps)
                pre = [q for q in self.recentes if q[0] > self.t_fim_seg]
                dentro = [q for q in pre if q[0] >= t - 0.3]
                # falhas da deteccao logo antes do gesto: pega os ultimos 3 quadros
                # vistos (ate 0,65 s), para ter a posicao/giro de partida
                pre = dentro if len(dentro) >= 3 else pre[-3:]
                self.seg = _Segmento(pre[0][0] if pre else t)
                for q in pre:
                    self.seg.add(*q, self.mov.tam.valor or 1.0)
            self.seg.add(t, forma, letra, top, iso, self.mov.tam.valor or 1.0)
            if vel >= self.mov_limiar:
                self.estavel, self.emitido = "", False
            elif conf >= self.conf_min and letra != self.estavel:
                # finalzinho do gesto (abaixo do limiar, dentro da histerese): a letra
                # parada ja comeca a contar, para nao atrasar a proxima letra
                self.estavel, self.desde, self.emitido = letra, t, False
            self.info_ant = dict(info)
            return info                  # nada entra durante o gesto

        self.recentes.append((t, forma, letra, top, iso))
        self.recentes = [q for q in self.recentes if t - q[0] <= 0.65]
        ok = conf >= self.conf_min
        info["aceita"] = ok
        if not ok:
            self.estavel, self.emitido = "", False
        elif letra != self.estavel:
            self.estavel, self.desde, self.emitido = letra, t, False

        if self.seg is not None:         # o gesto terminou? (a letra parada ja vai contando)
            if self.t_parou is None:
                self.t_parou = t
            if vel < self.mov_saida * self.mov_limiar:
                self.t_ult_parado = t        # parada de verdade (nao o 1o quadro de um movimento)
            if t - self.t_parou < self.fim_movimento:
                self.info_ant = dict(info)
                return info
            self._fechar_segmento(t, info)
            if info["acao"]:
                self.info_ant = dict(info)
                return info

        if not ok or self.emitido or t - self.desde < self.tempo_estavel:
            self.info_ant = dict(info)
            return info
        self.emitido = True
        if self.residuo is not None:
            if letra == self.residuo:
                self.info_ant = dict(info)
                return info              # final parado de um gesto: nao conta
            self.residuo = None
        acao = self._emitir(letra, t, False, None)
        if acao:
            self.forma_ultimo = familia_forma(forma_mao(geo))
            info["acao"] = acao
        self.info_ant = dict(info)
        return info


def accept_word(top: list[tuple[str, float]],
                min_conf: float = 0.45, min_margin: float = 0.15) -> Optional[str]:
    """Aceita a palavra top-1 se tiver confianca e margem sobre a 2a."""
    if not top:
        return None
    p1 = top[0][1]
    p2 = top[1][1] if len(top) > 1 else 0.0
    return top[0][0] if p1 >= min_conf and (p1 - p2) >= min_margin else None


class Camera:
    """cv2.VideoCapture que sabe qual camera e: `indice`, `backend`,
    `resolucao` (largura, altura) e `estado` da imagem ao abrir ("ok", "imagem
    parada" ou "so imagem preta"). Todo o resto (read, set, get, isOpened...)
    e repassado ao VideoCapture."""

    def __init__(self, cap, indice: int, backend: str, resolucao: tuple[int, int],
                 estado: str = "ok") -> None:
        self.cap = cap
        self.indice = indice
        self.backend = backend
        self.resolucao = resolucao
        self.estado = estado

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def __getattr__(self, nome):
        return getattr(self.cap, nome)


class ExposicaoCamera:
    """Exposicao no SENSOR da webcam (teclas [ e ] do app). Com muita luz (pele clara,
    parede branca) a mao estoura e some para o MediaPipe; filtro na imagem nao devolve
    o contorno perdido (testado: piora a deteccao), entao o ajuste e na camera, antes
    da imagem. Tenta a exposicao manual; se a camera nao deixar, o brilho."""

    def __init__(self, cam) -> None:
        self.cam = cam
        self.cap = getattr(cam, "cap", cam)
        self.backend = getattr(cam, "backend", "")
        self.manual = False
        self.inicial: dict = {}
        self.nivel = 0                    # passos a partir do normal: voltar a 0 = automatico

    def _auto_valores(self) -> tuple[float, float]:
        # (manual, automatico) de CAP_PROP_AUTO_EXPOSURE: DirectShow usa 0.25/0.75; V4L2 1/3
        return (0.25, 0.75) if self.backend == "DirectShow" else (1.0, 3.0)

    def ajustar(self, passo: int) -> str:
        """passo -1 = mais escura, +1 = mais clara. Devolve o texto para o painel."""
        import cv2
        c = self.cap
        if self.nivel + passo == 0 and (self.manual or self.nivel):
            self.automatico()             # voltou ao ponto de partida: exposicao automatica de novo
            return "exposicao da camera: normal (automatica)"
        if not self.inicial:
            self.inicial = {"exp": c.get(cv2.CAP_PROP_EXPOSURE), "auto": c.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                            "brilho": c.get(cv2.CAP_PROP_BRIGHTNESS)}
        manual, _ = self._auto_valores()
        if not self.manual:
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
            self.manual = True
        atual = c.get(cv2.CAP_PROP_EXPOSURE)
        if atual <= 0:                    # DirectShow/MSMF: log2 dos segundos (-13 a 0)
            novo = float(min(0.0, max(-13.0, round(atual) + passo)))
        else:                             # V4L2: unidades de 100 us
            novo = float(max(1.0, atual * (1.5 if passo > 0 else 1 / 1.5)))
        if c.set(cv2.CAP_PROP_EXPOSURE, novo) and abs(c.get(cv2.CAP_PROP_EXPOSURE) - atual) > 1e-6:
            self.nivel += passo
            return (f"exposicao da camera: {self.nivel:+d} em relacao ao normal "
                    f"({'mais clara' if passo > 0 else 'mais escura'}; E volta ao normal)")
        brilho = c.get(cv2.CAP_PROP_BRIGHTNESS)       # sem exposicao manual: tenta o brilho
        passo_b = 16 if abs(brilho) > 2 else 0.06
        if c.set(cv2.CAP_PROP_BRIGHTNESS, brilho + passo * passo_b) and c.get(cv2.CAP_PROP_BRIGHTNESS) != brilho:
            self.nivel += passo
            return f"brilho da camera: {c.get(cv2.CAP_PROP_BRIGHTNESS):g} (a camera nao deixa mudar a exposicao)"
        return "esta camera nao deixa mudar exposicao nem brilho: use fundo escuro e luz de frente"

    def automatico(self) -> str:
        import cv2
        _, auto = self._auto_valores()
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
        if self.inicial.get("brilho") is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.inicial["brilho"])
        self.manual = False
        self.nivel = 0
        return "exposicao da camera: normal (automatica)"


def estourada(frame) -> float:
    """Fracao da imagem quase branca (sem contorno), medida numa amostra."""
    p = frame[::8, ::8]
    return float((p.mean(axis=2) >= 245).mean())


def _backends() -> list:
    import sys
    import cv2
    if sys.platform.startswith("win"):
        return [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "padrao")]
    return [(cv2.CAP_ANY, "padrao")]


def _imagem_parada(cap, quadro, segundos: float = 1.2) -> bool:
    """True se a imagem nao muda NADA por ~1 s. Camera de verdade sempre tem
    ruido do sensor, entao dois quadros nunca saem identicos; imagem identica e
    camera virtual mostrando uma tela fixa (OBS com a camera virtual desligada,
    NVIDIA Broadcast, DroidCam sem o celular...) ou tampa de privacidade."""
    import time
    ref = quadro[::4, ::4].copy()
    lidos = 0
    t0 = time.time()
    while time.time() - t0 < segundos:
        ok, q = cap.read()
        if ok and q is not None and q.shape == quadro.shape:
            lidos += 1
            if not np.array_equal(q[::4, ::4], ref):
                return False
        time.sleep(0.03)
    return lidos >= 3


def _abrir_indice(idx: int, largura: int, altura: int, exigente: bool):
    """Abre a camera `idx` testando os backends. Devolve (Camera|None, motivo, backend).

    exigente=True (escolha automatica): so devolve a camera se a imagem estiver
    viva; preta ou parada volta como None com o motivo. exigente=False (camera
    escolhida pelo usuario): devolve qualquer camera que entregue imagem."""
    import time
    import cv2
    for backend, nome in _backends():
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # muitas webcams so entregam 10 fps em 1280x720 no formato padrao (YUY2);
        # em MJPG chegam a 30 fps
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        cap.set(cv2.CAP_PROP_FPS, 30)
        quadro = None
        t0 = time.time()
        while time.time() - t0 < (2.5 if exigente else 1.0):   # exposicao automatica
            ok, q = cap.read()
            if ok and q is not None and q.size:
                quadro = q
                if float(q.mean()) > 8.0:          # tem imagem de verdade
                    break
            else:
                time.sleep(0.03)
        if quadro is None:
            # o indice existe, mas a camera nao manda imagem (em uso por outro
            # programa); outro backend nao resolve
            cap.release()
            return None, "nao entregou imagem", nome
        if float(quadro.mean()) <= 8.0:
            estado = "so imagem preta"
        elif exigente and _imagem_parada(cap, quadro):
            estado = "imagem parada"
        else:
            estado = "ok"
        if exigente and estado != "ok":
            cap.release()
            return None, estado, nome
        h, w = quadro.shape[:2]
        return Camera(cap, idx, nome, (w, h), estado), estado, nome
    return None, "nao abriu", None


_DICAS_CAMERA = ("Dicas: feche programas que usam a webcam (Teams, Discord, navegador, OBS,\n"
                 "NVIDIA Broadcast) ou escolha a camera pelo numero: --camera 0, 1, 2...")


def abrir_camera(largura: int = 1280, altura: int = 720, indice: Optional[int] = None,
                 avisar: bool = True):
    """Abre a webcam e devolve uma Camera (ou None).

    indice=None (padrao): testa as cameras 0 a 5 e fica com a primeira de imagem
    VIVA. Pula as que so mandam imagem preta (NVIDIA Broadcast, OBS... quando o
    programa delas nao esta transmitindo) e as de imagem PARADA (camera virtual
    do OBS desligada, DroidCam sem o celular...); essas so sao usadas se nao
    houver outra.
    indice=N (ou a variavel de ambiente LIBRES_CAMERA=N): abre exatamente a
    camera N, mesmo que a imagem esteja preta ou parada.
    No Windows o backend padrao (MSMF) as vezes abre mas nao le ("can't grab
    frame" = camera em uso por outro programa); o DirectShow e testado primeiro.
    """
    import os

    if indice is None and os.environ.get("LIBRES_CAMERA", "").strip():
        indice = int(os.environ["LIBRES_CAMERA"])
    if indice is not None:
        cam, motivo, nome = _abrir_indice(indice, largura, altura, exigente=False)
        if avisar:
            if cam is None:
                print(f"Camera {indice}: {motivo}." + (f" (via {nome})" if nome else ""))
                print(_DICAS_CAMERA)
            else:
                w, h = cam.resolucao
                print(f"Camera {indice} aberta via {nome} ({w}x{h})"
                      + ("  [imagem preta]" if cam.estado != "ok" else ""))
        return cam

    pulei, reserva = [], None
    relatorio = []
    for idx in range(6):
        cam, motivo, nome = _abrir_indice(idx, largura, altura, exigente=True)
        if cam is not None:
            if avisar:
                for linha in pulei:
                    print(linha)
                w, h = cam.resolucao
                print(f"Camera {idx} aberta via {nome} ({w}x{h})")
                if pulei:
                    print("  (para usar outra: tecla C na janela, ou --camera N)")
            return cam
        relatorio.append(f"  camera {idx}" + (f" via {nome}" if nome else "") + f": {motivo}")
        if motivo == "imagem parada":
            pulei.append(f"  camera {idx} ({nome}): imagem parada, parece camera virtual (OBS...) - pulada")
        elif motivo == "so imagem preta":
            pulei.append(f"  camera {idx} ({nome}): so imagem preta - pulada")
        elif motivo == "nao entregou imagem":
            pulei.append(f"  camera {idx} ({nome}): nao entregou imagem (em uso por outro programa?)")
        if motivo in ("imagem parada", "so imagem preta") and reserva is None:
            reserva = idx

    if avisar:
        print("Nenhuma camera com imagem viva:")
        print("\n".join(relatorio))
        print(_DICAS_CAMERA)
    if reserva is not None:
        if avisar:
            print(f"Abrindo mesmo assim a camera {reserva}; na janela, a tecla C troca de camera.")
        return abrir_camera(largura, altura, indice=reserva, avisar=avisar)
    return None


def trocar_camera(atual, largura: int = 1280, altura: int = 720, n_max: int = 6):
    """Tecla C: fecha a camera atual e abre a proxima (indice +1, +2, ...,
    voltando ao 0). Pula as que nao entregam imagem; as de imagem preta ou
    parada entram (quem esta trocando ve e aperta C de novo). Devolve a nova
    Camera, ou None se nenhuma abrir (nem a atual)."""
    import time
    idx0 = getattr(atual, "indice", 0)
    try:
        atual.release()
    except Exception:  # noqa: BLE001
        pass
    depois = list(range(idx0 + 1, n_max))
    antes = list(range(0, idx0))
    for grupo in (depois, antes):
        for idx in grupo:
            cam, motivo, _ = _abrir_indice(idx, largura, altura, exigente=False)
            if cam is not None:
                return cam
            if motivo == "nao abriu" and grupo is depois:
                break                          # os indices sao seguidos: acabou a lista
    for tentativa in range(3):                 # nenhuma outra: volta para a mesma camera
        cam, _, _ = _abrir_indice(idx0, largura, altura, exigente=False)
        if cam is not None:
            return cam
        time.sleep(0.5)                        # o Windows pode demorar a soltar a camera
    return None


class MonitorImagem:
    """Avisa quando a imagem da camera fica PARADA (identica quadro a quadro:
    camera virtual mostrando uma tela fixa) ou PRETA por mais de `segundos`.
    update() devolve "", "parada" ou "preta"."""

    def __init__(self, segundos: float = 1.5) -> None:
        self.segundos = segundos
        self.reset()

    def reset(self) -> None:
        self.ref = None
        self.t_viva: Optional[float] = None

    def update(self, quadro, t: float) -> str:
        peq = quadro[::8, ::8]
        preta = float(peq.mean()) <= 8.0
        igual = (self.ref is not None and peq.shape == self.ref.shape
                 and np.array_equal(peq, self.ref))
        self.ref = peq.copy()
        if self.t_viva is None or not (preta or igual):
            self.t_viva = t
            return ""
        if t - self.t_viva < self.segundos:
            return ""
        return "preta" if preta else "parada"


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


# ============================================================ PALAVRAS v3
class SignSegmenterV3(SignSegmenter):
    """Mesmo segmentador (mesmos tempos e limites), mas para quadros com ate
    DUAS maos e a pose: o sinal comeca quando alguma mao aparece, a velocidade e
    a da mao que mais se move e o que sai e o pacote cru do sinal
    {"maos" (T,2,21,3), "lado" (T,2), "pose" (T,33,4)} para o WordRecognizerV3."""

    def reset(self) -> None:
        super().reset()
        from libres_v3 import RastreadorMaos
        self._rast = RastreadorMaos()
        self._vels = [_VelocidadeSuave(), _VelocidadeSuave()]
        self._tams = [_TamanhoMao(), _TamanhoMao()]
        self.qbuf: list = []

    def _emit(self):
        quadros, self.qbuf = self.qbuf, []
        tempos = list(self.tbuf)
        seq = super()._emit()          # aplica os mesmos limites de duracao/quadros
        if seq is None or not quadros:
            return None
        n = len(seq)                   # o buffer pode ter sido cortado (cauda parada)
        quadros, tempos = quadros[:n], tempos[:n]
        dt = np.diff(tempos) if len(tempos) >= 2 else np.array([])
        fps = float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else 30.0
        return {"maos": np.stack([q["maos"] for q in quadros]),
                "lado": np.stack([q["lado"] for q in quadros]),
                "pose": np.stack([q["pose"] for q in quadros]),
                "fps": fps}            # fps real da camera (o --recortar mede velocidade por segundo)

    def update(self, quadro: Optional[dict], t: Optional[float] = None):
        import time as _time
        t = _time.time() if t is None else float(t)
        maos = None if quadro is None else np.asarray(quadro["maos"], dtype=np.float64).reshape(2, 21, 3)
        if maos is None or np.isnan(maos[:, 0, 0]).all():
            if self._t_mao is not None and t - self._t_mao >= self.gap_time:
                self._rast.reset()
                for v in self._vels:
                    v.reset()
                for tm in self._tams:
                    tm.reset()
            return super().update(None, t)

        lugares = self._rast.update(maos, np.asarray(quadro["lado"]).reshape(2))
        vel = 0.0
        for k in range(2):
            if np.isnan(lugares[k, 0, 0]):
                self._vels[k].reset()
                self._tams[k].reset()
                continue
            tam = self._tams[k].update(lugares[k])
            vel = max(vel, self._vels[k].update(lugares[k, 0, :2] / tam, t))
        # a mao "principal" so serve para o buffer da classe base; quem manda e vel
        principal = lugares[0] if not np.isnan(lugares[0, 0, 0]) else lugares[1]
        self._quadro_atual = {"maos": np.asarray(quadro["maos"], dtype=np.float32).reshape(2, 21, 3),
                              "lado": np.asarray(quadro["lado"]).reshape(2),
                              "pose": np.asarray(quadro["pose"], dtype=np.float32).reshape(33, 4)}
        return self._passo(principal.reshape(-1), vel, t)

    def _passo(self, lm: np.ndarray, vel: float, t: float):
        """A maquina de estados do SignSegmenter.update, com a velocidade ja medida."""
        self._t_mao = t
        dt = min(max(t - self._t_last, 1 / 120), 0.5) if self._t_last is not None else 0.0
        desloc = vel * dt
        self._last, self._t_last = lm, t
        if self.waiting:
            if vel < self.rearm_speed:
                return None
            self.waiting = False
        self.buf.append(lm)
        self.tbuf.append(t)
        self.qbuf.append(self._quadro_atual)
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
            self.buf, self.tbuf, self.qbuf = self.buf[-1:], self.tbuf[-1:], self.qbuf[-1:]
            self.path = 0.0
            self.still_since = t
            return None
        if parado:
            corte = next((i for i, ti in enumerate(self.tbuf) if ti >= self.still_since + self.cauda_parada),
                         len(self.tbuf))
            corte = max(corte, 1)
            self.buf, self.tbuf, self.qbuf = self.buf[:corte], self.tbuf[:corte], self.qbuf[:corte]
            self.waiting = True
            return self._emit()
        return None


class WordRecognizerV3:
    """Modelo de palavras v3 (modelo_palavras_v3.pkl)."""

    def __init__(self, model_path: Path) -> None:
        from libres_v3 import FEATURE_VERSION_V3
        data = carregar_modelo(model_path)
        if data.get("feature_version") != FEATURE_VERSION_V3:
            raise RuntimeError(f"{model_path.name} nao e um modelo v3. Rode `python treino_v3.py`.")
        self.model, self.scaler, self.le = data["model"], data["scaler"], data["label_encoder"]
        self.classes = list(self.le.classes_)
        self.versao = 3
        self.recortar = bool(data.get("recortar", False))     # o mesmo preparo do treino
        self.sem_mao_parada = bool(data.get("sem_mao_parada", False))

    def predict(self, sinal: dict, aspecto: float, top_k: int = 3) -> list[tuple[str, float]]:
        from libres_v3 import features_v3, preparar
        fps = float(sinal.get("fps", 30.0))
        seq = preparar(sinal["maos"], sinal["lado"], sinal["pose"], aspecto, recortar_repouso=self.recortar,
                       sem_mao_parada=self.sem_mao_parada, fps=fps if 5.0 <= fps <= 120.0 else 30.0)
        if seq is None:
            return []
        proba = self.model.predict_proba(self.scaler.transform(features_v3(seq).reshape(1, -1)))[0]
        order = np.argsort(proba)[::-1][:top_k]
        return [(str(self.le.inverse_transform([i])[0]), float(proba[i])) for i in order]
