"""
libres_v3.py
------------
Palavras v3: DUAS maos + posicao em relacao ao CORPO.

A v2 olhava uma mao so e media o movimento em relacao a ela mesma. Isso deixa
de fora duas coisas que distinguem sinais em Libras:
  - a outra mao (America, esquina, acontecer, sapo, banheiro usam as duas);
  - o ponto de articulacao: o mesmo movimento perto do rosto ou do peito.

Cada quadro cru tem (formato de minds_v3/*.npz e da camera):
  maos  (2, 21, 3)  ate duas maos do MediaPipe Hand Landmarker (NaN = ausente)
  lado  (2,)        0 = "Left", 1 = "Right", -1 = ausente (rotulo do MediaPipe)
  pose  (33, 4)     MediaPipe Pose Landmarker: x, y, z, visibilidade (NaN = sem pose)
e o aspecto da imagem (largura / altura), para que x e y fiquem na mesma escala.

Daqui sai, por quadro, a mao DOMINANTE (a que mais se move no sinal) e a outra,
cada uma normalizada como na v2 (punho na origem, tamanho da mao = 1), mais a
posicao das maos medida em "larguras de ombro" a partir do centro dos ombros e
do nariz. O vetor de features reaproveita compute_enhanced_features da v2.

FONTE UNICA DA VERDADE para treino_v3.py, avaliar_v3.py e a camera.
"""

from __future__ import annotations

import numpy as np

from libres_features import (FEAT_DIM, N_COORDS, N_LANDMARKS, compute_enhanced_features,
                             normalize_landmarks, resample_sequence, trajectory_features)

FEATURE_VERSION_V3: int = 3
N_FRAMES: int = 30
NARIZ, OMBRO_E, OMBRO_D = 0, 11, 12

# colunas do quadro normalizado (dim 135)
C_FORMA_D = slice(0, 63)        # forma da mao dominante
C_LOC_D = slice(63, 65)         # punho dominante em relacao ao centro dos ombros
C_DEDO_D = slice(65, 67)        # ponta do indicador dominante em relacao ao nariz
C_PRES_O = 67                   # outra mao presente (0/1)
C_FORMA_O = slice(68, 131)      # forma da outra mao
C_LOC_O = slice(131, 133)       # punho da outra mao em relacao ao centro dos ombros
C_REL_O = slice(133, 135)       # punho da outra mao em relacao ao punho dominante
DIM_QUADRO: int = 135
COLS_X: np.ndarray = np.r_[np.arange(0, 63, 3), 63, 65, np.arange(68, 131, 3), 131, 133]


# ------------------------------------------------------------------ rastreio
class RastreadorMaos:
    """Mantem cada mao no mesmo "lugar" (0 ou 1) de um quadro para o outro.

    O MediaPipe devolve as maos em qualquer ordem e o rotulo Left/Right as vezes
    troca num quadro. Aqui cada mao nova vai para o lugar cuja ultima posicao do
    punho esta mais perto; o rotulo so desempata. Funciona igual na camera
    (quadro a quadro) e nos videos do MINDS."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = [None, None]       # ultima posicao (x, y) do punho em cada lugar
        self.rot = [-1, -1]           # rotulo Left/Right visto por ultimo

    def update(self, maos: np.ndarray, lado: np.ndarray) -> np.ndarray:
        """maos (2,21,3) com NaN, lado (2,) -> (2,21,3) na ordem dos lugares."""
        maos = np.asarray(maos, dtype=np.float64).reshape(2, N_LANDMARKS, N_COORDS)
        presentes = [k for k in range(2) if not np.isnan(maos[k, 0, 0])]
        out = np.full_like(maos, np.nan)
        if not presentes:
            return out

        def custo(k: int, lugar: int) -> float:
            p = maos[k, 0, :2]
            c = 0.0 if self.pos[lugar] is None else float(np.linalg.norm(p - self.pos[lugar]))
            if self.rot[lugar] >= 0 and lado[k] >= 0 and lado[k] != self.rot[lugar]:
                c += 0.05             # rotulo diferente pesa pouco (o MediaPipe erra)
            return c

        if len(presentes) == 2:
            a, b = presentes
            if self.pos[0] is None and self.pos[1] is None:
                # comeco: ordena pelo rotulo (Right primeiro) ou pela posicao
                ordem = (a, b) if (lado[a], -maos[a, 0, 0]) >= (lado[b], -maos[b, 0, 0]) else (b, a)
            else:
                direto = custo(a, 0) + custo(b, 1)
                trocado = custo(a, 1) + custo(b, 0)
                ordem = (a, b) if direto <= trocado else (b, a)
            atrib = {ordem[0]: 0, ordem[1]: 1}
        else:
            k = presentes[0]
            if self.pos[0] is None and self.pos[1] is None:
                atrib = {k: 0}
            elif self.pos[1] is None:
                atrib = {k: 0} if custo(k, 0) < 0.25 else {k: 1}
            elif self.pos[0] is None:
                atrib = {k: 1} if custo(k, 1) < 0.25 else {k: 0}
            else:
                atrib = {k: 0 if custo(k, 0) <= custo(k, 1) else 1}
        for k, lugar in atrib.items():
            out[lugar] = maos[k]
            self.pos[lugar] = maos[k, 0, :2].copy()
            self.rot[lugar] = int(lado[k])
        return out


def rastrear(maos: np.ndarray, lado: np.ndarray) -> np.ndarray:
    """Sequencia (T,2,21,3) -> mesma sequencia com as maos em lugares estaveis."""
    r = RastreadorMaos()
    return np.stack([r.update(maos[t], lado[t]) for t in range(len(maos))]) if len(maos) else maos


# ------------------------------------------------------------------ preparo
def _interpolar(arr: np.ndarray) -> np.ndarray:
    """Preenche NaN no tempo por interpolacao linear (coluna a coluna);
    nas pontas repete o valor valido mais proximo. (T, D) -> (T, D)."""
    arr = np.array(arr, dtype=np.float64)
    T = len(arr)
    idx = np.arange(T)
    for d in range(arr.shape[1]):
        col = arr[:, d]
        ok = ~np.isnan(col)
        if ok.all() or not ok.any():
            continue
        arr[:, d] = np.interp(idx, idx[ok], col[ok])
    return arr


def _referencia_corpo(pose: np.ndarray, aspecto: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Centro dos ombros, largura dos ombros e nariz (medianas do sinal), com x
    multiplicado pelo aspecto. Sem pose: usa um corpo "padrao" no meio da tela."""
    p = np.array(pose, dtype=np.float64)
    if p.ndim != 3 or not np.isfinite(p[:, [NARIZ, OMBRO_E, OMBRO_D], :2]).any():
        return np.array([0.5 * aspecto, 0.75]), 0.35 * aspecto, np.array([0.5 * aspecto, 0.35])
    xy = p[:, :, :2].copy()
    xy[..., 0] *= aspecto
    oe, od, na = xy[:, OMBRO_E], xy[:, OMBRO_D], xy[:, NARIZ]
    centro = np.nanmedian((oe + od) / 2, axis=0)
    largura = float(np.nanmedian(np.linalg.norm(oe - od, axis=1)))
    nariz = np.nanmedian(na, axis=0)
    if not np.isfinite(largura) or largura < 1e-3:
        largura = 0.35 * aspecto
    if not np.isfinite(centro).all():
        centro = np.array([0.5 * aspecto, 0.75])
    if not np.isfinite(nariz).all():
        nariz = centro + np.array([0.0, -0.6 * largura])
    return centro, largura, nariz


REPOUSO_Y = 1.0            # mao "baixa": punho mais de 1 largura de ombro abaixo dos ombros (cintura)
REPOUSO_VEL = 0.35         # mao "parada": menos de 0,35 largura de ombro por segundo
REPOUSO_MARGEM = 0.1       # segundos mantidos antes e depois do sinal
REPOUSO_VOLTA = 0.2        # ou: maos a menos disto (larguras de ombro) de onde estavam no comeco
                           # do video (maos descansando juntas na frente da barriga)...
REPOUSO_DESCANSO = 0.6     # ... se esse lugar do comeco e baixo (mais que isto abaixo dos ombros)


def trecho_sem_repouso(maos: np.ndarray, centro: np.ndarray, larg: float, fps: float = 30.0) -> tuple[int, int]:
    """Indices (j0, j1) do sinal sem a mao em REPOUSO no comeco e no fim.

    Repouso = (a) todas as maos visiveis BAIXAS (no colo, na cintura) E PARADAS,
    ou (b) todas as maos visiveis ainda (ou de novo) onde estavam no comeco do
    video, quando o video ja comeca com elas a vista (maos descansando juntas na
    frente da barriga). So corta trechos assim nas pontas: a mao entrando no
    quadro (rapida) e um numero segurado na altura do peito continuam. maos ja
    rastreadas e com x (e z) multiplicados pelo aspecto; centro e larg da
    _referencia_corpo."""
    T = len(maos)
    tem = ~np.isnan(maos[:, :, 0, 0])
    if T < 3:
        return 0, T
    vel = np.zeros(T)
    for k in range(2):
        p = maos[:, k, :, :2] / larg
        dv = np.nan_to_num(np.linalg.norm(np.diff(p, axis=0), axis=2).mean(axis=1), nan=0.0) * fps
        vel[1:] = np.maximum(vel[1:], dv)
    vel[0] = vel[1]
    if T >= 5:
        vel = np.convolve(np.r_[vel[0], vel, vel[-1]], np.ones(3) / 3, mode="valid")
    altura = (maos[:, :, 0, 1] - centro[1]) / larg             # (T, 2): + = abaixo dos ombros
    baixa = np.where(tem, altura > REPOUSO_Y, True).all(axis=1)
    # quadro sem mao (o detector perde a mao parada no colo) nao conta como sinal
    repouso = ~tem.any(axis=1) | (baixa & (vel < REPOUSO_VEL))
    # (b) maos onde estavam no comeco do video
    n0 = max(2, int(round(0.15 * fps)))
    p0 = {}
    for k in range(2):
        if tem[:n0, k].sum() >= max(2, n0 // 2):
            pk = np.median(maos[:n0][tem[:n0, k], k, 0, :2], axis=0)
            if (pk[1] - centro[1]) / larg > REPOUSO_DESCANSO:     # comeca embaixo: e descanso, nao sinal
                p0[k] = pk
    if p0:
        no_lugar = tem.any(axis=1).copy()
        for k in range(2):
            if k in p0:
                d = np.linalg.norm(maos[:, k, 0, :2] - p0[k], axis=1) / larg
                no_lugar &= ~tem[:, k] | (d < REPOUSO_VOLTA)
            else:
                no_lugar &= ~tem[:, k]                  # mao que nao estava la no comeco: e sinal
        repouso |= no_lugar
    sinal = np.where(~repouso)[0]
    if len(sinal) < 3:                                         # nada alem do repouso: nao corta
        return 0, T
    m = int(round(REPOUSO_MARGEM * fps))
    return max(0, int(sinal[0]) - m), min(T, int(sinal[-1]) + 1 + m)


# outra mao PARADA longe do sinal (no colo, na cintura) num sinal de uma mao: algumas
# pessoas deixam a mao a vista em vez de abaixar (LIBRAS-EQT-UECE i05; MINDS s11, s12)
# Medido no MEIO do sinal (sem os 20% do comeco e do fim): quem descansa as maos juntas na
# frente da barriga encosta uma na outra so antes e depois do sinal.
OUTRA_FAIXA = 0.35         # o punho da outra mao anda menos que isto (larguras de ombro)
OUTRA_LONGE = 0.25         # ... as duas maos nunca chegam a menos disto uma da outra (sem toque)
OUTRA_ALTA = 0.6           # ... e o punho dela fica mais que isto abaixo dos ombros (abaixo do peito:
                           # barriga, colo, braco pendurado). Mais alta, pode ser a mao de apoio de um
                           # sinal de duas maos, parada no espaco do sinal sem ser tocada
# Ou: o video comeca com a outra mao a vista e, no miolo do sinal (30% a 70%), ela continua
# onde estava (maos descansando na frente da barriga; LIBRAS-EQT-UECE i05)
OUTRA_FICOU = 0.15         # o punho se afasta menos que isto da posicao do comeco
OUTRA_DESCANSO = 0.6       # ... e essa posicao e baixa (mais que isto abaixo dos ombros)
OUTRA_TOQUE = 0.15         # ... e a dominante nao a toca no miolo (mao de apoio e tocada)


def caminho_suave(w: np.ndarray) -> float:
    """Caminho percorrido por um ponto (n, 2), com media movel de 5 quadros antes:
    o tremor da deteccao numa mao parada nao vira "caminho"."""
    w = np.asarray(w, dtype=np.float64)
    if len(w) < 2:
        return 0.0
    if len(w) >= 5:
        k = np.ones(5) / 5
        w = np.stack([np.convolve(np.r_[[w[0, d]] * 2, w[:, d], [w[-1, d]] * 2], k, mode="valid")
                      for d in range(2)], axis=1)
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())


def escolher_dominante(maos: np.ndarray, sem_tremor: bool = False) -> int:
    """Mao dominante = a que mais anda (empate: a que aparece mais). Com
    sem_tremor, se a escolhida so "andou" por causa do tremor da deteccao (a outra
    anda bem mais com o caminho suavizado), troca. Num sinal de duas maos que
    andam igual, fica a escolha de sempre."""
    tem = ~np.isnan(maos[:, :, 0, 0])
    caminho = []
    for k in range(2):
        w = maos[tem[:, k], k, 0, :2]
        caminho.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0)
    dom = int(np.argmax([caminho[k] + 1e-3 * tem[:, k].sum() for k in range(2)]))
    if sem_tremor and tem[:, 1 - dom].sum() >= 3:
        suave = [caminho_suave(maos[tem[:, k], k, 0, :2]) for k in range(2)]
        if suave[1 - dom] > 1.5 * suave[dom]:
            dom = 1 - dom
    return dom


def medidas_outra_mao(maos: np.ndarray, dom: int, centro: np.ndarray, larg: float) -> dict | None:
    """Da outra mao, no meio do sinal: faixa de movimento do punho, altura (abaixo
    dos ombros) e o quanto as duas maos chegam perto (menor distancia entre
    quaisquer pontos delas; 0 = se tocam), em larguras de ombro. None se ela
    quase nao aparece."""
    out = 1 - dom
    tem = ~np.isnan(maos[:, :, 0, 0])
    vo = tem[:, out]
    if vo.sum() < 3:
        return None
    T = len(maos)
    a, b = int(round(0.2 * T)), int(round(0.8 * T))
    meio = np.zeros(T, bool)
    meio[a:max(b, a + 3)] = True
    vm = vo & meio
    if vm.sum() < 3:                             # sinal curto ou mao so nas pontas
        vm = vo
    wo = maos[vm, out, 0, :2] / larg
    faixa = float((np.percentile(wo, 95, axis=0) - np.percentile(wo, 5, axis=0)).max())
    altura = float(np.median((maos[vm, out, 0, 1] - centro[1]) / larg))
    ambas = tem[:, dom] & vm
    perto = np.inf
    if ambas.sum() >= 3:
        A, B = maos[ambas, dom, :, :2], maos[ambas, out, :, :2]          # (n, 21, 2)
        gap = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=3).min(axis=(1, 2)) / larg
        perto = float(np.percentile(gap, 5))
    # ficou onde estava no comeco? (so se a outra mao ja aparece no comeco do video)
    afastou, altura_ini, perto_nucleo = np.inf, np.nan, np.inf
    ini = vo[:max(3, int(round(0.1 * T)))]
    nucleo = np.zeros(T, bool)
    nucleo[int(round(0.3 * T)):max(int(round(0.7 * T)), int(round(0.3 * T)) + 2)] = True
    if ini.sum() >= 2 and (vo & nucleo).sum() >= 2:
        p0 = np.median(maos[:len(ini)][ini, out, 0, :2], axis=0)
        afastou = float(np.linalg.norm(maos[vo & nucleo, out, 0, :2] - p0, axis=1).max() / larg)
        altura_ini = float((p0[1] - centro[1]) / larg)
        an = tem[:, dom] & vo & nucleo
        if an.sum() >= 2:
            A, B = maos[an, dom, :, :2], maos[an, out, :, :2]
            gap = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=3).min(axis=(1, 2)) / larg
            perto_nucleo = float(np.percentile(gap, 5))
    return {"faixa": faixa, "altura": altura, "perto": perto, "visivel": float(vo.mean()),
            "afastou": afastou, "altura_ini": altura_ini, "perto_nucleo": perto_nucleo}


def regras_outra_mao(med: dict | None) -> tuple[bool, bool]:
    """(estacionada, ficou): as duas regras do outra_mao_parada, separadas (o
    diagnostico_v3.py mostra quanto cada uma pega)."""
    if med is None:
        return False, False
    estacionada = med["faixa"] < OUTRA_FAIXA and med["perto"] > OUTRA_LONGE and med["altura"] > OUTRA_ALTA
    ficou = (med["afastou"] < OUTRA_FICOU and np.isfinite(med["altura_ini"]) and med["altura_ini"] > OUTRA_DESCANSO
             and med["perto_nucleo"] > OUTRA_TOQUE)
    return bool(estacionada), bool(ficou)


def outra_mao_parada(med: dict | None) -> bool:
    """A outra mao nao faz parte do sinal quando (a) fica "estacionada": parada,
    abaixo do peito e sem tocar a dominante; ou (b) o video comeca com ela
    descansando embaixo (barriga, colo) e, no miolo do sinal, ela continua la
    sem ser tocada pela dominante.
    A mao de apoio dos sinais de duas maos (no espaco do sinal, ou tocada pela
    dominante) continua."""
    return any(regras_outra_mao(med))


def preparar(maos: np.ndarray, lado: np.ndarray, pose: np.ndarray, aspecto: float,
             n_frames: int = N_FRAMES, simetrico_pela_posicao: bool = False,
             recortar_repouso: bool = False, fps: float = 30.0,
             sem_mao_parada: bool = False) -> np.ndarray | None:
    """Quadros crus de UM sinal -> sequencia normalizada (n_frames, 135).

    Corta os quadros sem mao do comeco e do fim, mantem as maos em lugares
    estaveis, escolhe a dominante (a que mais anda), preenche falhas curtas e
    reamostra o sinal inteiro para n_frames. None se nao ha mao.

    simetrico_pela_posicao: num sinal de duas maos que andam quase o mesmo tanto
    (pao, por favor), "a que mais anda" vira sorteio e a mesma pessoa gera
    sequencias espelhadas de uma vez para outra. Com True (calibracao), nesses
    casos a dominante e a que fica mais a direita da imagem. O modelo geral
    (treino_v3) continua com a regra original.

    recortar_repouso: tira do comeco e do fim os quadros com a mao parada no colo
    (videos em que a mao ja esta no quadro antes do sinal, como na
    LIBRAS-EQT-UECE); so com o corpo detectado na maior parte do sinal.

    sem_mao_parada: num sinal de uma mao, a outra mao parada a vista (no colo, na
    cintura) e tratada como ausente, igual a de quem abaixa a mao para fora do
    quadro; e a dominante e escolhida pelo caminho sem o tremor da deteccao."""
    maos = np.asarray(maos, dtype=np.float64).reshape(-1, 2, N_LANDMARKS, N_COORDS)
    lado = np.asarray(lado).reshape(-1, 2)
    tem = ~np.isnan(maos[:, :, 0, 0])
    alguma = tem.any(axis=1)
    if alguma.sum() < 3:
        return None
    i0, i1 = int(np.argmax(alguma)), len(alguma) - int(np.argmax(alguma[::-1]))
    maos, lado = maos[i0:i1], lado[i0:i1]
    pose = np.asarray(pose, dtype=np.float64)
    pose = pose[i0:i1] if pose.ndim == 3 and len(pose) >= i1 else pose

    maos = rastrear(maos, lado)
    maos[..., [0, 2]] *= aspecto                 # x, y e z na mesma escala (z vem na escala de x)
    centro, larg, nariz = _referencia_corpo(pose, aspecto)
    if recortar_repouso and pose.ndim == 3 and len(pose) == len(maos):
        corpo = np.isfinite(pose[:, [OMBRO_E, OMBRO_D], :2]).all(axis=(1, 2)).mean()
        if corpo >= 0.5:                         # sem o corpo nao da para saber o que e "baixo"
            j0, j1 = trecho_sem_repouso(maos, centro, larg, fps)
            maos = maos[j0:j1]

    tem = ~np.isnan(maos[:, :, 0, 0])
    caminho = []
    for k in range(2):
        w = maos[tem[:, k], k, 0, :2]
        caminho.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0)
    # dominante: a que mais anda; empate (ou uma so), a que aparece mais
    dom = escolher_dominante(maos, sem_tremor=sem_mao_parada)
    if (simetrico_pela_posicao and min(tem[:, 0].sum(), tem[:, 1].sum()) >= 0.5 * len(maos)
            and min(caminho) >= 0.7 * max(caminho)):
        dom = int(np.argmax([np.nanmean(maos[tem[:, k], k, 0, 0]) for k in range(2)]))
    if tem[:, dom].sum() < 3:
        dom = 1 - dom
    out = 1 - dom
    if sem_mao_parada and outra_mao_parada(medidas_outra_mao(maos, dom, centro, larg)):
        tem[:, out] = False                      # outra mao "estacionada": como se tivesse abaixado

    def forma(k: int) -> np.ndarray:
        f = np.full((len(maos), FEAT_DIM), np.nan)
        ok = tem[:, k]
        if ok.any():
            f[ok] = normalize_landmarks(maos[ok, k].reshape(-1, FEAT_DIM))
        return f

    seq = np.full((len(maos), DIM_QUADRO), np.nan)
    seq[:, C_FORMA_D] = forma(dom)
    seq[:, C_LOC_D] = (maos[:, dom, 0, :2] - centro) / larg
    seq[:, C_DEDO_D] = (maos[:, dom, 8, :2] - nariz) / larg
    seq[:, C_PRES_O] = tem[:, out].astype(float)
    if tem[:, out].sum() >= 2:
        seq[:, C_FORMA_O] = forma(out)
        seq[:, C_LOC_O] = (maos[:, out, 0, :2] - centro) / larg
        seq[:, C_REL_O] = (maos[:, out, 0, :2] - maos[:, dom, 0, :2]) / larg
    else:                                        # sinal de uma mao: outra mao "zerada"
        seq[:, C_FORMA_O] = 0.0
        seq[:, C_LOC_O] = 0.0
        seq[:, C_REL_O] = 0.0
        seq[:, C_PRES_O] = 0.0
    seq = _interpolar(seq)
    seq = np.nan_to_num(seq, nan=0.0)
    return resample_sequence(seq, n_frames)


def mao_dominante_crua(maos: np.ndarray, lado: np.ndarray, aspecto: float = 16 / 9) -> np.ndarray | None:
    """So a mao dominante, crua (T, 63), nos quadros em que ela aparece: o que a
    v2 usaria. Serve para comparar v2 x v3 com exatamente os mesmos videos."""
    maos = rastrear(np.asarray(maos, dtype=np.float64).reshape(-1, 2, N_LANDMARKS, N_COORDS),
                    np.asarray(lado).reshape(-1, 2))
    tem = ~np.isnan(maos[:, :, 0, 0])
    if tem.any(axis=1).sum() < 3:
        return None
    caminho = []
    for k in range(2):
        w = maos[tem[:, k], k, 0, :2] * np.array([aspecto, 1.0])
        caminho.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0)
    dom = int(np.argmax([caminho[k] + 1e-3 * tem[:, k].sum() for k in range(2)]))
    if tem[:, dom].sum() < 3:
        dom = 1 - dom
    return maos[tem[:, dom], dom].reshape(-1, FEAT_DIM)


def preparar_amostra(d: dict, n_frames: int = N_FRAMES, recortar_repouso: bool = False,
                     sem_mao_parada: bool = False) -> np.ndarray | None:
    """Atalho para um .npz do minds_v3 (ou um sinal gravado pela camera)."""
    fps = float(d["fps"]) if "fps" in getattr(d, "files", d) else 30.0
    return preparar(d["maos"], d["lado"], d["pose"], float(d["aspecto"]), n_frames,
                    recortar_repouso=recortar_repouso, fps=fps if 5.0 <= fps <= 120.0 else 30.0,
                    sem_mao_parada=sem_mao_parada)


# ------------------------------------------------------------------ features
def features_v3(seq: np.ndarray) -> np.ndarray:
    """(30, 135) -> vetor de features (1341 valores)."""
    seq = np.asarray(seq, dtype=np.float64)
    dom = compute_enhanced_features(np.hstack([seq[:, C_FORMA_D], seq[:, C_LOC_D]]))
    dedo = trajectory_features(seq[:, C_DEDO_D])
    pres = seq[:, C_PRES_O]
    outra = compute_enhanced_features(np.hstack([seq[:, C_FORMA_O], seq[:, C_LOC_O]]))
    rel = trajectory_features(seq[:, C_REL_O])
    return np.concatenate([dom, dedo, [pres.mean(), pres.min(), pres.max()], outra, rel])


def nomes_features_v3() -> list[str]:
    base = ([f"lm{d}_{s}" for d in range(FEAT_DIM)
             for s in ("media", "std", "min", "max", "range", "diff", "veloc", "mediana")]
            + [f"centro{c}_{s}" for c in range(3) for s in ("media", "std", "range", "diff", "veloc")]
            + [f"rel{d}_{s}" for d in range(14) for s in ("media", "std", "range", "diff", "veloc")]
            + [f"ang{d}_{s}" for d in range(10) for s in ("media", "std", "range", "diff", "veloc")])
    traj = lambda p: ([f"{p}{e}_{s}" for e in "xy"
                       for s in ("std", "range", "diff", "veloc", "terco1", "terco2", "terco3")]
                      + [f"{p}_caminho"])
    return ([f"dom_{n}" for n in base] + traj("dom_local_") + traj("dom_dedo_nariz_")
            + ["outra_presenca_media", "outra_presenca_min", "outra_presenca_max"]
            + [f"outra_{n}" for n in base] + traj("outra_local_") + traj("outra_rel_dom_"))


# ------------------------------------------------------------------ aumento
def espelhar_v3(seq: np.ndarray) -> np.ndarray:
    out = np.array(seq, dtype=np.float64)
    out[:, COLS_X] *= -1
    return out


def aumentar_v3(seq: np.ndarray, rng: np.random.Generator, n: int = 8) -> list[np.ndarray]:
    """Variacoes de uma sequencia normalizada (30, 135): ruido, tamanho da mao,
    braco mais longo ou mais curto, sinal um pouco deslocado, velocidade,
    quadros repetidos, falhas da outra mao e espelhamento (canhotos e camera
    espelhada x videos do MINDS). A outra mao so e mexida onde ela existe:
    num sinal de uma mao ela continua exatamente zero, como no dado real."""
    T, D = seq.shape
    f_dom, f_out = np.arange(0, 63), np.arange(68, 131)
    l_dom, l_out = np.r_[63, 64, 65, 66], np.r_[131, 132, 133, 134]
    saida = []
    for _ in range(n):
        a = np.array(seq, dtype=np.float64)
        tem_o = a[:, C_PRES_O] > 0.5
        sig = rng.uniform(0.002, 0.008)
        esc_mao = rng.uniform(0.92, 1.08)
        esc_braco = rng.uniform(0.85, 1.15)
        desloc = rng.uniform(-0.12, 0.12, 2)
        a[:, f_dom] = (a[:, f_dom] + rng.normal(0, sig, (T, 63))) * esc_mao
        a[:, l_dom] *= esc_braco
        a[:, 63:65] += desloc
        a[:, l_dom] += rng.normal(0, 0.01, (T, 4))
        if tem_o.any():
            a[np.ix_(tem_o, f_out)] = (a[np.ix_(tem_o, f_out)]
                                       + rng.normal(0, sig, (tem_o.sum(), 63))) * esc_mao
            a[np.ix_(tem_o, l_out)] *= esc_braco
            a[np.ix_(tem_o, np.r_[131, 132])] += desloc
            a[np.ix_(tem_o, l_out)] += rng.normal(0, 0.01, (tem_o.sum(), 4))
            if rng.random() < 0.2:                            # outra mao "congela" um trecho
                ini = int(rng.integers(1, T - 2))
                fim = min(T, ini + int(rng.integers(2, max(3, T // 4))))
                a[ini:fim, f_out] = a[ini - 1, f_out]
        if rng.random() < 0.5:
            a = espelhar_v3(a)
        if rng.random() < 0.4:                                # mais rapido / mais lento
            novo = max(2, int(T * rng.uniform(0.85, 1.15)))
            esticado = resample_sequence(a, novo)
            a = esticado[:T] if novo >= T else np.vstack([esticado, np.repeat(esticado[-1:], T - novo, 0)])
        if rng.random() < 0.3:                                # quadros repetidos (falha de deteccao)
            for idx in rng.choice(np.arange(1, T), int(T * rng.uniform(0.05, 0.15)), replace=False):
                a[idx] = a[idx - 1]
        saida.append(a)
    return saida
