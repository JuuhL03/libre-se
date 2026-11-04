"""
libres_features.py
------------------
Modulo unico de extracao de features a partir das sequencias de landmarks
MediaPipe Hands usadas no Libre-se.

FONTE UNICA DA VERDADE: serve tanto o treino (train_rotulado.py) quanto a
inferencia em tempo real (test_camera_temporal.py). Altere aqui uma vez e
os dois lados ficam em sync -- evita o bug classico: a normalizacao muda
no treino mas a inferencia fica presa numa versao diferente, gerando
predicoes fora da distribuicao que o modelo conhece.
"""

import numpy as np

N_FRAMES: int = 30
N_LANDMARKS: int = 21
N_COORDS: int = 3
FEAT_DIM: int = N_LANDMARKS * N_COORDS

# --- PALAVRAS (v2) ----------------------------------------------------------
# normalize_landmarks() zera o punho em todo frame: otimo pro ALFABETO
# (so a forma da mao importa), mas apaga o MOVIMENTO e o deslocamento da mao,
# que sao justamente o que diferencia muitas PALAVRAS em Libras. Na v2 cada
# frame da sequencia de palavra carrega, alem da forma (63 valores), a
# trajetoria do punho (x, y) medida em "tamanhos de mao".
N_TRAJ: int = 2
SEQ_DIM: int = FEAT_DIM + N_TRAJ          # 65 valores por frame (palavras)
FEATURE_VERSION: int = 2                  # gravado no .pkl do modelo de palavras
X_COLS: np.ndarray = np.r_[np.arange(0, FEAT_DIM, N_COORDS), FEAT_DIM]  # colunas x (espelhamento)


def normalize_landmarks(seq: np.ndarray) -> np.ndarray:
    """Converte coordenadas absolutas para posicoes relativas ao punho e a
    escala da propria mao (independe de posicao, distancia da camera e
    resolucao). Sem isso, a mesma forma feita perto/longe gera features
    diferentes e causa falsos positivos (ex: 'a' -> 'familia').
    """
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    r = r - r[:, 0:1]

    # Escala: distancia punho(0) -> MCP do dedo medio(9). Estabiliza o
    # tamanho da mao dentro do frame, igualando sinais em qualquer distancia.
    scale = np.linalg.norm(r[:, 9], axis=1)[:, None, None]
    scale = np.clip(scale, 1e-6, None)
    r = r / scale

    return r.reshape(-1, FEAT_DIM)


def resample_sequence(seq: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """Reamostra uma sequencia (T, D) de qualquer tamanho para (n_frames, D)
    por interpolacao linear no tempo.

    Substitui o antigo "pega os 30 primeiros frames e repete o ultimo":
    assim o sinal INTEIRO cabe na janela, seja ele rapido ou lento, e
    a camera (15 fps) e os videos do dataset (30/60 fps) ficam comparaveis.
    """
    seq = np.asarray(seq, dtype=np.float64)
    T = len(seq)
    if T == n_frames:
        return seq.copy()
    if T == 1:
        return np.repeat(seq, n_frames, axis=0)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n_frames)
    out = np.empty((n_frames, seq.shape[1]))
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(dst, src, seq[:, d])
    return out


def wrist_trajectory(raw_seq: np.ndarray) -> np.ndarray:
    """Trajetoria (T, 2) do punho em coordenadas de imagem, centralizada na
    posicao media e dividida pelo tamanho da mao (punho -> MCP do dedo medio).

    Dividir pelo tamanho da mao torna o movimento independente da distancia
    da camera: "a mao andou 2 maos para a direita" vale perto ou longe.
    """
    r = np.asarray(raw_seq, dtype=np.float64).reshape(-1, N_LANDMARKS, N_COORDS)
    wrist = r[:, 0, :2]
    size = np.median(np.linalg.norm(r[:, 9, :2] - r[:, 0, :2], axis=1))
    size = max(float(size), 1e-6)
    return (wrist - wrist.mean(axis=0)) / size


def normalize_sequence(raw_seq: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """Sequencia CRUA (T, 63) de qualquer tamanho -> (n_frames, 65) para PALAVRAS.

    Colunas 0..62: forma da mao (normalize_landmarks, igual ao alfabeto).
    Colunas 63..64: trajetoria do punho (wrist_trajectory).
    Use SEMPRE esta funcao no treino e na camera de palavras.
    """
    raw = resample_sequence(np.asarray(raw_seq, dtype=np.float64).reshape(-1, FEAT_DIM),
                            n_frames)
    return np.hstack([normalize_landmarks(raw), wrist_trajectory(raw)])


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Espelha horizontalmente uma sequencia ja normalizada (63 ou 65 colunas).

    Usado como augmentacao: torna o modelo indiferente a camera espelhada
    (webcam com flip) x video normal (MINDS) e a sinalizantes canhotos.
    """
    out = seq.copy()
    cols = X_COLS[X_COLS < seq.shape[1]]
    out[:, cols] *= -1
    return out


def compute_finger_angles(seq: np.ndarray) -> np.ndarray:
    """Calcula angulos das articulacoes dos dedos.
    Retorna (T, 10): 2 angulos por dedo.
    """
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)

    triples = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]

    all_angles = []
    for a, b, c in triples:
        v1 = r[:, a] - r[:, b]
        v2 = r[:, c] - r[:, b]
        dot = np.sum(v1 * v2, axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        norm = np.clip(norm, 1e-8, None)
        cos_ang = np.clip(dot / norm, -1.0, 1.0)
        all_angles.append(np.arccos(cos_ang))

    return np.column_stack(all_angles)


def add_relative_features(seq: np.ndarray) -> np.ndarray:
    """Distancias relativas entre landmarks."""
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    extra = []

    wrist = r[:, 0]
    for tip in [4, 8, 12, 16, 20]:
        dist = np.linalg.norm(r[:, tip] - wrist, axis=1)
        extra.append(dist)

    for a, b in [(4, 8), (8, 12), (12, 16), (16, 20),
                 (4, 3), (8, 7), (12, 11), (16, 15), (20, 19)]:
        dist = np.linalg.norm(r[:, a] - r[:, b], axis=1)
        extra.append(dist)

    return np.column_stack(extra) if extra else np.zeros((seq.shape[0], 1))


def trajectory_features(traj: np.ndarray) -> list[float]:
    """Features do movimento do punho (traj = colunas 63..64 da v2).

    Por eixo: dispersao, amplitude, deslocamento liquido, velocidade media e
    a posicao media em cada terco do sinal (o "desenho" do caminho: sobe e
    desce, vai e volta, arco...). Mais o comprimento total do caminho.
    """
    T = len(traj)
    feats: list[float] = []
    thirds = np.array_split(np.arange(T), 3)
    for c in range(traj.shape[1]):
        col = traj[:, c]
        feats.extend([
            np.std(col), np.max(col) - np.min(col), col[-1] - col[0],
            np.mean(np.abs(np.diff(col))),
        ])
        feats.extend(float(np.mean(col[idx])) for idx in thirds)
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    feats.append(float(steps.sum()))
    return feats


def compute_enhanced_features(seq: np.ndarray) -> np.ndarray:
    """Vetor de features: estatisticas temporais + relativas + angulares.

    Aceita (T, 63) [v1, formato antigo] ou (T, 65) [v2, palavras com
    trajetoria]. Para (T, 63) o resultado e identico ao da versao anterior.
    """
    wrist_traj = None
    if seq.shape[1] == SEQ_DIM:
        seq, wrist_traj = seq[:, :FEAT_DIM], seq[:, FEAT_DIM:]
    T, D = seq.shape
    features = []

    for d in range(D):
        col = seq[:, d]
        features.extend([
            np.mean(col), np.std(col), np.min(col), np.max(col),
            np.max(col) - np.min(col), col[-1] - col[0],
            np.mean(np.abs(np.diff(col))), np.median(col),
        ])

    centroids = seq.reshape(T, N_LANDMARKS, N_COORDS).mean(axis=1)
    for c in range(N_COORDS):
        traj = centroids[:, c]
        features.extend([
            np.mean(traj), np.std(traj), np.max(traj) - np.min(traj),
            traj[-1] - traj[0], np.mean(np.abs(np.diff(traj))),
        ])

    rel = add_relative_features(seq)
    for d in range(rel.shape[1]):
        col = rel[:, d]
        features.extend([
            np.mean(col), np.std(col), np.max(col) - np.min(col),
            col[-1] - col[0], np.mean(np.abs(np.diff(col))),
        ])

    angles = compute_finger_angles(seq)
    for d in range(angles.shape[1]):
        col = angles[:, d]
        features.extend([
            np.mean(col), np.std(col), np.max(col) - np.min(col),
            col[-1] - col[0], np.mean(np.abs(np.diff(col))),
        ])

    if wrist_traj is not None:
        features.extend(trajectory_features(wrist_traj))

    return np.array(features)


# --- ALFABETO (v2) ----------------------------------------------------------
# Alem da forma da mao normalizada, o alfabeto v2 gira a mao para uma postura
# padrao (punho -> base do dedo medio apontando para cima). Assim a mesma letra
# feita com a mao um pouco inclinada -- cada pessoa inclina de um jeito --
# gera as mesmas features. A inclinacao original nao se perde: vira 2 features
# (seno e cosseno), e o modelo decide quanto ela importa. Somam-se os angulos
# dos dedos e as distancias entre pontas, que nao dependem de rotacao.
ALPHABET_FEATURE_VERSION: int = 2


def alphabet_features(raw: np.ndarray) -> np.ndarray:
    """(N, 63) ou (63,) landmarks crus -> (N, 89) features do alfabeto v2."""
    r = normalize_landmarks(np.asarray(raw, dtype=np.float64).reshape(-1, FEAT_DIM))
    r = r.reshape(-1, N_LANDMARKS, N_COORDS)
    v = r[:, 9, :2]
    phi = np.arctan2(v[:, 0], -v[:, 1])           # 0 quando o dedo medio aponta para cima
    c, s = np.cos(phi), np.sin(phi)
    x, y = r[:, :, 0], r[:, :, 1]
    rot = r.copy()
    rot[:, :, 0] = c[:, None] * x + s[:, None] * y
    rot[:, :, 1] = -s[:, None] * x + c[:, None] * y
    flat = rot.reshape(-1, FEAT_DIM)
    ang = compute_finger_angles(flat)              # (N, 10)
    rel = add_relative_features(flat)              # (N, 14)
    return np.hstack([flat, ang, rel, np.sin(phi)[:, None], np.cos(phi)[:, None]])


def mirror_raw(raw: np.ndarray) -> np.ndarray:
    """Espelha landmarks crus (N, 63): troca mao direita <-> esquerda."""
    out = np.asarray(raw, dtype=np.float64).reshape(-1, FEAT_DIM).copy()
    out[:, 0::N_COORDS] *= -1
    return out


# --- GEOMETRIA DA MAO (regras para letras que os datasets nao cobrem bem) ----
_DEDOS = {"indicador": (5, 6, 7, 8), "medio": (9, 10, 11, 12),
          "anelar": (13, 14, 15, 16), "minimo": (17, 18, 19, 20)}


def _ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cosv = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))


def geometria_mao(raw: np.ndarray) -> dict:
    """Estado de cada dedo a partir dos 21 pontos (qualquer escala/posicao).

    Retorna angulos (graus; 180 = reto), 'esticado' por dedo, se indicador e
    medio estao cruzados ou afastados, e se o indicador esta em gancho."""
    r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
    r = r - r[0]
    tam = max(float(np.linalg.norm(r[9])), 1e-9)
    r = r / tam
    g = {"angulos": {}, "esticado": {}}
    for nome, (m, p, d, t) in _DEDOS.items():
        a_pip, a_dip = _ang(r[m], r[p], r[d]), _ang(r[p], r[d], r[t])
        g["angulos"][nome] = (a_pip, a_dip)
        g["esticado"][nome] = a_pip > 140 and a_dip > 120 and \
            np.linalg.norm(r[t]) > np.linalg.norm(r[p]) * 1.15
    a_mcp, a_ip = _ang(r[1], r[2], r[3]), _ang(r[2], r[3], r[4])
    g["angulos"]["polegar"] = (a_mcp, a_ip)
    # polegar "para fora": reto e com a ponta longe da base do indicador
    g["esticado"]["polegar"] = a_ip > 140 and np.linalg.norm(r[4] - r[5]) > 0.75
    # indicador x medio, no plano da imagem, num referencial em que o punho -> base
    # do medio aponta para cima (independe da inclinacao da mao e de espelhamento)
    eixo = r[9, :2] / max(np.linalg.norm(r[9, :2]), 1e-9)
    lado = np.array([-eixo[1], eixo[0]])
    x = lambda i: float(np.dot(r[i, :2], lado))
    g["cruzados"] = np.sign(x(5) - x(9)) != np.sign(x(8) - x(12)) and abs(x(8) - x(12)) > 0.05
    v_i, v_m = r[8, :2] - r[5, :2], r[12, :2] - r[9, :2]
    cosv = np.dot(v_i, v_m) / max(np.linalg.norm(v_i) * np.linalg.norm(v_m), 1e-9)
    g["abertura"] = float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    g["dist_pontas"] = float(np.linalg.norm(r[8, :2] - r[12, :2]))
    g["gancho"] = 60 < g["angulos"]["indicador"][0] < 140
    return g


def separacao_pontas(raw: np.ndarray) -> float:
    """Posicao lateral da ponta do indicador em relacao a do medio, no
    referencial da mao (punho -> base do medio = "para cima"), com o sinal
    ajustado pelo lado das bases. Medido nos datasets (mediana):
    R (cruzados) ~0 ou negativa | U (juntos) +0,12 a +0,30 | V (afastados) ~+0,58."""
    r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
    r = r - r[0]
    r = r / max(float(np.linalg.norm(r[9])), 1e-9)
    eixo = r[9, :2] / max(float(np.linalg.norm(r[9, :2])), 1e-9)
    lado = np.array([-eixo[1], eixo[0]])
    x = lambda i: float(np.dot(r[i, :2], lado))
    return (x(8) - x(12)) * (1.0 if x(5) - x(9) >= 0 else -1.0)


def letra_urv(raw: np.ndarray) -> str | None:
    """Desempata U / R / V pela separacao das pontas (None = zona de duvida)."""
    s = separacao_pontas(raw)
    if s < 0.04:
        return "R"
    if 0.07 <= s <= 0.32:
        return "U"
    if s >= 0.36:
        return "V"
    return None


def polegar_ase(raw: np.ndarray) -> tuple[float, float]:
    """Onde esta o polegar numa mao fechada: (t, d). t = posicao da ponta do polegar
    ao longo da linha indicador -> minimo (base dos dedos): no A ele fica AO LADO do
    indicador (t < 0); no S e no E, na frente dos dedos (t ~ 0,5). d = distancia da
    ponta do polegar a 2a articulacao do medio, em tamanhos de mao: no S o polegar
    cruza o meio dos dedos (d pequeno); no E fica embaixo das pontas (d maior)."""
    a = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
    tam = max(float(np.linalg.norm(a[9] - a[0])), 1e-9)
    v = a[17] - a[5]
    t = float(np.dot(a[4] - a[5], v) / max(float(np.dot(v, v)), 1e-12))
    return t, float(np.linalg.norm(a[4] - a[10])) / tam


def letra_ase(raw: np.ndarray, p_s: float = 0.0, p_e: float = 0.0) -> str | None:
    """Desempata A / S / E pelo polegar (None = deixa para o modelo). Treinando sem uma
    das fontes de fotos e testando nela (pessoas que o modelo nunca viu): A de 52-100%
    para 89-100%, S de 4-37% para 60-99%, E de 54-64% para 57-89%."""
    g = geometria_mao(raw)
    if any(g["esticado"][d] for d in ("indicador", "medio", "anelar", "minimo")):
        return None
    t, d = polegar_ase(raw)
    if t < 0.05:
        return "A"
    return "S" if d < 0.46 else "E"


def letra_para_baixo(raw: np.ndarray) -> str | None:
    """M, N e Q da Libras: dedos esticados apontando para BAIXO (o resto da mao fechada).
    M = indicador, medio e anelar; N = indicador e medio juntos (paralelos: no P eles se
    abrem); Q = so o indicador. Nas poses de pessoas que o modelo nao viu: N 61% (o
    modelo, 50%), Q 76% (o modelo, 0%), e o P quase nunca vira N."""
    a = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
    tam = max(float(np.linalg.norm(a[9] - a[0])), 1e-9)
    e = geometria_mao(raw)["esticado"]
    baixo = lambda b, t: (a[t, 1] - a[b, 1]) / tam > 0.35      # y da imagem cresce para baixo
    i, m, an, mi = e["indicador"], e["medio"], e["anelar"], e["minimo"]
    if i and m and an and not mi and baixo(5, 8) and baixo(9, 12) and baixo(13, 16):
        return "M"
    if i and m and not an and not mi and baixo(5, 8) and baixo(9, 12):
        u, v = a[8] - a[5], a[12] - a[9]
        cos = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))
        if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) < 15:
            return "N"
        return None
    if i and not m and not an and not mi and baixo(5, 8):
        return "Q"
    return None


def eh_y(raw: np.ndarray) -> bool:
    """Y: minimo reto, polegar para fora, indicador/medio/anelar bem dobrados.
    Nos datasets: 100% de precisao, pega 92% dos Y (o modelo sozinho: 0%)."""
    g = geometria_mao(raw)
    a = g["angulos"]
    dobrado = lambda d: a[d][0] < 110
    return (a["minimo"][0] > 150 and a["minimo"][1] > 140 and g["esticado"]["polegar"]
            and dobrado("indicador") and dobrado("medio") and dobrado("anelar"))


def indicador_gancho(raw: np.ndarray, g: dict | None = None) -> bool:
    """Indicador curvado em gancho (X da Libras): a primeira falange do
    indicador segue a mao, a ponta dobra, e medio, anelar e minimo estao
    fechados. Distingue do punho fechado (A, S, E) porque ali as articulacoes
    do indicador ficam na mesma altura das do medio. Nos datasets dispara em
    36 de 5.622 poses (A e S); em ganchos simulados a partir de L, G e X, em
    60% a 90%."""
    g = g or geometria_mao(raw)
    a, e = g["angulos"], g["esticado"]
    if e["indicador"] or any(a[d][0] >= 110 for d in ("medio", "anelar", "minimo")):
        return False
    r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
    r = (r - r[0]) / max(float(np.linalg.norm(r[9] - r[0])), 1e-9)
    d = lambda k: float(np.linalg.norm(r[k]))
    return 50 < a["indicador"][0] < 160 and d(6) > d(10) * 1.12 and d(7) > d(11) * 1.2


def polegar_em_l(raw: np.ndarray) -> bool:
    """Polegar aberto para o lado, em angulo com o indicador (o L). No D, no G, no Z
    e no X o polegar fica junto da mao (encostado no medio ou paralelo ao indicador).
    Nas poses reais com so o indicador esticado: pega 91% dos L e 0% dos D, G e X."""
    r = np.asarray(raw, dtype=np.float64).reshape(N_LANDMARKS, N_COORDS)
    tam = max(float(np.linalg.norm(r[9] - r[0])), 1e-9)
    longe = float(np.linalg.norm(r[4] - r[10])) / tam >= 0.8      # ponta do polegar longe do medio
    v1, v2 = r[4] - r[2], r[8] - r[5]
    cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9))
    return longe and np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) >= 38


def forma_mao(raw: np.ndarray) -> str | None:
    """Configuracoes simples e bem definidas (None = deixa para o modelo):
    'Y', 'I' (so o minimo), 'U'/'R'/'V'/'DOIS' (indicador + medio),
    'INDICADOR' (so o indicador), 'L' (indicador e polegar aberto), 'GANCHO'
    (indicador curvado, o X)."""
    if eh_y(raw):
        return "Y"
    g = geometria_mao(raw)
    e = g["esticado"]
    i, m, a, mi = e["indicador"], e["medio"], e["anelar"], e["minimo"]
    if mi and not (i or m or a) and not e["polegar"]:
        return "I"
    if i and m and not (a or mi):
        return letra_urv(raw) or "DOIS"
    if i and not (m or a or mi):
        # o L nao e base de gesto: mexer a mao em L nao pode virar Z ou X
        return "L" if polegar_em_l(raw) else "INDICADOR"
    if indicador_gancho(raw, g):
        return "GANCHO"
    return None


def familia_forma(forma: str | None) -> str | None:
    """Agrupa formas que valem como a mesma num gesto: U/R/V/DOIS -> DOIS
    (H e K); INDICADOR/GANCHO -> IND (Z e X)."""
    if forma in ("U", "R", "V", "DOIS"):
        return "DOIS"
    if forma in ("INDICADOR", "GANCHO"):
        return "IND"
    return forma
