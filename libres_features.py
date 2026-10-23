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
