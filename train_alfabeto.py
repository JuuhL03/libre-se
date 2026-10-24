#!/usr/bin/env python3
"""
Treino de ALFABETO (sinais estaticos) com Random Forest.

Diferente do train_rotulado.py (palavras dinamicas, sequencias de 30
frames), aqui cada letra e uma POSE: um unico frame de landmarks
normalizados. Extrai todos os frames de mao parada de cada video
palavras2/<LETRA>.mp4, filtra os frames de transicao pela velocidade
da mao e treina um classificador simples.

Uso: python3 train_alfabeto.py
"""

import pickle
import numpy as np
from pathlib import Path

import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from libres_features import (N_LANDMARKS, N_COORDS, FEAT_DIM, normalize_landmarks,
                             alphabet_features, ALPHABET_FEATURE_VERSION)

PROJECT_DIR = Path(__file__).parent
PALAVRAS_DIR = PROJECT_DIR / "palavras2"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
OUTPUT_MODEL = PROJECT_DIR / "modelo_alfabeto.pkl"

# Letras de 1 caractere seguidas de .mp4 (A.mp4, B.mp4 ...). Z nao
# aparece: em Libras, J e Z envolvem movimento, nao pose estatica.
LETTER_RE = None  # filtro feito via nome do arquivo abaixo

# frames de transicao (mao se mexendo) viram lixo: mantemos so os 60%
# mais parados de cada video
VELOCITY_KEEP_PCT = 0.60

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

print("=" * 60)
print("TREINO ALFABETO - Random Forest (poses estaticas)")
print("=" * 60)

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))


def extract_static_poses(video: Path) -> np.ndarray:
    """Extrai landmarks normalizados dos frames de mao parada do video.

    Retorna (N, 63): uma linha por frame estatico. Usa a velocidade da mao
    para descartar a transicao de entrada/saida do sinal.
    """
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))
        if res and res.hand_landmarks:
            frames.append([c for lm in res.hand_landmarks[0]
                           for c in (lm.x, lm.y, lm.z)])
    cap.release()

    if len(frames) < 3:
        return np.empty((0, FEAT_DIM))

    seq = normalize_landmarks(np.array(frames)).reshape(-1, N_LANDMARKS, N_COORDS)

    # velocidade media da maquinaria de pontos entre frames consecutivos
    vel = np.linalg.norm(np.diff(seq, axis=0), axis=2).mean(axis=1)
    vel = np.concatenate([[vel[0]], vel])
    keep = vel <= np.quantile(vel, VELOCITY_KEEP_PCT)
    return seq[keep].reshape(-1, FEAT_DIM)


COLETA_DIR = PROJECT_DIR / "coleta_alfabeto"


def carregar_coleta() -> tuple[list, list, list]:
    """coleta_alfabeto/<LETRA>/PessoaNN_<LETRA>_nnn.npy (coletar_alfabeto.py).

    Aceita tambem o formato antigo do alphabet_camera.py (uma pose media 21x3).
    Retorna poses normalizadas, letras e o grupo (pessoa) de cada pose.
    """
    import re
    X, y, g = [], [], []
    if not COLETA_DIR.exists():
        return X, y, g
    for pasta in sorted(p for p in COLETA_DIR.iterdir() if p.is_dir()):
        letra = pasta.name.upper()
        for arq in sorted(pasta.glob("*.npy")):
            arr = np.load(arq).astype(np.float64).reshape(-1, FEAT_DIM)
            m = re.match(r"(Pessoa\d+)_", arq.name)
            grupo = m.group(1) if m else "coleta_antiga"
            for pose in normalize_landmarks(arr):
                X.append(pose); y.append(letra); g.append(grupo)
    return X, y, g


def espelhar(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    out[:, 0::N_COORDS] *= -1
    return out


AUG_COPIAS = 4


def aumentar(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variacoes realistas de uma pose: giro da palma para os lados (yaw),
    para frente/tras (pitch), tamanho e tremor. Simula pessoas diferentes."""
    r = X.reshape(-1, N_LANDMARKS, N_COORDS).copy()
    n = len(r)
    for eixo_a, eixo_b, lim in ((0, 2, 20), (1, 2, 15)):          # yaw (x-z), pitch (y-z)
        a = np.radians(rng.uniform(-lim, lim, n))[:, None]
        pa, pb = r[:, :, eixo_a].copy(), r[:, :, eixo_b].copy()
        r[:, :, eixo_a] = np.cos(a) * pa + np.sin(a) * pb
        r[:, :, eixo_b] = -np.sin(a) * pa + np.cos(a) * pb
    r *= rng.uniform(0.9, 1.1, (n, 1, 1))
    r += rng.normal(0, 0.03, r.shape)
    return r.reshape(-1, FEAT_DIM)


def preparar_treino(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Originais + espelhadas + AUG_COPIAS variacoes de cada -> features v2."""
    base = np.vstack([X, espelhar(X)])
    yb = np.concatenate([y, y])
    partes, ys = [base], [yb]
    for _ in range(AUG_COPIAS):
        partes.append(aumentar(base, rng)); ys.append(yb)
    return alphabet_features(np.vstack(partes)), np.concatenate(ys)


def novo_modelo():
    return RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                  n_jobs=-1, random_state=42)


def main() -> None:
    X, y, groups = [], [], []

    print("\n[1/3] Carregando dados...")
    Xc, yc, gc = carregar_coleta()
    if Xc:
        print(f"  coleta_alfabeto/: {len(Xc)} poses de {len(set(gc))} pessoa(s): "
              f"{', '.join(sorted(set(gc)))}")
    X += Xc; y += yc; groups += gc

    if PALAVRAS_DIR.exists():
        videos = sorted(v for v in PALAVRAS_DIR.glob("*.mp4")
                        if len(v.stem) == 1 and v.stem.isalpha())
        print(f"  palavras2/: {len(videos)} videos de letra")
        for v in videos:
            for pose in extract_static_poses(v):
                X.append(pose); y.append(v.stem.upper()); groups.append("palavras2")
    else:
        print("  palavras2/: nao encontrada (ok, usando so a coleta)")

    if not X:
        raise SystemExit("Nenhum dado. Rode antes:  python coletar_alfabeto.py --pessoa 1")

    X = np.array(X); y = np.array(y); groups = np.array(groups)
    print(f"\n  Total: {len(X)} poses, {len(np.unique(y))} letras, "
          f"{len(np.unique(groups))} fonte(s)")
    for letra in sorted(np.unique(y)):
        print(f"    {letra}: {(y == letra).sum():4d}")

    # ---------------- avaliacao
    print("\n[2/3] Avaliacao...")
    rng = np.random.default_rng(42)
    fontes = sorted(np.unique(groups))
    if len(fontes) >= 2:
        # deixa uma pessoa/fonte inteira de fora: mede se reconhece maos novas
        accs = []
        for f in fontes:
            te = groups == f
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            Xtr, ytr = preparar_treino(X[tr], y[tr], rng)
            sc = StandardScaler().fit(Xtr)
            m = novo_modelo().fit(sc.transform(Xtr), ytr)
            validas = np.isin(y[te], np.unique(ytr))
            acc = accuracy_score(y[te][validas],
                                 m.predict(sc.transform(alphabet_features(X[te][validas]))))
            accs.append(acc)
            print(f"    testando em {f:14s} (treino sem ele): {acc*100:5.1f}%")
        if accs:
            print(f"  >>> MEDIA por pessoa/fonte: {np.mean(accs)*100:.1f}% <<<")
    else:
        idx = rng.permutation(len(X)); cut = int(len(X) * 0.8)
        tr, te = idx[:cut], idx[cut:]
        Xtr, ytr = preparar_treino(X[tr], y[tr], rng)
        sc = StandardScaler().fit(Xtr)
        m = novo_modelo().fit(sc.transform(Xtr), ytr)
        acc = accuracy_score(y[te], m.predict(sc.transform(alphabet_features(X[te]))))
        print(f"  So uma fonte de dados: teste 80/20 (otimista) = {acc*100:.1f}%")
        print("  Para medir com maos novas, colete com mais de uma pessoa (--pessoa 2, 3...).")

    # ---------------- modelo final com tudo (+ espelhado)
    print("\n[3/3] Treinando modelo final...")
    Xall, yall = preparar_treino(X, y, rng)
    scaler = StandardScaler().fit(Xall)
    model = novo_modelo().fit(scaler.transform(Xall), yall)
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler,
                     "letters": sorted(np.unique(y).tolist()),
                     "feature_version": ALPHABET_FEATURE_VERSION}, f)
    print(f"Modelo salvo em: {OUTPUT_MODEL}  ({len(np.unique(y))} letras)")


if __name__ == "__main__":
    main()
