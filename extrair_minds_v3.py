#!/usr/bin/env python3
"""
Extracao v3 dos videos do MINDS-Libras: DUAS maos + POSE (corpo) por quadro.

Uma passada so por video guarda tudo o que o treino v3 precisa:
  maos (T, 2, 21, 3), lado (T, 2), pose (T, 33, 4), aspecto, fps
em minds_v3/<palavra>/minds_SinalizadorNN_<video>.npz

1. Baixe os ZIPs em https://zenodo.org/records/2667329 (Sinalizador01..12.zip)
2. Coloque os .zip (ou as pastas extraidas) em minds_videos/
3. Rode:  python extrair_minds_v3.py

Nao precisa descompactar. Pode parar e rodar de novo: o que ja foi feito e
pulado. Na primeira vez o script baixa sozinho o modelo de pose do MediaPipe
(pose_landmarker_lite.task, ~6 MB).

Opcoes:
  --pasta CAMINHO     pasta com os .zip/.mp4 (padrao: minds_videos)
  --processos N       videos em paralelo (padrao: metade dos nucleos)
  --teste N           processa so N videos e mostra o que saiu (para conferir)
"""

import argparse
import os
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from extrair_minds_videos import (SINAIS, VIDEO_EXT, abrir_membro, eh_lixo_mac, identificar)

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "minds_v3"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
POSE_MODEL = PROJECT_DIR / "pose_landmarker_lite.task"
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
LARGURA_PROCESSAMENTO = 640
MIN_QUADROS_COM_MAO = 10


def garantir_modelo_pose(caminho: Path = POSE_MODEL) -> Path:
    """Baixa o pose_landmarker_lite.task na primeira vez."""
    if caminho.exists() and caminho.stat().st_size > 1_000_000:
        return caminho
    print(f"Baixando o modelo de pose do MediaPipe para {caminho.name}...", flush=True)
    try:
        tmp = caminho.with_suffix(".part")
        urllib.request.urlretrieve(POSE_URL, tmp)
        tmp.replace(caminho)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Nao consegui baixar o modelo de pose ({e}).\n"
                 f"Baixe manualmente em {POSE_URL}\n"
                 f"e salve como {caminho}")
    return caminho


def nome_saida(palavra: str, sinalizador: int, caminho: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(caminho).stem)
    return OUTPUT_DIR / palavra / f"minds_Sinalizador{sinalizador:02d}_{stem}.npz"


# --------------------------------------------------------------- worker
_maos = _pose = None


def criar_detectores(hand_model: Path = HAND_MODEL, pose_model: Path = POSE_MODEL):
    """Hand Landmarker (2 maos) + Pose Landmarker, modo IMAGEM (o mesmo da camera)."""
    import mediapipe as mp
    v = mp.tasks.vision
    maos = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=hand_model.read_bytes()),
        num_hands=2, running_mode=v.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=pose_model.read_bytes()),
        num_poses=1, running_mode=v.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        output_segmentation_masks=False))
    return maos, pose


def detectar_quadro(det_maos, det_pose, rgb) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uma imagem RGB -> (maos (2,21,3), lado (2,), conf (2,), pose (33,4)), NaN = ausente."""
    import mediapipe as mp
    img = mp.Image(mp.ImageFormat.SRGB, data=rgb)
    maos = np.full((2, 21, 3), np.nan, np.float32)
    lado = np.full(2, -1, np.int8)
    conf = np.zeros(2, np.float32)
    res = det_maos.detect(img)
    if res and res.hand_landmarks:
        for k, mao in enumerate(res.hand_landmarks[:2]):
            maos[k] = [(p.x, p.y, p.z) for p in mao]
            if res.handedness and len(res.handedness) > k and res.handedness[k]:
                cat = res.handedness[k][0]
                lado[k] = 1 if cat.category_name.lower().startswith("r") else 0
                conf[k] = cat.score
    pose = np.full((33, 4), np.nan, np.float32)
    rp = det_pose.detect(img)
    if rp and rp.pose_landmarks:
        pose[:] = [(p.x, p.y, p.z, getattr(p, "visibility", 1.0) or 0.0) for p in rp.pose_landmarks[0]]
    return maos, lado, conf, pose


class DetectorV3:
    """Para a camera e a coleta: duas maos + pose com os MESMOS detectores (e
    limiares) da extracao dos videos de treino. A pose (nariz e ombros) muda
    devagar: roda a cada `pose_cada` segundos e o ultimo resultado e reaproveitado."""

    def __init__(self, hand_model: Path = HAND_MODEL, pose_model: Path | None = None,
                 pose_cada: float = 0.2) -> None:
        self.maos, self.pose = criar_detectores(hand_model, pose_model or garantir_modelo_pose())
        self.pose_cada = pose_cada
        self._pose: np.ndarray | None = None
        self._t_pose = -1e9

    def detectar(self, imagem, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """imagem RGB (numpy) ou mp.Image -> (maos (2,21,3), lado (2,), pose (33,4)), NaN = ausente."""
        import mediapipe as mp
        img = mp.Image(mp.ImageFormat.SRGB, data=imagem) if isinstance(imagem, np.ndarray) else imagem
        maos = np.full((2, 21, 3), np.nan)
        lado = np.full(2, -1)
        res = self.maos.detect(img)
        if res and res.hand_landmarks:
            for k, mao in enumerate(res.hand_landmarks[:2]):
                maos[k] = [(q.x, q.y, q.z) for q in mao]
                if res.handedness and len(res.handedness) > k and res.handedness[k]:
                    lado[k] = 1 if res.handedness[k][0].category_name.lower().startswith("r") else 0
        if self._pose is None or t - self._t_pose >= self.pose_cada:
            self._pose = np.full((33, 4), np.nan)
            rp = self.pose.detect(img)
            if rp and rp.pose_landmarks:
                self._pose[:] = [(q.x, q.y, q.z, getattr(q, "visibility", 1.0) or 0.0)
                                 for q in rp.pose_landmarks[0]]
            self._t_pose = t
        return maos, lado, self._pose.copy()


def _init_worker(hand_model: str, pose_model: str):
    global _maos, _pose
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    _maos, _pose = criar_detectores(Path(hand_model), Path(pose_model))


def extrair_video(video_path: str, det_maos=None, det_pose=None) -> dict:
    import cv2
    det_maos, det_pose = det_maos or _maos, det_pose or _pose
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    M, L, C, P = [], [], [], []
    aspecto = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > LARGURA_PROCESSAMENTO:
            frame = cv2.resize(frame, (LARGURA_PROCESSAMENTO, int(h * LARGURA_PROCESSAMENTO / w)))
        aspecto = frame.shape[1] / frame.shape[0]
        m, l, c, p = detectar_quadro(det_maos, det_pose, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        M.append(m); L.append(l); C.append(c); P.append(p)
    cap.release()
    return {"maos": np.array(M, np.float32).reshape(-1, 2, 21, 3),
            "lado": np.array(L, np.int8).reshape(-1, 2),
            "conf_lado": np.array(C, np.float32).reshape(-1, 2),
            "pose": np.array(P, np.float32).reshape(-1, 33, 4),
            "aspecto": np.float32(aspecto or 16 / 9), "fps": np.float32(fps)}


def processar(tarefa: tuple) -> tuple[str, str, int]:
    origem, membro, palavra, saida = tarefa
    tmp = None
    try:
        if membro is None:
            video = origem
        else:
            with zipfile.ZipFile(origem) as z, abrir_membro(z, origem, membro) as src:
                fd, tmp = tempfile.mkstemp(suffix=Path(membro).suffix)
                with os.fdopen(fd, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            video = tmp
        d = extrair_video(video)
        n = int((~np.isnan(d["maos"][:, :, 0, 0])).any(axis=1).sum())
        if n < MIN_QUADROS_COM_MAO:
            return "poucos_quadros", saida, n
        Path(saida).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(saida, **d)
        return "ok", saida, n
    except Exception as e:  # noqa: BLE001
        return f"erro: {e}", saida, 0
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def listar_tarefas(pasta: Path) -> tuple[list, list]:
    tarefas, ignorados = [], []

    def adicionar(origem, membro, caminho_logico):
        palavra, sinalizador = identificar(caminho_logico)
        if palavra is None or sinalizador is None:
            ignorados.append(caminho_logico)
            return
        tarefas.append((str(origem), membro, palavra, str(nome_saida(palavra, sinalizador, caminho_logico))))

    for arq in sorted(pasta.rglob("*")):
        if arq.suffix.lower() == ".zip":
            with zipfile.ZipFile(arq) as z:
                for membro in z.namelist():
                    if membro.lower().endswith(VIDEO_EXT) and not eh_lixo_mac(membro):
                        adicionar(arq, membro, f"{arq.stem}/{membro}")
        elif arq.suffix.lower() in VIDEO_EXT and not eh_lixo_mac(str(arq)):
            adicionar(arq, None, str(arq.relative_to(pasta)))
    return tarefas, ignorados


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(PROJECT_DIR / "minds_videos"))
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--teste", type=int, default=0)
    args = ap.parse_args()

    pasta = Path(args.pasta)
    if not pasta.exists():
        sys.exit(f"Pasta {pasta} nao encontrada. Crie e coloque os ZIPs do MINDS-Libras nela.")
    garantir_modelo_pose()

    print("=" * 60)
    print("EXTRACAO v3 (duas maos + pose) DOS VIDEOS MINDS-LIBRAS")
    print("=" * 60)
    tarefas, ignorados = listar_tarefas(pasta)
    pendentes = [t for t in tarefas if not Path(t[3]).exists()]
    sinalizadores = sorted({int(re.search(r"Sinalizador(\d+)", t[3]).group(1)) for t in tarefas})
    print(f"Videos: {len(tarefas)} | ja feitos: {len(tarefas) - len(pendentes)} | a fazer: {len(pendentes)}")
    print(f"Sinalizadores: {', '.join(f'{s:02d}' for s in sinalizadores) or '-'}")
    if ignorados:
        print(f"Ignorados (palavra ou sinalizador nao identificado): {len(ignorados)}")
    if args.teste:
        pendentes = pendentes[:args.teste]
        args.processos = 1
    if not pendentes:
        print("Nada a fazer.")
        return

    print(f"Processando com {args.processos} processo(s)...\n")
    t0 = time.time()
    ok = falhas = 0
    with ProcessPoolExecutor(max_workers=args.processos, initializer=_init_worker,
                             initargs=(str(HAND_MODEL), str(POSE_MODEL))) as ex:
        futs = [ex.submit(processar, t) for t in pendentes]
        for i, fut in enumerate(as_completed(futs), 1):
            status, saida, n = fut.result()
            if status == "ok":
                ok += 1
                if args.teste:
                    d = np.load(saida)
                    duas = int((~np.isnan(d["maos"][:, :, 0, 0])).all(axis=1).sum())
                    com_pose = int((~np.isnan(d["pose"][:, 0, 0])).sum())
                    print(f"   {Path(saida).name}: {len(d['maos'])} quadros, {n} com mao, "
                          f"{duas} com as duas, {com_pose} com pose")
            else:
                falhas += 1
                print(f"   [{status}] {Path(saida).name} ({n} quadros com mao)")
            if i % 20 == 0 or i == len(pendentes):
                gasto = time.time() - t0
                print(f"  {i}/{len(pendentes)} videos | {gasto / 60:.1f} min | "
                      f"faltam ~{gasto / i * (len(pendentes) - i) / 60:.0f} min", flush=True)

    print(f"\nPronto: {ok} salvos, {falhas} descartados/erros.")
    print("Por palavra:")
    for s in SINAIS:
        n = len(list((OUTPUT_DIR / s).glob("*.npz"))) if (OUTPUT_DIR / s).exists() else 0
        if n:
            print(f"   {s:12s} {n}")
    print("\nProximo passo:  python avaliar_v3.py   (e depois python treino_v3.py)")


if __name__ == "__main__":
    main()
