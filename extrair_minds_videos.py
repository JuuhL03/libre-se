#!/usr/bin/env python3
"""
Extrai os pontos da mao direto dos VIDEOS do MINDS-Libras (Zenodo) e salva
em coletados/<palavra>/ no mesmo formato do integrate_minds.py.
Use quando nao tiver o CSV (minds/libras_minds.csv).

1. Baixe os ZIPs dos sinalizadores em https://zenodo.org/records/2667329
   (ex.: Sinalizador01.zip, Sinalizador03.zip, Sinalizador07.zip ...)
2. Coloque os .zip (ou as pastas ja extraidas) dentro de minds_videos/
3. Rode:  python extrair_minds_videos.py

Nao precisa extrair os ZIPs: o script le um video por vez de dentro do .zip.
Pode parar e rodar de novo: videos ja processados sao pulados.

Opcoes:
  --pasta CAMINHO     pasta com os .zip/.mp4 (padrao: minds_videos)
  --processos N       videos em paralelo (padrao: metade dos nucleos)
"""

import argparse
import os
import re
import sys
import tempfile
import time
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "coletados"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"

# Os 20 sinais do MINDS-Libras (sem acento, como as pastas do integrate_minds)
SINAIS = ["acontecer", "aluno", "amarelo", "america", "aproveitar", "bala", "banco",
          "banheiro", "barulho", "cinco", "conhecer", "espelho", "esquina", "filho",
          "maca", "medo", "ruim", "sapo", "vacina", "vontade"]
MIN_HAND_FRAMES = 10          # mesmo filtro do integrate_minds.py
LARGURA_PROCESSAMENTO = 640   # reduz o Full HD antes do MediaPipe (bem mais rapido)
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def sem_acento(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def identificar(caminho: str) -> tuple[str | None, int | None]:
    """Descobre (palavra, sinalizador) pelo caminho do arquivo, sem depender
    de um padrao fixo de nome."""
    txt = sem_acento(caminho.replace("\\", "/"))
    m = re.search(r"sinalizador\s*0*(\d+)", txt)
    sinalizador = int(m.group(1)) if m else None
    nome = txt.rsplit("/", 1)[-1]
    # procura no nome do arquivo primeiro; se nao achar, nas pastas
    for alvo in (nome, txt):
        achados = [s for s in SINAIS if s in alvo]
        if achados:
            # 'banco' x 'banheiro' etc. nao colidem; pega o mais longo por seguranca
            return max(achados, key=len), sinalizador
    return None, sinalizador


def nome_saida(palavra: str, sinalizador: int, caminho: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(caminho).stem)
    # 'Sinalizador' no nome mantem o agrupamento do eval_loso.py / train_lstm.py
    return OUTPUT_DIR / palavra / f"minds_Sinalizador{sinalizador:02d}_{stem}_seg000.npy"


# --------------------------------------------------------------- worker
_landmarker = None


def _init_worker():
    global _landmarker
    import mediapipe as mp
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
        num_hands=1,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)


def _extrair_video(video_path: str) -> np.ndarray:
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > LARGURA_PROCESSAMENTO:
            frame = cv2.resize(frame, (LARGURA_PROCESSAMENTO, int(h * LARGURA_PROCESSAMENTO / w)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))
        if res and res.hand_landmarks:
            frames.append([c for lm in res.hand_landmarks[0] for c in (lm.x, lm.y, lm.z)])
    cap.release()
    return np.array(frames, dtype=np.float32).reshape(-1, 63)


def processar(tarefa: tuple[str, str | None, str, str]) -> tuple[str, str, int]:
    """tarefa = (origem, membro_do_zip ou None, palavra, saida)."""
    origem, membro, palavra, saida = tarefa
    tmp = None
    try:
        if membro is None:
            video = origem
        else:
            with zipfile.ZipFile(origem) as z, z.open(membro) as src:
                fd, tmp = tempfile.mkstemp(suffix=Path(membro).suffix)
                with os.fdopen(fd, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            video = tmp
        seq = _extrair_video(video)
        if len(seq) < MIN_HAND_FRAMES:
            return "poucos_frames", saida, len(seq)
        Path(saida).parent.mkdir(parents=True, exist_ok=True)
        np.save(saida, seq)
        return "ok", saida, len(seq)
    except Exception as e:  # noqa: BLE001
        return f"erro: {e}", saida, 0
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------- main
def listar_tarefas(pasta: Path) -> tuple[list, list]:
    tarefas, ignorados = [], []

    def adicionar(origem, membro, caminho_logico):
        palavra, sinalizador = identificar(caminho_logico)
        if palavra is None or sinalizador is None:
            ignorados.append(caminho_logico)
            return
        tarefas.append((str(origem), membro, palavra,
                        str(nome_saida(palavra, sinalizador, caminho_logico))))

    for arq in sorted(pasta.rglob("*")):
        if arq.suffix.lower() == ".zip":
            with zipfile.ZipFile(arq) as z:
                for membro in z.namelist():
                    if membro.lower().endswith(VIDEO_EXT):
                        adicionar(arq, membro, f"{arq.stem}/{membro}")
        elif arq.suffix.lower() in VIDEO_EXT:
            adicionar(arq, None, str(arq.relative_to(pasta)))
    return tarefas, ignorados


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(PROJECT_DIR / "minds_videos"))
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    pasta = Path(args.pasta)
    if not pasta.exists():
        sys.exit(f"Pasta {pasta} nao encontrada. Crie e coloque os ZIPs do MINDS-Libras nela.")

    print("=" * 60)
    print("EXTRACAO DOS VIDEOS MINDS-LIBRAS")
    print("=" * 60)
    tarefas, ignorados = listar_tarefas(pasta)
    pendentes = [t for t in tarefas if not Path(t[3]).exists()]
    sinalizadores = sorted({int(re.search(r"Sinalizador(\d+)", t[3]).group(1)) for t in tarefas})
    print(f"Videos encontrados: {len(tarefas)} | ja processados: {len(tarefas) - len(pendentes)} "
          f"| a processar: {len(pendentes)}")
    print(f"Sinalizadores: {', '.join(f'{s:02d}' for s in sinalizadores) or '-'}")
    if ignorados:
        print(f"Ignorados (palavra ou sinalizador nao identificado): {len(ignorados)}")
        for x in ignorados[:5]:
            print(f"   ex.: {x}")
    if not pendentes:
        print("Nada a fazer.")
        return

    print(f"Processando com {args.processos} processo(s)...\n")
    t0 = time.time()
    ok = falhas = 0
    with ProcessPoolExecutor(max_workers=args.processos, initializer=_init_worker) as ex:
        futs = [ex.submit(processar, t) for t in pendentes]
        for i, fut in enumerate(as_completed(futs), 1):
            status, saida, n = fut.result()
            if status == "ok":
                ok += 1
            else:
                falhas += 1
                print(f"   [{status}] {Path(saida).name} ({n} frames com mao)")
            if i % 20 == 0 or i == len(pendentes):
                gasto = time.time() - t0
                resta = gasto / i * (len(pendentes) - i)
                print(f"  {i}/{len(pendentes)} videos | {gasto/60:.1f} min | "
                      f"faltam ~{resta/60:.0f} min", flush=True)

    print(f"\nPronto: {ok} salvos, {falhas} descartados/erros.")
    print("Por palavra:")
    for s in SINAIS:
        n = len(list((OUTPUT_DIR / s).glob("minds_*.npy"))) if (OUTPUT_DIR / s).exists() else 0
        if n:
            print(f"   {s:12s} {n}")
    print("\nProximo passo:  python train_rotulado.py")


if __name__ == "__main__":
    main()
