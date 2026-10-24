#!/usr/bin/env python3
"""
Escolhe as melhores palavras para a demonstracao.

Para cada palavra do vocabulario mostra:
  - acerto com uma pessoa que o modelo nunca viu (avaliacao por sinalizante);
  - % de quadros com DUAS maos nos videos do MINDS (o prototipo acompanha uma
    mao so: palavras de duas maos nao tem como dar certo).
E sugere as palavras e os comandos para gravar/treinar so com elas.

Uso:  python analisar_palavras.py            (usa coletados/ e minds_videos/)
      python analisar_palavras.py --n 8      (quantas palavras sugerir)
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent
VIDEOS_DIR = PROJECT_DIR / "minds_videos"
LIMIAR_DUAS_MAOS = 0.30     # acima disso a palavra provavelmente usa as duas maos
LIMIAR_ACERTO = 0.70

_lm = None


def _init():
    global _lm
    import mediapipe as mp
    import extrair_minds_videos as ex
    _lm = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=ex.HAND_MODEL.read_bytes()),
            num_hands=2, running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5))


def _contar_maos(tarefa):
    """Fracao de quadros (com mao) em que aparecem 2 maos."""
    import tempfile, zipfile
    import cv2
    import mediapipe as mp
    import extrair_minds_videos as ex
    origem, membro, palavra, _ = tarefa
    tmp = None
    try:
        if membro is None:
            video = origem
        else:
            with zipfile.ZipFile(origem) as z, ex.abrir_membro(z, origem, membro) as src:
                fd, tmp = tempfile.mkstemp(suffix=Path(membro).suffix)
                with os.fdopen(fd, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            video = tmp
        cap = cv2.VideoCapture(video)
        com_mao = duas = i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % 2:
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            res = _lm.detect(mp.Image(mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            n = len(res.hand_landmarks) if res and res.hand_landmarks else 0
            com_mao += n > 0
            duas += n >= 2
        cap.release()
        return palavra, (duas / com_mao) if com_mao else None
    except Exception:  # noqa: BLE001
        return palavra, None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def duas_maos_por_palavra() -> dict:
    if not VIDEOS_DIR.exists():
        return {}
    import extrair_minds_videos as ex
    tarefas, _ = ex.listar_tarefas(VIDEOS_DIR)
    escolhidas = {}
    for t in tarefas:                      # 1 video por (palavra, sinalizador)
        sid = re.search(r"Sinalizador(\d+)", t[3]).group(1)
        escolhidas.setdefault((t[2], sid), t)
    print(f"\nContando maos em {len(escolhidas)} videos do MINDS...", flush=True)
    res = defaultdict(list)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2), initializer=_init) as exe:
        for palavra, r in exe.map(_contar_maos, escolhidas.values()):
            if r is not None:
                res[palavra].append(r)
    return {p: float(np.mean(v)) for p, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="quantas palavras sugerir")
    args, resto = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + resto       # eval_loso le os proprios argumentos

    import eval_loso
    print("Avaliando cada palavra com uma pessoa fora do treino...")
    linhas = eval_loso.main()              # [(acerto, palavra, confusao), ...]
    maos = duas_maos_por_palavra()

    print("\n" + "=" * 72)
    print(f"{'palavra':12s} {'acerto':>7s} {'2 maos':>7s}   observacao")
    print("-" * 72)
    boas = []
    for acc, palavra, conf in linhas:
        dm = maos.get(palavra)
        dm_txt = f"{dm*100:5.0f}%" if dm is not None else "   ?  "
        obs = []
        if dm is not None and dm >= LIMIAR_DUAS_MAOS:
            obs.append("usa as DUAS maos")
        if acc < LIMIAR_ACERTO:
            obs.append("acerto baixo" + conf)
        if not obs:
            obs.append("BOA para a demonstracao")
            boas.append(palavra)
        print(f"{palavra:12s} {acc*100:6.0f}% {dm_txt}   {'; '.join(obs)}")
    print("=" * 72)
    if not maos:
        print("(nao deu para contar as maos: falta a pasta minds_videos/ com os ZIPs do MINDS)")

    sugestao = boas[:args.n] if len(boas) >= 3 else [p for _, p, _ in linhas][:args.n]
    lista = ",".join(sugestao)
    print(f"\nSugestao de vocabulario para a demonstracao ({len(sugestao)} palavras):")
    print(f"  {lista}")
    print("\nAssista aos videos dessas palavras no MINDS e depois:")
    print(f"  python coletar_palavras.py --sinalizador 50 --palavras {lista} --takes 10")
    print(f"  python train_rotulado.py --palavras {lista}")
    print(f"  python eval_loso.py --palavras {lista}")
    print("  python libras_camera.py")


if __name__ == "__main__":
    main()
