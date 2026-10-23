#!/usr/bin/env python3
"""
Extrai sequencias de landmarks do dataset MINDS e salva como .npy
pra usar no treino junto com os dados coletados.
"""
import sys, os, csv, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_DIR = Path(__file__).parent
MINDS_CSV = PROJECT_DIR / "minds" / "libras_minds.csv"
OUTPUT_DIR = PROJECT_DIR / "coletados"

N_FRAMES = 30
N_LANDMARKS = 21
N_COORDS = 3
FEATS_PER_FRAME = N_LANDMARKS * N_COORDS

print("="*50)
print("INTEGRACAO DATASET MINDS")
print("="*50)

if not MINDS_CSV.exists():
    print(f"ERRO: {MINDS_CSV} nao encontrado")
    exit(1)

print("Lendo CSV...")
videos = defaultdict(lambda: {'frames': [], 'category': ''})

with open(MINDS_CSV) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        vname = row['video_name']
        videos[vname]['category'] = row['category']
        hand = []
        missing_hand = row.get('missing_hand', 'False') == 'True'
        if not missing_hand:
            for j in range(N_LANDMARKS):
                x = float(row.get(f'hand_0_{j}_x', 0))
                y = float(row.get(f'hand_0_{j}_y', 0))
                z = float(row.get(f'hand_0_{j}_z', 0))
                hand.extend([x, y, z])
        else:
            hand = [0.0] * FEATS_PER_FRAME
        videos[vname]['frames'].append(hand)

print(f"Total videos: {len(videos)}")

print("\nProcessando videos...")

# Remove os segmentos gerados pela versao antiga (blocos de 30 frames).
# Sao arquivos derivados do CSV: nada se perde, sao regerados abaixo.
antigos = list(OUTPUT_DIR.glob("*/minds_*_seg*.npy")) if OUTPUT_DIR.exists() else []
if antigos:
    print(f"  removendo {len(antigos)} segmentos antigos (minds_*_seg*.npy)")
    for f in antigos:
        f.unlink()

total_salvos = 0
classes_salvas = set()

# v2: UM arquivo por video, com o sinal INTEIRO (todos os frames com mao).
# A versao antiga cortava o video em blocos de 30 frames e dava o rotulo da
# palavra a todos -- inclusive a blocos que eram so a mao subindo/descendo.
# O treino agora reamostra cada sinal completo para 30 frames.
for vname, vdata in videos.items():
    label = vdata['category'].lower().replace(' ', '_')
    frames = np.array(vdata['frames'])

    valid_frames = frames[~np.all(frames == 0, axis=1)]
    if len(valid_frames) < 10:
        continue

    pasta = OUTPUT_DIR / label
    pasta.mkdir(parents=True, exist_ok=True)

    # nome mantem o sufixo _seg000 para o agrupamento por sinalizador
    # (eval_loso / train_lstm) continuar funcionando
    fname = pasta / f"minds_{Path(vname).stem}_seg000.npy"
    np.save(fname, valid_frames.astype(np.float32))
    total_salvos += 1
    classes_salvas.add(label)

print(f"\nResumo:")
print(f"  Total videos salvos: {total_salvos}")
print(f"  Classes: {len(classes_salvas)}")
for c in sorted(classes_salvas):
    pasta = OUTPUT_DIR / c
    n = len(list(pasta.glob("minds_*.npy")))
    print(f"    {c}: {n} videos")
