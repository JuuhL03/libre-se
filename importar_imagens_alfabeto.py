#!/usr/bin/env python3
"""
Importa IMAGENS de datasets publicos do alfabeto (Roboflow, Mendeley, Kaggle...)
para coleta_alfabeto/, no mesmo formato do coletar_alfabeto.py.

Formatos aceitos (detectados sozinhos):
  1) Pasta por letra:      <pasta>/A/*.jpg, <pasta>/B/*.jpg ...
     (Mendeley "Brazilian Sign Language Alphabet"; Roboflow export "Folder Structure")
  2) YOLO (Roboflow):      <pasta>/data.yaml + train|valid|test/images + labels

Uso:
  python importar_imagens_alfabeto.py --pasta C:\\Downloads\\alfabeto_v2 --nome roboflow
  python importar_imagens_alfabeto.py --pasta C:\\Downloads\\mendeley --nome mendeley
  python train_alfabeto.py

--nome identifica a fonte: na avaliacao do train_alfabeto.py cada fonte conta
como uma "pessoa" (testa-se em cada fonte com o modelo treinado sem ela).
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_DIR = Path(__file__).parent
COLETA_DIR = PROJECT_DIR / "coleta_alfabeto"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LETRAS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def letra_de(nome: str) -> str | None:
    """'A', 'a', 'letra_A', 'A-libras' -> 'A'."""
    n = nome.strip().upper()
    if n in LETRAS:
        return n
    m = re.fullmatch(r"(?:LETRA[_\- ]?)?([A-Z])(?:[_\- ].*)?", n)
    return m.group(1) if m else None


def listar_pasta_por_letra(raiz: Path) -> list[tuple[Path, str]]:
    itens = []
    for pasta in raiz.rglob("*"):
        if pasta.is_dir() and (letra := letra_de(pasta.name)):
            for img in pasta.iterdir():
                if img.suffix.lower() in IMG_EXT:
                    itens.append((img, letra))
    return itens


def listar_yolo(raiz: Path) -> list[tuple[Path, str]]:
    yaml_txt = (raiz / "data.yaml").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"names\s*:\s*\[(.*?)\]", yaml_txt, re.S)
    if m:
        nomes = [x.strip().strip("'\"") for x in m.group(1).split(",")]
    else:   # formato em lista ( - A \n - B ) ou dicionario (0: A)
        bloco = yaml_txt.split("names", 1)[1]
        nomes = re.findall(r"^\s*(?:-\s*|\d+\s*:\s*)['\"]?([^'\"\n]+?)['\"]?\s*$", bloco, re.M)
    itens = []
    for lbl in raiz.rglob("labels/*.txt"):
        linhas = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if not linhas:
            continue
        letra = letra_de(nomes[int(linhas[0][0])]) if int(linhas[0][0]) < len(nomes) else None
        if not letra:
            continue
        pasta_img = lbl.parent.parent / "images"
        for ext in IMG_EXT:
            img = pasta_img / (lbl.stem + ext)
            if img.exists():
                itens.append((img, letra))
                break
    return itens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", required=True)
    ap.add_argument("--nome", required=True, help="nome curto da fonte, ex.: roboflow, mendeley")
    args = ap.parse_args()
    raiz = Path(args.pasta)
    fonte = re.sub(r"[^A-Za-z0-9]+", "", args.nome) or "fonte"
    if not raiz.exists():
        sys.exit(f"Pasta nao encontrada: {raiz}")

    itens = listar_yolo(raiz) if (raiz / "data.yaml").exists() else listar_pasta_por_letra(raiz)
    if not itens:
        sys.exit("Nenhuma imagem encontrada. Esperado: pastas com o nome da letra (A, B, ...) "
                 "ou export YOLO com data.yaml.")
    print(f"{len(itens)} imagens de {len({l for _, l in itens})} letras em {raiz}")

    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
            num_hands=1, running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))

    por_letra: dict[str, list[np.ndarray]] = {}
    sem_mao = 0
    for i, (img_path, letra) in enumerate(itens, 1):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) < 320:          # imagens pequenas (ex.: 200x200): amplia e da margem
            esc = 320 / max(h, w)
            img = cv2.resize(img, (int(w * esc), int(h * esc)))
            img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_REPLICATE)
        res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if res and res.hand_landmarks:
            por_letra.setdefault(letra, []).append(
                np.array([c for p in res.hand_landmarks[0] for c in (p.x, p.y, p.z)], dtype=np.float32))
        else:
            sem_mao += 1
        if i % 200 == 0:
            print(f"  {i}/{len(itens)} imagens...", flush=True)

    total = 0
    for letra, poses in sorted(por_letra.items()):
        pasta = COLETA_DIR / letra
        pasta.mkdir(parents=True, exist_ok=True)
        np.save(pasta / f"{fonte}_{letra}_000.npy", np.array(poses))
        total += len(poses)
        print(f"   {letra}: {len(poses)}")
    print(f"\nImportadas {total} poses ({sem_mao} imagens sem mao detectada) como fonte '{fonte}'.")
    print("Proximo passo:  python train_alfabeto.py")


if __name__ == "__main__":
    main()
