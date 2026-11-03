#!/usr/bin/env python3
"""
Baixa o que a pagina precisa e nao fica no repositorio (uma vez so):

  web/vendor/tasks-vision/   MediaPipe para navegador (@mediapipe/tasks-vision,
                             licenca Apache-2.0), da versao abaixo, conferida
                             pelo hash publicado no npm
  pose_landmarker_lite.task  modelo do corpo (se ainda nao estiver na pasta do
                             projeto; o extrair_minds_v3.py ja baixa ele)

Uso (na pasta do projeto):  python servidor/preparar_web.py
O Dockerfile roda o mesmo script na hora de publicar.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
VERSAO_TASKS_VISION = "0.10.32"
VENDOR = RAIZ / "web" / "vendor" / "tasks-vision"
ARQUIVOS_VENDOR = ["vision_bundle.mjs", "wasm/vision_wasm_internal.js", "wasm/vision_wasm_internal.wasm",
                   "wasm/vision_wasm_nosimd_internal.js", "wasm/vision_wasm_nosimd_internal.wasm"]
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
POSE_DESTINOS = [RAIZ / "pose_landmarker_lite.task", RAIZ / "servidor" / "modelos" / "pose_landmarker_lite.task"]


def baixar(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def preparar_vendor() -> None:
    if all((VENDOR / a).exists() for a in ARQUIVOS_VENDOR):
        print(f"  tasks-vision {VERSAO_TASKS_VISION}: ja esta em web/vendor/")
        return
    meta = json.loads(baixar(f"https://registry.npmjs.org/@mediapipe/tasks-vision/{VERSAO_TASKS_VISION}"))
    dist = meta["dist"]
    print(f"  baixando {dist['tarball']} ...")
    dados = baixar(dist["tarball"])
    algoritmo, esperado = dist["integrity"].split("-", 1)
    if algoritmo != "sha512" or base64.b64encode(hashlib.sha512(dados).digest()).decode() != esperado:
        sys.exit("  ERRO: o arquivo baixado nao confere com o hash do npm. Nada foi gravado.")
    with tarfile.open(fileobj=io.BytesIO(dados), mode="r:gz") as tar:
        for nome in ARQUIVOS_VENDOR:
            membro = tar.getmember(f"package/{nome}")
            destino = VENDOR / nome
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(tar.extractfile(membro).read())
    (VENDOR / "LEIA.txt").write_text(
        f"@mediapipe/tasks-vision {VERSAO_TASKS_VISION} (Google, licenca Apache-2.0), baixado do npm por "
        "servidor/preparar_web.py.\n", encoding="utf-8")
    print(f"  tasks-vision {VERSAO_TASKS_VISION}: ok ({len(ARQUIVOS_VENDOR)} arquivos)")


def modelo_valido(caminho: Path, membros: set[str]) -> bool:
    try:
        with zipfile.ZipFile(caminho) as z:
            return membros <= set(z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def preparar_pose() -> None:
    membros = {"pose_detector.tflite", "pose_landmarks_detector.tflite"}
    for c in POSE_DESTINOS:
        if c.exists() and modelo_valido(c, membros):
            print(f"  modelo do corpo: ja existe ({c.relative_to(RAIZ)})")
            return
    destino = POSE_DESTINOS[1]
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  baixando o modelo do corpo ({POSE_URL}) ...")
    tmp = destino.with_suffix(".baixando")
    tmp.write_bytes(baixar(POSE_URL))
    if not modelo_valido(tmp, membros):
        tmp.unlink(missing_ok=True)
        sys.exit("  ERRO: o modelo do corpo baixado nao e valido.")
    tmp.replace(destino)
    print(f"  modelo do corpo: ok ({destino.relative_to(RAIZ)})")


def conferir_maos() -> None:
    c = RAIZ / "hand_landmarker.task"
    if not modelo_valido(c, {"hand_detector.tflite", "hand_landmarks_detector.tflite"}):
        sys.exit("  ERRO: hand_landmarker.task (modelo das maos) nao esta na pasta do projeto.")
    print("  modelo das maos: ok")


if __name__ == "__main__":
    print("Preparando a versao web:")
    conferir_maos()
    preparar_pose()
    preparar_vendor()
    print("Pronto. Para testar no computador:  uvicorn servidor.app:app --port 8000")
