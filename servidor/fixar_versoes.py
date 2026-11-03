#!/usr/bin/env python3
"""
Fixa no servidor as MESMAS versoes de numpy, scikit-learn e xgboost do .venv
em que os modelos foram treinados (os .pkl do scikit-learn so abrem direito na
versao em que foram salvos). Rode no .venv do projeto, depois de treinar:

  python servidor/fixar_versoes.py

Grava servidor/requirements-modelos.txt (o Dockerfile instala por ele).
"""

import importlib.metadata as md
import platform
from pathlib import Path

ARQ = Path(__file__).resolve().parent / "requirements-modelos.txt"


def versao(pacote: str) -> str:
    try:
        return md.version(pacote)
    except md.PackageNotFoundError:
        raise SystemExit(f"{pacote} nao esta instalado neste ambiente. Ative o .venv do projeto e rode de novo.")


def main() -> None:
    linhas = ["# Gerado por servidor/fixar_versoes.py: as versoes do .venv em que os modelos foram treinados",
              f"# (Python {platform.python_version()} em {platform.system()})"]
    linhas.append(f"numpy=={versao('numpy')}")
    linhas.append(f"scikit-learn=={versao('scikit-learn')}")
    xgb = versao("xgboost")
    partes = [int(p) for p in xgb.split(".")[:3] if p.isdigit()]
    # no Linux, o pacote "xgboost" puxa ~300 MB de bibliotecas de GPU; o "xgboost-cpu" (a partir da 2.1.1)
    # e a mesma biblioteca sem a GPU e abre os mesmos modelos
    linhas.append(f"xgboost-cpu=={xgb}" if tuple(partes) >= (2, 1, 1) else f"xgboost=={xgb}")
    ARQ.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"gravado {ARQ.name}:")
    print("\n".join("  " + l for l in linhas[2:]))
    if not platform.python_version().startswith("3.12"):
        print(f"  (o servidor usa Python 3.12; aqui e {platform.python_version()}: tudo bem, os modelos abrem igual)")


if __name__ == "__main__":
    main()
