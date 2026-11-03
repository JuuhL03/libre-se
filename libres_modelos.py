"""
Salvar e abrir os modelos (.pkl) do projeto.

Os modelos vao para o GitHub, que recusa arquivos acima de 100 MB: o do
alfabeto (Random Forest com 200 arvores) tem 165 MB em pickle puro e 19 MB
comprimido. O joblib (instalado junto com o scikit-learn) abre os dois
formatos: o comprimido e o pickle puro dos modelos antigos.

Como no pickle, abra so modelos de fontes confiaveis (os do proprio projeto).
"""

from __future__ import annotations

from pathlib import Path

COMPRESSAO = ("gzip", 6)        # ~8x menor; abre em menos de 1 s


def salvar_modelo(dados: dict, caminho: Path | str) -> None:
    import joblib
    joblib.dump(dados, caminho, compress=COMPRESSAO)


def carregar_modelo(caminho: Path | str) -> dict:
    import joblib
    return joblib.load(caminho)
