"""
Alfabeto calibrado: as letras da PROPRIA pessoa somadas ao modelo geral.

O modelo do alfabeto foi treinado com fotos de outras pessoas; com uma mao que ele
nunca viu, letras parecidas (A/S/E, M/N, P/Q, G/L) se confundem. Na calibracao a
pessoa segura cada letra por ~1 s e os pontos viram exemplos dela. Para reconhecer,
a pose e comparada com esses exemplos (vizinhos mais proximos, no mesmo espaco de
caracteristicas do modelo) e o resultado e misturado com a probabilidade do modelo
geral. Se a pose nao se parece com nenhum exemplo (uma transicao, uma letra que ela
nao calibrou), vale so o modelo geral.

Salvo em calibracao/<perfil>/alfabeto.npz (so os 21 pontos de cada exemplo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from libres_features import FEAT_DIM, alphabet_features

PESO_PESSOAL = 0.75        # quanto o exemplo da pessoa pesa contra o modelo geral
K_VIZINHOS = 7
EXEMPLOS_POR_LETRA = 20    # quadros guardados por letra na calibracao (~1 s parado)


class AlfabetoPessoal:
    """Mesmo jeito de usar do LetterRecognizer (classes, predict_top, predict)."""

    def __init__(self, base, exemplos: dict[str, np.ndarray]) -> None:
        self.base = base
        self.classes = list(base.classes)
        self.letras = [l for l in self.classes if l in exemplos and len(exemplos[l])]
        if not self.letras:
            raise ValueError("calibracao do alfabeto vazia")
        X = np.vstack([np.asarray(exemplos[l], dtype=np.float64).reshape(-1, FEAT_DIM) for l in self.letras])
        self.rotulos = np.concatenate([[l] * len(np.asarray(exemplos[l]).reshape(-1, FEAT_DIM))
                                       for l in self.letras])
        self.F = self._feat(X)
        # escala das distancias: o vizinho mais proximo de OUTRA letra (tipico da pessoa)
        D = np.linalg.norm(self.F[:, None] - self.F[None], axis=2)
        outra = self.rotulos[:, None] != self.rotulos[None]
        Dm = np.where(outra, D, np.inf).min(axis=1)
        Dm = Dm[np.isfinite(Dm)]
        self.escala = float(np.median(Dm)) if len(Dm) else 1.0
        self.ultimo_confiante = False

    def _feat(self, raw: np.ndarray) -> np.ndarray:
        return self.base.scaler.transform(alphabet_features(np.asarray(raw).reshape(-1, FEAT_DIM)))

    def predict_top(self, landmarks: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        geral = dict(self.base.predict_top(landmarks, len(self.classes)))
        f = self._feat(landmarks)[0]
        d = np.linalg.norm(self.F - f, axis=1)
        idx = np.argsort(d)[:K_VIZINHOS]
        perto = d[idx[0]] <= 1.5 * self.escala          # parece com algum exemplo da pessoa
        pessoal = {}
        if perto:
            pesos = 1.0 / (d[idx] + 0.05 * self.escala)
            for i, w in zip(idx, pesos):
                pessoal[self.rotulos[i]] = pessoal.get(self.rotulos[i], 0.0) + float(w)
            tot = sum(pessoal.values())
            pessoal = {l: v / tot for l, v in pessoal.items()}
        w = PESO_PESSOAL if perto else 0.0
        final = {l: (1 - w) * geral.get(l, 0.0) + w * pessoal.get(l, 0.0) for l in self.classes}
        ordem = sorted(final.items(), key=lambda x: -x[1])
        # com o exemplo da pessoa claramente a favor, as regras de forma do soletrador
        # (feitas para quem nao calibrou) nao trocam a letra
        self.ultimo_confiante = bool(perto and pessoal and max(pessoal.values()) >= 0.7)
        return ordem[:k]

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        return self.predict_top(landmarks, 1)[0]


def arquivo(pasta: Path) -> Path:
    return Path(pasta) / "alfabeto.npz"


def salvar(pasta: Path, exemplos: dict[str, list]) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    dados = {l: np.asarray(v, dtype=np.float32).reshape(-1, FEAT_DIM) for l, v in exemplos.items() if len(v)}
    np.savez_compressed(arquivo(pasta), **dados)
    return arquivo(pasta)


def carregar(pasta: Path) -> Optional[dict[str, np.ndarray]]:
    a = arquivo(pasta)
    if not a.exists():
        return None
    with np.load(a) as d:
        return {k: d[k].astype(np.float64) for k in d.files}


def apagar(pasta: Path) -> None:
    a = arquivo(pasta)
    if a.exists():
        a.unlink()
