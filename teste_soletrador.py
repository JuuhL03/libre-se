#!/usr/bin/env python3
"""
Teste do soletrador (modo ALFABETO) sem camera: monta sequencias de letras
com poses REAIS dos datasets (coleta_alfabeto/), adiciona tremor de MediaPipe,
transicoes suaves entre letras e o movimento de J, Z, X e H, e confere o texto
que o app montaria.

Uso:
  python teste_soletrador.py                       # tremor 1, 2 e 3 px; 30 e 15 fps
  python teste_soletrador.py --sementes 0,11,22,33 --tremor 1,2,3,4

O tremor e o desvio-padrao do ruido por coordenada, em pixels de uma imagem de
1280 px de largura (MediaPipe em modo IMAGE treme ~1-3 px com a mao parada).
"""

import argparse
from pathlib import Path

import numpy as np

import train_alfabeto as T
from libres_features import forma_mao
from libres_realtime import LetterRecognizer, Soletrador

PROJECT_DIR = Path(__file__).parent
CENTRO = np.array([0.5, 0.55, 0.0])
ESC = 0.12                      # tamanho da mao na imagem
LARGURA_PX = 1280


def para_imagem(pose: np.ndarray, desloc=(0, 0)) -> np.ndarray:
    r = pose.reshape(21, 3) * ESC + CENTRO
    r[:, :2] += np.array(desloc) * ESC
    return r.reshape(-1)


def trajeto(nome: str, dur: float, fps: int) -> np.ndarray:
    """Caminho da mao (em tamanhos de mao) durante a letra com movimento."""
    n = int(dur * fps)
    t = np.linspace(0, 1, n)
    if nome == "J":       # desce e faz a curva
        return np.c_[-np.sin(t * np.pi) * 0.9 * (t > 0.4), t * 1.6]
    if nome == "Z":       # direita, diagonal para baixo-esquerda, direita
        pts = np.array([[0, 0], [1.6, 0], [0, 1.4], [1.6, 1.4]])
        seg = np.minimum((t * 3).astype(int), 2)
        f = t * 3 - seg
        return pts[seg] + (pts[seg + 1] - pts[seg]) * f[:, None]
    if nome == "X":       # uma curva pequena
        return np.c_[np.sin(t * np.pi) * 0.8, -t * 0.7]
    if nome == "X2":      # X sacudido 2 vezes na horizontal (quase nao desce)
        return np.c_[np.sin(t * 4 * np.pi) * 0.6, t * 0.15]
    if nome == "H":       # movimento lateral
        return np.c_[np.sin(t * 2 * np.pi) * 0.9, np.zeros(n)]
    raise ValueError(nome)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sementes", default="0", help="sementes do ruido, ex.: 0,11,22")
    ap.add_argument("--tremor", default="1,2,3", help="tremor em px, ex.: 1,2,3,4")
    ap.add_argument("--fps", default="30,15")
    args = ap.parse_args()
    sementes = [int(x) for x in args.sementes.split(",")]
    tremores = [float(x) for x in args.tremor.split(",")]
    fpss = [int(x) for x in args.fps.split(",")]

    rec = LetterRecognizer(PROJECT_DIR / "modelo_alfabeto.pkl")
    X, y, _ = T.carregar_coleta()
    X, y = np.array(X), np.array(y)

    # para cada letra, uma pose real que o pipeline parado reconhece
    escolha = np.random.default_rng(0)
    s_tmp = Soletrador(rec)
    B = {}
    for l in "ABCDEILORSUVWY":
        idx = np.where(y == l)[0]
        escolha.shuffle(idx)
        for k in idx[:200]:
            raw = para_imagem(X[k])
            letra, conf = s_tmp._estatica(raw, rec.predict_top(raw, len(rec.classes)))
            if letra == l and conf >= 0.6:
                B[l] = X[k]
                break
    pose_ind = next(X[k] for k in np.where(np.isin(y, ["D", "G", "L"]))[0]
                    if forma_mao(para_imagem(X[k])) == "INDICADOR")
    pose_I = next(X[k] for k in np.where(y == "I")[0] if forma_mao(para_imagem(X[k])) == "I")
    pose_U = next(X[k] for k in np.where(y == "U")[0] if forma_mao(para_imagem(X[k])) == "U")
    faltam = set("ABCDEILORSUVWY") - set(B)
    if faltam:
        raise SystemExit(f"sem pose reconhecida para: {''.join(sorted(faltam))}")

    H = 0.6
    casos = [
        ("UVA", [(B["U"], H, None), (B["V"], H, None), (B["A"], H, None)], "UVA"),
        ("RUA", [(B["R"], H, None), (B["U"], H, None), (B["A"], H, None)], "RUA"),
        ("BOLA", [(B["B"], H, None), (B["O"], H, None), (B["L"], H, None), (B["A"], H, None)], "BOLA"),
        ("SOL", [(B["S"], H, None), (B["O"], H, None), (B["L"], H, None)], "SOL"),
        ("LIVRE", [(B["L"], H, None), (B["I"], H, None), (B["V"], H, None), (B["R"], H, None),
                   (B["E"], H, None)], "LIVRE"),
        ("Y parado", [(B["Y"], 1.0, None)], "Y"),
        ("J (I + movimento)", [(pose_I, .3, None), (pose_I, .8, "J"), (pose_I, .5, None)], "J"),
        ("J segurando I antes", [(pose_I, .6, None), (pose_I, .8, "J"), (pose_I, .5, None)], "J"),
        ("I parado", [(pose_I, 1.0, None)], "I"),
        ("Z (zigue-zague)", [(pose_ind, .3, None), (pose_ind, 1.0, "Z"), (pose_ind, .5, None)], "Z"),
        ("X (curva)", [(pose_ind, .3, None), (pose_ind, .6, "X"), (pose_ind, .7, None)], "X"),
        ("X sacudido 2x", [(pose_ind, .3, None), (pose_ind, .8, "X2"), (pose_ind, .7, None)], "X"),
        ("H (U + lateral)", [(pose_U, .3, None), (pose_U, .6, "H"), (pose_U, .7, None)], "H"),
        ("U parado", [(pose_U, 1.0, None)], "U"),
        ("OSSO", [(B["O"], H, None), (B["S"], H, None), ("fora", .3), (B["S"], H, None),
                  (B["O"], H, None)], "OSSO"),
        ("H e abaixa a mao", [(pose_U, .3, None), (pose_U, .6, "H"), ("fora", .4)], "H"),
        ("Z e abaixa a mao", [(pose_ind, .3, None), (pose_ind, 1.0, "Z"), ("fora", .4)], "Z"),
        ("JUIZ", [(pose_I, .3, None), (pose_I, .8, "J"), (pose_I, .3, None), (pose_U, H, None),
                  (pose_I, H, None), (pose_ind, .3, None), (pose_ind, 1.0, "Z"), (pose_ind, .5, None)],
         "JUIZ"),
    ]

    def rodar(roteiro, fps, ruido, rng, transicao=0.25) -> str:
        """roteiro: (pose, segundos, movimento|None) ou ("fora", segundos).
        Entre poses diferentes, a mao passa suavemente de uma para a outra."""
        s = Soletrador(rec)
        txt: list[str] = []
        t = 0.0
        anterior, desloc_ant = None, np.zeros(2)

        def quadro(raw):
            nonlocal t
            info = s.update(raw, t)
            t += 1 / fps
            if info["acao"] == "troca" and txt:
                txt[-1] = info["letra"]
            elif info["acao"] in ("add", "troca"):
                txt.append(info["letra"])

        def tremido(pose, d):
            return para_imagem(pose, d) + rng.normal(0, ruido, 63)

        for passo in roteiro:
            if isinstance(passo[0], str):            # ("fora", segundos)
                for _ in range(int(passo[1] * fps)):
                    quadro(None)
                anterior = None
                continue
            pose, dur, mov = passo
            if anterior is not None and not np.array_equal(anterior, pose):
                n = int(transicao * fps)
                for k in range(1, n + 1):
                    a = k / n
                    quadro(tremido(anterior * (1 - a) + pose * a, desloc_ant))
            caminho = (trajeto(mov, dur, fps) + desloc_ant if mov
                       else np.repeat(desloc_ant[None], int(dur * fps), 0))
            for d in caminho:
                quadro(tremido(pose, d))
            anterior, desloc_ant = pose, caminho[-1]
        if anterior is not None:          # a pessoa continua com a mao ali mais um pouco
            for _ in range(int(0.4 * fps)):
                quadro(tremido(anterior, desloc_ant))
        return "".join(txt)

    total = certos = 0
    for semente in sementes:
        rng = np.random.default_rng(semente)
        for px in tremores:
            for fps in fpss:
                erros = []
                for nome, rot, esperado in casos:
                    saida = rodar(rot, fps, px / LARGURA_PX, rng)
                    if saida != esperado:
                        erros.append(f"{nome} -> {saida!r}")
                ok = len(casos) - len(erros)
                total += len(casos)
                certos += ok
                print(f"semente {semente:>2} | tremor {px:.0f} px | {fps} fps: {ok}/{len(casos)}  "
                      + ("; ".join(erros) if erros else "tudo certo"), flush=True)
    print(f"\nTOTAL: {certos}/{total} sequencias corretas ({100 * certos / total:.1f}%)")


if __name__ == "__main__":
    main()
