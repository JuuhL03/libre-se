#!/usr/bin/env python3
"""Alfabeto com fotos de uma origem que o modelo nao viu (secao 3.6.2 do TCC).

Em cada rodada, o Random Forest do alfabeto (mesmo treino de train_alfabeto.py: espelhamento,
aumento de dados e semente 42) e treinado sem as poses de uma origem e testado nelas. Cada pose
e classificada de dois jeitos:
  - sem regras: a letra parada mais provavel do modelo (como em predict_top, que testa a mao
    como esta e espelhada e fica com a mais confiante);
  - com regras: a mesma decisao do soletrador (Soletrador._estatica), com as regras de geometria
    do Y, de M/N/Q para baixo, de U/R/V e de A/S/E pelo polegar.
O resultado principal e a media do acerto por letra. No fim, a recusa por confianca: o
soletrador so escreve uma letra parada com confianca de pelo menos 55% (conf_min); abaixo disso
ela e tratada como "incerto". A conta mostra quantas poses seriam recusadas e o acerto das aceitas
(nas fotos nao ha a exigencia de 0,3 s de estabilidade, que depende do video). As poses do 'alfalibras' (5 fotos isoladas)
ficam fora. Nada e salvo: o modelo do aplicativo continua sendo o de train_alfabeto.py.

Uso: python avaliar_alfabeto.py
"""
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler

import train_alfabeto as t
import libres_realtime as lr
from libres_features import alphabet_features, mirror_raw

IGNORAR = {"alfalibras"}


class _Rec:
    def __init__(self, classes):
        self.classes = list(classes)


def main() -> None:
    X, y, g = t.carregar_coleta()
    X, y, g = np.array(X), np.array(y), np.array(g)
    k = ~np.isin(g, list(IGNORAR))
    X, y, g = X[k], y[k], g[k]
    print(f"{len(X)} poses, {len(np.unique(y))} letras, origens: {', '.join(sorted(set(g)))}")

    rng = np.random.default_rng(42)
    pred = {"sem regras": [], "com regras": []}
    confs = []
    real, orig = [], []
    for f in sorted(set(g)):
        te, tr = g == f, g != f
        Xtr, ytr = t.preparar_treino(X[tr], y[tr], rng)
        sc = StandardScaler().fit(Xtr)
        m = t.novo_modelo().fit(sc.transform(Xtr), ytr)
        v = np.isin(y[te], np.unique(ytr))          # letras que so existem nesta origem ficam fora
        Xt, yt = X[te][v], y[te][v]
        cl = np.array([str(c) for c in m.classes_])
        p1 = m.predict_proba(sc.transform(alphabet_features(Xt)))
        p2 = m.predict_proba(sc.transform(alphabet_features(mirror_raw(Xt))))
        P = np.where((p2.max(1) > p1.max(1))[:, None], p2, p1)
        sol = lr.Soletrador.__new__(lr.Soletrador)
        sol.rec = _Rec(cl)
        for i in range(len(Xt)):
            top = [(cl[j], float(P[i][j])) for j in np.argsort(P[i])[::-1]]
            est = [(l, c) for l, c in top if l not in lr.LETRAS_DINAMICAS] or top
            pred["sem regras"].append(est[0][0])
            letra, conf = sol._estatica(Xt[i], top)
            pred["com regras"].append(letra)
            confs.append(conf)
        real += list(yt); orig += [f] * len(yt)
        print(f"  {f:10s}: {te.sum()} poses, {len(yt)} avaliadas")

    real, orig = np.array(real), np.array(orig)
    letras = np.unique(real)
    for nome, p in pred.items():
        p = np.array(p)
        por_letra = {l: float(np.mean(p[real == l] == l)) for l in letras}
        print(f"\n{nome.upper()}: media por letra {np.mean(list(por_letra.values())) * 100:.1f}%  "
              f"| por pose {np.mean(p == real) * 100:.1f}%")
        print("  por origem: " + ", ".join(f"{f} {np.mean(p[orig == f] == real[orig == f]) * 100:.1f}%"
                                         for f in sorted(set(orig))))
        for l in letras:
            erros = Counter(p[(real == l) & (p != l)]).most_common(1)
            conf = f" (confunde com {erros[0][0]})" if erros else ""
            print(f"    {l}: {por_letra[l] * 100:5.1f}%  ({(real == l).sum()} poses){conf}")

    p, c = np.array(pred["com regras"]), np.array(confs)
    print("\nRECUSA POR CONFIANCA (com regras)")
    for lim in (0.55, 0.70):
        ac = c >= lim
        print(f"  limite {lim * 100:.0f}%: recusadas {np.mean(~ac) * 100:.1f}% ({(~ac).sum()} de {len(c)}); "
              f"acerto das aceitas {np.mean(p[ac] == real[ac]) * 100:.1f}%; "
              f"erros aceitos {np.sum(p[ac] != real[ac])} ({np.mean((p != real) & ac) * 100:.1f}% do total); "
              f"acertos recusados {np.sum((p == real) & ~ac)}")
        print("    por origem: " + ", ".join(
            f"{f} recusa {np.mean(~ac[orig == f]) * 100:.1f}% / acerto aceitas "
            f"{np.mean(p[ac & (orig == f)] == real[ac & (orig == f)]) * 100:.1f}%" for f in sorted(set(orig))))


if __name__ == "__main__":
    main()
