#!/usr/bin/env python3
"""Modelo conjunto (as duas bases, 65 palavras), com uma pessoa fora do treino em cada rodada.

E o modelo que o aplicativo usa (treino_v3.py com as duas bases), avaliado no mesmo protocolo
das Tabelas 2 e 3: em cada rodada, todos os sinais de uma pessoa (11 do MINDS-Libras e 5 da
LIBRAS-EQT-UECE, 16 rodadas) ficam fora do treino. Alem do acerto, mede:
  - confusao entre as bases: quantos sinais de uma base viram uma palavra da outra;
  - recusa (RNF08): a regra da camera (accept_word: a primeira com pelo menos 45% e 15 pontos
    a frente da segunda) aplicada as probabilidades de cada sinal fora do treino; quantos
    seriam recusados como incertos, o acerto dos aceitos e quantos erros passam.
O treino, o aumento de dados e as sementes sao os de treino_v3.py / avaliar_v3.py. Nada e salvo.

Uso: python avaliar_conjunto.py --recortar --sem-mao-parada
"""
import argparse
import json
from pathlib import Path
import time
from collections import Counter

import numpy as np

import treino_v3 as T
from libres_realtime import accept_word


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recortar", action="store_true")
    ap.add_argument("--sem-mao-parada", action="store_true")
    ap.add_argument("--min-conf", type=float, default=0.45)
    ap.add_argument("--min-margem", type=float, default=0.15)
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print("=" * 60)
    print("MODELO CONJUNTO (MINDS-Libras + LIBRAS-EQT-UECE) - PESSOA FORA DO TREINO")
    print("=" * 60)
    T.escolher_dados("minds,eqt")
    T.RECORTAR = args.recortar
    T.SEM_MAO_PARADA = args.sem_mao_parada
    print(f"  --recortar: {args.recortar}  --sem-mao-parada: {args.sem_mao_parada}")
    seqs, rot, grupos, sinal, arqs = T.carregar(None)
    le = LabelEncoder()
    y = le.fit_transform(rot)
    print("\n  calculando features (uma vez so)...")
    reais, aum = T.calcular_features(seqs, chaves=[a.name for a in arqs])
    sinal = np.asarray(sinal)
    base_sinal = np.array([s[0] for s in sinal])
    # base de cada palavra: a base das pessoas que a sinalizam
    base_palavra = {}
    for c in np.unique(y):
        b = Counter(base_sinal[y == c]).most_common(1)[0][0]
        base_palavra[c] = b
    print(f"\n  {len(y)} sinais, {len(le.classes_)} palavras, {len(set(sinal.tolist()))} pessoas")

    ids = sorted(set(sinal.tolist()))
    print(f"\n  {len(ids)} rodadas. Cada rodada leva varios minutos. Nao aperte Ctrl+C.")
    # parcial: cada rodada pronta fica gravada; se o programa parar, a proxima execucao continua dali
    parcial = Path(__file__).with_name(
        f"avaliacao_conjunto_parcial{'_r' if args.recortar else ''}{'_m' if args.sem_mao_parada else ''}.json")
    feito = json.loads(parcial.read_text()) if parcial.exists() else {}
    if feito:
        print(f"  continuando: {len(feito)} rodadas ja feitas ({', '.join(sorted(feito))})")
    reg = []                      # (pessoa, real, previsto, aceito)
    accs = {}
    for sid, dados in sorted(feito.items()):
        accs[sid] = dados["acc"]
        reg += [tuple(r) for r in dados["reg"]]
    for k, sid in enumerate(ids, 1):
        if sid in feito:
            print(f"  [{k:2d}/{len(ids)}] {sid}: acc={accs[sid] * 100:5.1f}%  (ja feita)")
            continue
        te = np.where(sinal == sid)[0]
        tr = np.where(sinal != sid)[0]
        te = te[np.isin(y[te], np.unique(y[tr]))]
        if len(te) == 0:
            continue
        t0 = time.time()
        Xtr, Ytr = T.empilhar(reais, aum, y, tr)
        sc = StandardScaler().fit(Xtr)
        m = T.novo_modelo(T.N_ARVORES, rotulo=f"rodada {k}/{len(ids)} ({sid})", cada=100)
        m.fit(sc.transform(Xtr), Ytr, verbose=False)
        proba = m.predict_proba(sc.transform(reais[te]))
        classes = m.classes_
        pred = classes[proba.argmax(axis=1)]
        for i, j in enumerate(te):
            ordem = np.argsort(proba[i])[::-1][:2]
            top = [(int(classes[o]), float(proba[i][o])) for o in ordem]
            aceito = accept_word(top, args.min_conf, args.min_margem) is not None
            reg.append((sid, int(y[j]), int(pred[i]), aceito))
        accs[sid] = accuracy_score(y[te], pred)
        feito[sid] = {"acc": float(accs[sid]), "reg": [list(r) for r in reg if r[0] == sid]}
        parcial.write_text(json.dumps(feito))
        bal = balanced_accuracy_score(y[te], pred)
        print(f"  [{k:2d}/{len(ids)}] {sid}: acc={accs[sid] * 100:5.1f}%  bal_acc={bal * 100:5.1f}%  "
              f"({len(te)} sinais, {time.time() - t0:.0f}s)", flush=True)

    pessoa = np.array([r[0] for r in reg])
    real = np.array([r[1] for r in reg])
    prev = np.array([r[2] for r in reg])
    aceito = np.array([r[3] for r in reg])
    certo = real == prev
    print("\n" + "=" * 60)
    print(f">>> LOSO CONJUNTO (65 palavras)  media entre pessoas={np.mean(list(accs.values())) * 100:.1f}%  "
          f"(pior={min(accs.values()) * 100:.1f}%  melhor={max(accs.values()) * 100:.1f}%) <<<")
    for b, nome in (("s", "MINDS-Libras"), ("i", "LIBRAS-EQT-UECE")):
        v = [a for s, a in accs.items() if s[0] == b]
        mb = np.array([p[0] == b for p in pessoa])
        outra = np.array([base_palavra[c] != b for c in prev[mb]])
        print(f"    {nome}: media {np.mean(v) * 100:.1f}% ({len(v)} pessoas, {mb.sum()} sinais); "
              f"previstos como palavra da outra base: {outra.sum()} ({outra.mean() * 100:.1f}%)")
    print("=" * 60)

    print(f"\nRECUSA (RNF08: pelo menos {args.min_conf * 100:.0f}% e {args.min_margem * 100:.0f} pontos a frente)")
    for nome, mb in (("Total", np.ones(len(reg), bool)),
                     ("MINDS-Libras", np.array([p[0] == "s" for p in pessoa])),
                     ("LIBRAS-EQT-UECE", np.array([p[0] == "i" for p in pessoa]))):
        a, c = aceito[mb], certo[mb]
        print(f"  {nome}: {mb.sum()} sinais; recusados {np.mean(~a) * 100:.1f}% ({(~a).sum()}); "
              f"acerto dos aceitos {np.mean(c[a]) * 100:.1f}%; erros aceitos {np.sum(a & ~c)} "
              f"({np.mean(a & ~c) * 100:.1f}% do total); erros recusados {np.sum(~a & ~c)} de {np.sum(~c)}; "
              f"acertos recusados {np.sum(~a & c)}")

    print("\nPares mais confundidos (real -> previsto, vezes):")
    pares = Counter((le.classes_[r], le.classes_[p]) for r, p in zip(real, prev) if r != p)
    for (a, b), n in pares.most_common(15):
        marca = " (outra base)" if base_palavra[le.transform([a])[0]] != base_palavra[le.transform([b])[0]] else ""
        print(f"  {T.NOMES.get(a, a):24s} -> {T.NOMES.get(b, b):24s} {n}{marca}")


if __name__ == "__main__":
    main()
