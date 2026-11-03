#!/usr/bin/env python3
"""
Avaliacao das PALAVRAS v3 deixando um sinalizador inteiro fora do treino (LOSO).

E o numero honesto para o TCC: o modelo e testado em uma pessoa que ele nunca
viu, rodada por rodada, com cada um dos sinalizadores de minds_v3/.

Uso:
  python avaliar_v3.py                         # as 20 palavras
  python avaliar_v3.py --palavras vacina,ruim,maca,banco,filho,bala
  python avaliar_v3.py --comparar              # tambem mede a v2 (1 mao) nos mesmos videos
  python avaliar_v3.py --dados eqt             # so a LIBRAS-EQT-UECE (6 informantes)
  python avaliar_v3.py --dados minds,eqt       # as duas juntas (o vocabulario das duas)
  python avaliar_v3.py --dados eqt --recortar  # sem a mao parada no colo antes/depois do sinal
  python avaliar_v3.py --dados eqt --sem-mao-parada   # ignora a outra mao estacionada a vista
  python avaliar_v3.py --dados minds --sem-mao-parada --antes-depois   # sem x com, pessoa por pessoa

No fim sugere um vocabulario: as palavras que passaram de --meta (padrao 85%).
"""

import argparse
import time

import numpy as np

import treino_v3 as T


def loso(reais, aum, y, sinal, rotulo: str):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    accs, baccs, reais_y, prev, por_pessoa = [], [], [], [], {}
    ids = sorted(set(sinal.tolist()))
    print(f"\n  {len(ids)} rodadas (uma pessoa fora em cada). Cada rodada leva alguns minutos;")
    print("  o andamento aparece a cada 100 arvores. Nao aperte Ctrl+C (interrompe tudo).")
    for k, sid in enumerate(ids, 1):
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
        p = m.predict(sc.transform(reais[te]))
        accs.append(accuracy_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], p))
        por_pessoa[sid] = accs[-1]
        reais_y.extend(y[te]); prev.extend(p)
        print(f"  [{k:2d}/{len(ids)}] {sid}: acc={accs[-1] * 100:5.1f}%  bal_acc={baccs[-1] * 100:5.1f}%  "
              f"({len(te)} sinais, {time.time() - t0:.0f}s)", flush=True)
    print("\n" + "=" * 60)
    print(f">>> LOSO {rotulo}  acc={np.mean(accs) * 100:.1f}%  bal_acc={np.mean(baccs) * 100:.1f}% <<<")
    print(f"    (pior={min(accs) * 100:.1f}%  melhor={max(accs) * 100:.1f}%)")
    bases = sorted({sid[0] for sid in por_pessoa})
    if len(bases) > 1:                       # mais de uma base: o acerto de cada uma
        for b in bases:
            v = [a for sid, a in por_pessoa.items() if sid[0] == b]
            print(f"    {T.BASE_DO_PREFIXO.get(b, b)}: {np.mean(v) * 100:.1f}% "
                  f"(media de {len(v)} pessoas fora do treino)")
    print("=" * 60)
    global ULTIMO_POR_PESSOA
    ULTIMO_POR_PESSOA = dict(por_pessoa)
    return np.array(reais_y), np.array(prev), float(np.mean(accs))


ULTIMO_POR_PESSOA: dict = {}          # acerto de cada pessoa no ultimo loso() (para o --antes-depois)


def antes_depois(args) -> None:
    """Roda o LOSO sem as opcoes e com elas, nos mesmos sinais, e compara pessoa por pessoa."""
    from sklearn.preprocessing import LabelEncoder
    if not (args.recortar or args.sem_mao_parada):
        raise SystemExit("--antes-depois vai junto com --recortar e/ou --sem-mao-parada")
    opcoes = " ".join(o for o, v in (("--recortar", args.recortar), ("--sem-mao-parada", args.sem_mao_parada)) if v)
    res = {}
    for rotulo, rec, smp in (("sem opcoes", False, False), (opcoes, args.recortar, args.sem_mao_parada)):
        print(f"\n{'#' * 60}\n# {rotulo}\n{'#' * 60}")
        T.RECORTAR, T.SEM_MAO_PARADA = rec, smp
        seqs, rot, _, sinal, arqs = T.carregar(T.ler_palavras(args.palavras))
        le = LabelEncoder()
        y = le.fit_transform(rot)
        reais, aum = T.calcular_features(seqs, chaves=[a.name for a in arqs])
        r, p, media = loso(reais, aum, y, sinal, rotulo)
        palavra = {le.classes_[c]: float(np.mean(p[r == c] == c)) for c in np.unique(r)}
        res[rotulo] = (media, dict(ULTIMO_POR_PESSOA), palavra)
    (m0, p0, w0), (m1, p1, w1) = res["sem opcoes"], res[opcoes]
    print("\n" + "=" * 60)
    print(f"ANTES x DEPOIS ({opcoes}) - pessoa fora do treino")
    print("=" * 60)
    print(f"  {'pessoa':8s} {'sem opcoes':>10s} {'com':>8s} {'diferenca':>10s}")
    for sid in sorted(set(p0) | set(p1)):
        a, b = p0.get(sid, np.nan), p1.get(sid, np.nan)
        print(f"  {sid:8s} {a * 100:9.1f}% {b * 100:7.1f}% {(b - a) * 100:+9.1f}")
    print(f"  {'MEDIA':8s} {m0 * 100:9.1f}% {m1 * 100:7.1f}% {(m1 - m0) * 100:+9.1f}")
    mudou = sorted(((w1[k] - w0[k], k) for k in set(w0) & set(w1) if abs(w1[k] - w0[k]) >= 0.05), reverse=True)
    if mudou:
        print("\n  palavras que mais mudaram (5 pontos ou mais):")
        for d, k in [x for x in mudou if x[0] > 0][:8] + [x for x in mudou if x[0] < 0][-8:]:
            print(f"    {T.NOMES.get(k, k):24s} {w0[k] * 100:5.1f}% -> {w1[k] * 100:5.1f}%  ({d * 100:+.1f})")
    print("\n  (cada sinal tem o mesmo sorteio de variantes nas duas rodadas: a diferenca vem so"
          "\n   dos sinais que as opcoes mudaram)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--palavras", default=None)
    ap.add_argument("--dados", default=None,
                    help="bases avaliadas, ex.: eqt ou minds,eqt (padrao: todas as que existirem)")
    ap.add_argument("--comparar", action="store_true", help="mede tambem a v2 (uma mao) nos mesmos videos")
    ap.add_argument("--recortar", action="store_true",
                    help="tira a mao parada no colo do comeco e do fim de cada sinal (diagnostico_v3.py)")
    ap.add_argument("--sem-mao-parada", action="store_true",
                    help="ignora a outra mao quando ela fica parada a vista longe do sinal (diagnostico_v3.py)")
    ap.add_argument("--meta", type=float, default=85.0, help="acerto minimo para sugerir a palavra")
    ap.add_argument("--antes-depois", action="store_true",
                    help="roda sem as opcoes e com elas e compara pessoa por pessoa (leva o dobro do tempo)")
    args = ap.parse_args()

    from sklearn.preprocessing import LabelEncoder

    print("=" * 60)
    print("AVALIACAO v3 (duas maos + corpo) - PESSOA FORA DO TREINO (LOSO)")
    print("=" * 60)
    T.escolher_dados(args.dados)
    if args.antes_depois:
        antes_depois(args)
        return
    T.RECORTAR = args.recortar
    T.SEM_MAO_PARADA = args.sem_mao_parada
    if args.recortar:
        print("  --recortar: sem a mao parada no comeco e no fim de cada sinal")
    if args.sem_mao_parada:
        print("  --sem-mao-parada: outra mao parada longe do sinal conta como ausente")
    seqs, rot, grupos, sinal, arqs = T.carregar(T.ler_palavras(args.palavras))
    le = LabelEncoder()
    y = le.fit_transform(rot)
    print("\n  calculando features (uma vez so)...")
    reais, aum = T.calcular_features(seqs, chaves=[a.name for a in arqs])
    r, p, media_v3 = loso(reais, aum, y, sinal, "v3")

    print("\nAcerto por palavra (pessoa fora do treino):")
    linhas = []
    for c in np.unique(r):
        m = r == c
        acc = float(np.mean(p[m] == c))
        erros = p[m][p[m] != c]
        conf = ""
        if len(erros):
            v, n = np.unique(erros, return_counts=True)
            conf = f"  (confunde com {T.NOMES.get(le.classes_[v[n.argmax()]], le.classes_[v[n.argmax()]])})"
        linhas.append((acc, le.classes_[c], int(m.sum()), conf))
    for acc, nome, n, conf in sorted(linhas, reverse=True):
        print(f"  {T.NOMES.get(nome, nome):12s} {acc * 100:5.1f}%  ({n} sinais){conf}")
    boas = [nome for acc, nome, _, _ in sorted(linhas, reverse=True) if acc * 100 >= args.meta]
    if boas:
        print(f"\nPalavras com pelo menos {args.meta:.0f}% ({len(boas)}): {','.join(boas)}")
        print(f"  para um vocabulario so com elas:  python avaliar_v3.py --palavras {','.join(boas)}")

    if args.comparar:
        print("\n" + "-" * 60)
        print("COMPARACAO: v2 (uma mao, sem corpo) nos MESMOS videos")
        print("-" * 60)
        from libres_features import compute_enhanced_features, normalize_sequence
        from libres_v3 import mao_dominante_crua
        import train_rotulado as TR
        np.random.seed(42)
        rv, av, ok = [], [], []
        for f in arqs:
            with np.load(f) as d:
                cru = mao_dominante_crua(d["maos"], d["lado"], float(d["aspecto"]))
            if cru is None or len(cru) < TR.MIN_HAND_FRAMES:
                ok.append(False)
                continue
            s2 = normalize_sequence(cru)
            rv.append(compute_enhanced_features(s2))
            av.append([compute_enhanced_features(a) for a in TR.augment_sequence(s2)])
            ok.append(True)
        ok = np.array(ok)
        _, _, media_v2 = loso(np.array(rv), np.array(av), y[ok], sinal[ok], "v2")
        print(f"\nv2 (uma mao): {media_v2 * 100:.1f}%  ->  v3 (duas maos + corpo): {media_v3 * 100:.1f}%")


if __name__ == "__main__":
    main()
