#!/usr/bin/env python3
"""
Treino das PALAVRAS v3 (duas maos + posicao em relacao ao corpo).

Le minds_v3/ (gerado por extrair_minds_v3.py), treina o XGBoost e salva
modelo_palavras_v3.pkl. O modelo v2 (modelo_rotulado_xgb.pkl) continua la:
a camera usa o v3 quando ele existe.

Uso:
  python treino_v3.py
  python treino_v3.py --palavras vacina,ruim,maca,banco,filho,bala
  python treino_v3.py --sem-teste      # pula a conferencia e vai direto ao modelo final

Leva alguns minutos e mostra o andamento a cada 25 arvores. No terminal, Ctrl+C
interrompe o programa (nao use para copiar texto enquanto ele treina).

Antes, para saber o numero do TCC (pessoa fora do treino):  python avaliar_v3.py
"""

import argparse
import re
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from libres_modelos import salvar_modelo
from libres_v3 import (FEATURE_VERSION_V3, N_FRAMES, aumentar_v3, features_v3, nomes_features_v3,
                       preparar_amostra)

PROJECT_DIR = Path(__file__).parent
# bases (extraidas no mesmo formato): MINDS-Libras, LIBRAS-EQT-UECE e coleta propria
DADOS = {"minds": PROJECT_DIR / "minds_v3", "eqt": PROJECT_DIR / "eqt_v3", "coleta": PROJECT_DIR / "coleta_v3"}
DADOS_DIRS = list(DADOS.values())          # padrao: todas as que existirem
# quem fez o sinal, pelo nome do arquivo -> prefixo do id (s01 = MINDS, i01 = EQT, p01 = coleta)
PREFIXO_PESSOA = {"sinalizador": "s", "informante": "i", "pessoa": "p"}
BASE_DO_PREFIXO = {"s": "MINDS-Libras", "i": "LIBRAS-EQT-UECE", "p": "coleta propria"}
MODELO_V3 = PROJECT_DIR / "modelo_palavras_v3.pkl"
AUMENTO = 8
MIN_POR_CLASSE = 5
# --recortar: tira a mao parada no colo do comeco e do fim de cada sinal (videos em
# que a mao ja aparece antes do sinal, como na LIBRAS-EQT-UECE). Vai junto no modelo.
RECORTAR = False
# --sem-mao-parada: num sinal de uma mao, a outra mao parada a vista (no colo) conta
# como ausente, igual a de quem abaixa a mao. Vai junto no modelo.
SEM_MAO_PARADA = False
# o MESMO numero de arvores na avaliacao por sinalizante (avaliar_v3.py) e no modelo
# final: o acerto medido no LOSO vale para o modelo que a camera usa
N_ARVORES = 300
NOMES = {"maca": "maçã", "america": "América"}


def escolher_dados(texto: str | None) -> None:
    """--dados minds,eqt -> so essas bases (padrao: todas as que existirem)."""
    global DADOS_DIRS
    if not texto:
        return
    nomes = [t.strip().lower() for t in texto.split(",") if t.strip()]
    desconhecidas = [n for n in nomes if n not in DADOS]
    if desconhecidas:
        sys.exit(f"Base desconhecida: {', '.join(desconhecidas)} (use {', '.join(DADOS)})")
    DADOS_DIRS = [DADOS[n] for n in nomes]


def ler_palavras(texto: str | None) -> list[str] | None:
    if not texto:
        return None
    import unicodedata
    sem = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return [sem(p.strip().lower()) for p in texto.split(",") if p.strip()]


def carregar(palavras: list[str] | None = None, verbose: bool = True):
    """-> seqs (N,30,135), rotulos (N,), grupos (N,), sinalizadores (N,), arquivos."""
    seqs, rot, grupos, sinal, arqs = [], [], [], [], []
    descartados = 0
    for base in DADOS_DIRS:
        if not base.exists():
            continue
        for pasta in sorted(p for p in base.iterdir() if p.is_dir()):
            nome = pasta.name.lower()
            if palavras and nome not in palavras:
                continue
            for f in sorted(pasta.glob("*.npz")):
                with np.load(f) as d:
                    seq = preparar_amostra(d, recortar_repouso=RECORTAR, sem_mao_parada=SEM_MAO_PARADA)
                if seq is None:
                    descartados += 1
                    continue
                m = re.search(r"(Sinalizador|Pessoa|Informante)(\d+)", f.stem, re.I)
                seqs.append(seq)
                rot.append(nome)
                grupos.append(f.stem)
                sinal.append(f"{PREFIXO_PESSOA[m.group(1).lower()]}{int(m.group(2)):02d}" if m else f.stem)
                arqs.append(f)
    if not seqs:
        sys.exit(f"Nenhum dado em {', '.join(d.name + '/' for d in DADOS_DIRS)}. Rode antes: "
                 "python extrair_minds_v3.py (MINDS) ou python extrair_eqt_v3.py (LIBRAS-EQT-UECE)")
    rot, sinal = np.array(rot), np.array(sinal)
    cont = Counter(rot.tolist())
    manter = np.array([cont[r] >= MIN_POR_CLASSE for r in rot])
    if verbose:
        por_base = Counter(BASE_DO_PREFIXO.get(s[0], "?") for s in sinal)
        print("  bases: " + ", ".join(f"{b} ({n} sinais)" for b, n in sorted(por_base.items())))
        print(f"  {len(seqs)} sinais | {len(cont)} palavras | {len(set(sinal))} sinalizadores "
              f"({', '.join(sorted(set(sinal)))})"
              + (f" | {descartados} sem mao descartados" if descartados else ""))
        poucas = sorted(r for r, n in cont.items() if n < MIN_POR_CLASSE)
        if poucas:
            print(f"  palavras com menos de {MIN_POR_CLASSE} sinais, fora do treino: {', '.join(poucas)}")
        if palavras:
            faltam = sorted(set(palavras) - set(cont))
            if faltam:
                print(f"  ATENCAO: sem dados para: {', '.join(faltam)}")
    seqs = np.array(seqs)[manter]
    return seqs, rot[manter], np.array(grupos)[manter], sinal[manter], [a for a, k in zip(arqs, manter) if k]


def calcular_features(seqs: np.ndarray, aumento: int = AUMENTO, semente: int = 42, verbose: bool = True,
                      chaves: list[str] | None = None):
    """Features das amostras reais e das variantes aumentadas (calculadas uma vez).

    Cada sinal tem o seu proprio sorteio das variantes (semente + nome do arquivo em
    `chaves`, ou + posicao): mudar um sinal (com --recortar ou --sem-mao-parada) nao
    muda as variantes dos outros. Com um sorteio unico para todos, um sinal mudado
    deslocava o sorteio de todos os seguintes, e a comparacao com e sem uma opcao
    misturava o efeito da opcao com o acaso."""
    t0 = time.time()
    reais = np.array([features_v3(s) for s in seqs])
    aum = np.empty((len(seqs), aumento, reais.shape[1]))
    for i, s in enumerate(seqs):
        rng = np.random.default_rng([semente, zlib.crc32(chaves[i].encode("utf8")) if chaves else i])
        for j, a in enumerate(aumentar_v3(s, rng, aumento)):
            aum[i, j] = features_v3(a)
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(seqs)} sinais...", flush=True)
    if verbose:
        print(f"  features prontas em {(time.time() - t0) / 60:.1f} min ({reais.shape[1]} por sinal)")
    return reais, aum


def novo_modelo(n_arvores: int = N_ARVORES, early: int | None = None, rotulo: str | None = None,
                cada: int = 25):
    """XGBoost das palavras. Com `rotulo`, mostra o andamento a cada `cada` arvores
    (o treino leva minutos e, sem isso, parece travado)."""
    import xgboost as xgb
    callbacks = [_progresso(n_arvores, rotulo, cada)] if rotulo else None
    return xgb.XGBClassifier(
        n_estimators=n_arvores, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5,
        min_child_weight=3, random_state=42, n_jobs=-1, eval_metric="mlogloss",
        tree_method="hist", early_stopping_rounds=early, callbacks=callbacks)


def _progresso(total: int, rotulo: str, cada: int = 25):
    import xgboost as xgb

    class Progresso(xgb.callback.TrainingCallback):
        def __init__(self) -> None:
            super().__init__()
            self.t0 = time.time()

        def after_iteration(self, model, epoch, evals_log) -> bool:
            k = epoch + 1
            if k % cada == 0 or k == total:
                gasto = time.time() - self.t0
                falta = gasto / k * (total - k)
                print(f"    {rotulo}: {k}/{total} arvores | {gasto / 60:.1f} min"
                      + (f" | faltam ~{falta / 60:.1f} min" if k < total else ""), flush=True)
            return False

    return Progresso()


def empilhar(reais, aum, y, idx):
    X = np.concatenate([reais[idx], aum[idx].reshape(-1, reais.shape[1])])
    Y = np.concatenate([y[idx], np.repeat(y[idx], aum.shape[1])])
    return X, Y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--palavras", default=None)
    ap.add_argument("--dados", default=None,
                    help="bases usadas, ex.: minds,eqt (padrao: todas as que existirem)")
    ap.add_argument("--recortar", action="store_true",
                    help="tira a mao parada no colo do comeco e do fim de cada sinal")
    ap.add_argument("--sem-mao-parada", action="store_true",
                    help="ignora a outra mao quando ela fica parada a vista longe do sinal")
    ap.add_argument("--arvores", type=int, default=N_ARVORES,
                    help=f"arvores do XGBoost (padrao {N_ARVORES}, o mesmo do avaliar_v3.py)")
    ap.add_argument("--sem-teste", action="store_true",
                    help="pula a conferencia com videos fora do treino e vai direto ao modelo final")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print("=" * 60)
    print("TREINO PALAVRAS v3 (duas maos + corpo) - XGBoost")
    print("=" * 60)
    global RECORTAR, SEM_MAO_PARADA
    escolher_dados(args.dados)
    RECORTAR = args.recortar
    SEM_MAO_PARADA = args.sem_mao_parada
    if RECORTAR:
        print("  --recortar: sem a mao parada no comeco e no fim de cada sinal")
    if SEM_MAO_PARADA:
        print("  --sem-mao-parada: outra mao parada longe do sinal conta como ausente")
    palavras = ler_palavras(args.palavras)
    seqs, rot, grupos, sinal, arqs = carregar(palavras)
    le = LabelEncoder()
    y = le.fit_transform(rot)
    print("\n[1/3] Calculando features...")
    reais, aum = calcular_features(seqs, chaves=[a.name for a in arqs])

    idx = np.arange(len(y))
    print("  (o treino leva alguns minutos; o andamento aparece a cada 25 arvores.")
    print("   No terminal, Ctrl+C interrompe o programa: nao use para copiar enquanto treina.)")
    if args.sem_teste:
        print("\n[2/3] Conferencia com videos fora do treino: pulada (--sem-teste)")
    else:
        # conferencia rapida: videos inteiros fora do treino (a pessoa aparece em outros videos)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr, te = next(gss.split(idx, y, grupos))
        print(f"\n[2/3] Conferencia: treino {len(tr)} sinais x{AUMENTO + 1}, teste {len(te)} videos fora do treino")
        Xtr, Ytr = empilhar(reais, aum, y, tr)
        sc = StandardScaler().fit(Xtr)
        m = novo_modelo(args.arvores, rotulo="conferencia")
        m.fit(sc.transform(Xtr), Ytr, verbose=False)
        pred = m.predict(sc.transform(reais[te]))
        print(f"  >>> ACURACIA NO TESTE (videos fora do treino): {accuracy_score(y[te], pred) * 100:.1f}% <<<")
        print("  (a pessoa do teste aparece no treino em outros videos; o numero honesto e o do avaliar_v3.py)")
        print(classification_report(y[te], pred, labels=np.unique(y[te]),
                                    target_names=le.inverse_transform(np.unique(y[te])), zero_division=0))

    print(f"[3/3] Modelo final com todos os sinais ({len(y)} x{AUMENTO + 1}, {args.arvores} arvores)...")
    Xall, Yall = empilhar(reais, aum, y, idx)
    sc = StandardScaler().fit(Xall)
    final = novo_modelo(args.arvores, rotulo="modelo final")
    final.fit(sc.transform(Xall), Yall, verbose=False)
    final.set_params(callbacks=None)          # o andamento nao vai para o arquivo do modelo
    nomes = nomes_features_v3()
    imp = final.feature_importances_
    print("  features que mais pesaram:")
    for k in np.argsort(imp)[::-1][:12]:
        print(f"    {nomes[k] if k < len(nomes) else k}: {imp[k]:.4f}")
    salvar_modelo({"model": final, "scaler": sc, "label_encoder": le, "feature_version": FEATURE_VERSION_V3,
                   "n_frames": N_FRAMES, "palavras": list(le.classes_), "recortar": RECORTAR,
                   "sem_mao_parada": SEM_MAO_PARADA,
                   "sinalizadores": sorted(set(sinal.tolist()))}, MODELO_V3)   # comprimido (cabe no GitHub)
    print(f"\nModelo salvo em: {MODELO_V3}")
    print(f"Palavras ({len(le.classes_)}): {', '.join(NOMES.get(c, c) for c in le.classes_)}")


if __name__ == "__main__":
    main()
