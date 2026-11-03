#!/usr/bin/env python3
"""
Diagnostico dos sinais extraidos (minds_v3/, eqt_v3/, coleta_v3/), POR PESSOA.

Serve para descobrir por que uma pessoa (ou uma base) foi pior na avaliacao:
le so os .npz ja extraidos (nada de video, ~1 minuto) e mostra, para cada
pessoa, o que pode atrapalhar o reconhecimento.

Tabela 1 (gravacao):
  imagem      largura/altura do video (abaixo de 1 = video em pe, de celular)
  mao%        quadros com mao detectada DENTRO do sinal (falhas no meio do sinal)
  2maos%      quadros com as duas maos, dentro do sinal
  corpo%      quadros com nariz e ombros detectados (posicao em relacao ao corpo)
  ini/fim     videos em que a mao ja aparece no 1o quadro / ainda aparece no ultimo
  parado      segundos de mao parada no comeco + no fim do sinal
  ativo%      parte do sinal em que a mao esta se mexendo
  mao/ombro   tamanho da mao / largura dos ombros (distancia e enquadramento)
  corte%      quanto do sinal o --recortar tiraria nas pontas: maos baixas e paradas, ou
              ainda (ou de novo) onde estavam no comeco do video, se comecaram embaixo

Tabela 2 (a OUTRA mao, a que nao e a dominante; medido no meio do sinal, sem os
20% do comeco e do fim - quem descansa as maos juntas so as encosta nas pontas):
  visivel%    sinais em que a outra mao aparece em pelo menos 30% dos quadros
  parada%     desses, em quantos ela quase nao se mexe
  altura      onde ela fica (larguras de ombro abaixo dos ombros; 1 = cintura)
  longe%      desses, em quantos as duas maos nunca se tocam (nem chegam perto)
  ignorada%   sinais em que o --sem-mao-parada a trataria como ausente: (a) parada,
              abaixo do peito e sem tocar a dominante, ou (b) o video comeca com
              ela descansando embaixo e ela continua la, sem toque
  (a)% (b)%   quanto cada uma das duas regras pega (um sinal pode estar nas duas)
  folga       nos sinais com a outra mao parada: o quanto as maos chegam perto
              (mediana, larguras de ombro; abaixo de 0.25 a regra (a) nao pega)
  afastou     nos sinais com a outra mao parada: o quanto ela sai do lugar do
              comeco do video (mediana; acima de 0.15 a regra (b) nao pega)
  c/rec%      sinais em que a outra mao e ignorada com as DUAS opcoes juntas
              (--recortar --sem-mao-parada): a regra olha o sinal ja recortado

Uso:
  python diagnostico_v3.py                  # todas as bases que existirem
  python diagnostico_v3.py --dados eqt      # so a LIBRAS-EQT-UECE
  python diagnostico_v3.py --dados minds,eqt

A saida e so texto sem acento: pode mandar com  > diagnostico.txt
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import treino_v3 as T
from libres_v3 import (NARIZ, OMBRO_D, OMBRO_E, OUTRA_ALTA, OUTRA_DESCANSO, OUTRA_FAIXA, OUTRA_FICOU,
                       OUTRA_LONGE, _referencia_corpo, escolher_dominante, medidas_outra_mao, rastrear,
                       regras_outra_mao, trecho_sem_repouso)

VEL_ATIVO = 0.35          # larguras de ombro por segundo: acima disso a mao "esta se mexendo"


def pessoa_de(arquivo: Path) -> str:
    m = re.search(r"(Sinalizador|Pessoa|Informante)(\d+)", arquivo.stem, re.I)
    return f"{T.PREFIXO_PESSOA[m.group(1).lower()]}{int(m.group(2)):02d}" if m else "?"


def medir(d) -> dict | None:
    """Medidas de um sinal (um .npz)."""
    maos = np.asarray(d["maos"], np.float64).reshape(-1, 2, 21, 3)
    lado = np.asarray(d["lado"]).reshape(-1, 2)
    pose = np.asarray(d["pose"], np.float64)
    asp = float(d["aspecto"])
    fps = float(d["fps"]) if "fps" in d.files and 5 <= float(d["fps"]) <= 120 else 30.0
    T_ = len(maos)
    tem = ~np.isnan(maos[:, :, 0, 0])
    alguma = tem.any(axis=1)
    if T_ == 0 or alguma.sum() < 3:
        return None
    corpo = (np.isfinite(pose[:, [NARIZ, OMBRO_E, OMBRO_D], :2]).all(axis=(1, 2))
             if pose.ndim == 3 else np.zeros(T_, bool))
    i0, i1 = int(np.argmax(alguma)), T_ - int(np.argmax(alguma[::-1]))     # trecho que o treino usa
    m = rastrear(maos[i0:i1], lado[i0:i1])
    m[..., [0, 2]] *= asp
    tem_s = ~np.isnan(m[:, :, 0, 0])
    pose_s = pose[i0:i1] if pose.ndim == 3 else pose
    centro, larg, _ = _referencia_corpo(pose_s, asp)
    corte = 0.0
    m_rec = m                                   # o sinal como o --recortar deixa (para a coluna c/rec%)
    if pose.ndim == 3 and np.isfinite(pose_s[:, [OMBRO_E, OMBRO_D], :2]).all(axis=(1, 2)).mean() >= 0.5:
        j0, j1 = trecho_sem_repouso(m, centro, larg, fps)
        corte = 1.0 - (j1 - j0) / len(m)
        m_rec = m[j0:j1]
    # o mesmo caminho do preparar com --recortar --sem-mao-parada
    dom_rec = escolher_dominante(m_rec, sem_tremor=True)
    if (~np.isnan(m_rec[:, dom_rec, 0, 0])).sum() < 3:
        dom_rec = 1 - dom_rec
    ign_rec = any(regras_outra_mao(medidas_outra_mao(m_rec, dom_rec, centro, larg)))
    # movimento: deslocamento medio dos 21 pontos de cada mao, em larguras de ombro por segundo
    vel = np.zeros(len(m))
    for k in range(2):
        p = m[:, k, :, :2] / larg
        dv = np.linalg.norm(np.diff(p, axis=0), axis=2).mean(axis=1) * fps        # (T-1,)
        dv = np.nan_to_num(dv, nan=0.0)
        vel[1:] = np.maximum(vel[1:], dv)
    if len(vel) >= 5:                                                         # suaviza (~0,1 s)
        vel = np.convolve(vel, np.ones(3) / 3, mode="same")
    ativo = vel > VEL_ATIVO
    if ativo.any():
        a0, a1 = int(np.argmax(ativo)), len(ativo) - 1 - int(np.argmax(ativo[::-1]))
        parado_ini, parado_fim = a0 / fps, (len(ativo) - 1 - a1) / fps
    else:
        parado_ini = parado_fim = len(ativo) / fps
    tam = []
    for k in range(2):
        ok = tem_s[:, k]
        if ok.any():
            tam.append(np.linalg.norm(m[ok, k, 9, :2] - m[ok, k, 0, :2], axis=1))
    tam_mao = float(np.median(np.concatenate(tam))) / larg if tam else np.nan
    # a outra mao (dominante = a que mais anda, sem o tremor da deteccao)
    dom = escolher_dominante(m, sem_tremor=True)              # a mesma escolha do --sem-mao-parada
    if tem_s[:, dom].sum() < 3:
        dom = 1 - dom
    outra = medidas_outra_mao(m, dom, centro, larg)
    if outra is not None and outra["visivel"] < 0.3:
        outra = None
    return {"quadros": T_, "fps": fps, "aspecto": asp, "mao": tem_s.any(axis=1).mean(),
            "duas": tem_s.all(axis=1).mean(), "corpo": corpo.mean(),
            "ini": bool(alguma[:2].any()), "fim": bool(alguma[-2:].any()),
            "parado": parado_ini + parado_fim, "trecho": (i1 - i0) / fps,
            "ativo": ativo.mean(), "mao_ombro": tam_mao, "corte": corte,
            "outra": outra, "regra_a": regras_outra_mao(outra)[0], "regra_b": regras_outra_mao(outra)[1],
            "ignorada": any(regras_outra_mao(outra)), "ign_rec": ign_rec}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default=None, help="ex.: eqt ou minds,eqt (padrao: todas as que existirem)")
    args = ap.parse_args()
    T.escolher_dados(args.dados)

    por_pessoa = defaultdict(list)
    sem_mao = defaultdict(int)
    for base in T.DADOS_DIRS:
        if not base.exists():
            continue
        for f in sorted(base.glob("*/*.npz")):
            try:
                with np.load(f) as d:
                    r = medir(d)
            except Exception:  # noqa: BLE001  (arquivo corrompido)
                r = None
            if r is None:
                sem_mao[pessoa_de(f)] += 1
            else:
                por_pessoa[pessoa_de(f)].append(r)
    if not por_pessoa:
        raise SystemExit("Nenhum sinal encontrado. Rode antes o extrator (extrair_minds_v3.py / extrair_eqt_v3.py).")

    print("=" * 108)
    print("1. GRAVACAO, POR PESSOA (medianas; % = media)")
    print("=" * 108)
    cab = (f"{'pessoa':7s} {'base':16s} {'sinais':>6s} {'quadros':>7s} {'fps':>4s} {'imagem':>6s} {'mao%':>5s} "
           f"{'2maos%':>6s} {'corpo%':>6s} {'ini/fim%':>9s} {'parado':>7s} {'ativo%':>6s} {'mao/ombro':>9s} "
           f"{'corte%':>6s}")
    print(cab)
    print("-" * len(cab))
    resumo = {}
    for sid in sorted(por_pessoa):
        L = por_pessoa[sid]
        med = lambda k: float(np.nanmedian([r[k] for r in L]))
        media = lambda k: float(np.nanmean([r[k] for r in L]))
        outras = [r["outra"] for r in L if r["outra"] is not None]
        s = {"n": len(L), "quadros": med("quadros"), "fps": med("fps"), "aspecto": med("aspecto"),
             "mao": media("mao"), "duas": media("duas"), "corpo": media("corpo"),
             "ini": media("ini"), "fim": media("fim"), "parado": med("parado"),
             "ativo": media("ativo"), "mao_ombro": med("mao_ombro"), "trecho": med("trecho"),
             "corte": media("corte"),
             "visivel": len(outras) / len(L),
             "parada": float(np.mean([o["faixa"] < OUTRA_FAIXA for o in outras])) if outras else np.nan,
             "altura": float(np.median([o["altura"] for o in outras])) if outras else np.nan,
             "longe": float(np.mean([o["perto"] > OUTRA_LONGE for o in outras])) if outras else np.nan,
             "ignorada": media("ignorada"), "regra_a": media("regra_a"), "regra_b": media("regra_b"),
             "ign_rec": media("ign_rec")}
        paradas = [o for o in outras if o["faixa"] < OUTRA_FAIXA]
        s["folga"] = float(np.median([min(o["perto"], 9.0) for o in paradas])) if paradas else np.nan
        afast = [o["afastou"] for o in paradas if np.isfinite(o["afastou"])]
        s["afastou"] = float(np.median(afast)) if afast else np.nan
        resumo[sid] = s
        base = T.BASE_DO_PREFIXO.get(sid[0], "?")
        print(f"{sid:7s} {base:16s} {s['n']:6d} {s['quadros']:7.0f} {s['fps']:4.0f} {s['aspecto']:6.2f} "
              f"{s['mao'] * 100:5.0f} {s['duas'] * 100:6.0f} {s['corpo'] * 100:6.0f} "
              f"{s['ini'] * 100:4.0f}/{s['fim'] * 100:<4.0f} {s['parado']:6.1f}s {s['ativo'] * 100:6.0f} "
              f"{s['mao_ombro']:9.2f} {s['corte'] * 100:6.0f}")
    if sem_mao:
        print("\nsinais sem mao suficiente (o treino descarta): "
              + ", ".join(f"{k} {v}" for k, v in sorted(sem_mao.items())))

    print("\n" + "=" * 108)
    print("2. A OUTRA MAO (a que nao e a dominante), POR PESSOA")
    print("=" * 108)
    cab2 = (f"{'pessoa':7s} {'base':16s} {'visivel%':>8s} {'parada%':>8s} {'altura':>7s} {'longe%':>7s} "
            f"{'(a)%':>5s} {'(b)%':>5s} {'ignorada%':>9s} {'folga':>6s} {'afastou':>7s} {'c/rec%':>6s}")
    print(cab2)
    print("-" * len(cab2))
    fmt = lambda v, esc=100, casas=0: "   -" if not np.isfinite(v) else f"{v * esc:.{casas}f}"
    for sid in sorted(resumo):
        s = resumo[sid]
        print(f"{sid:7s} {T.BASE_DO_PREFIXO.get(sid[0], '?'):16s} {fmt(s['visivel']):>8s} {fmt(s['parada']):>8s} "
              f"{fmt(s['altura'], 1, 2):>7s} {fmt(s['longe']):>7s} {fmt(s['regra_a']):>5s} {fmt(s['regra_b']):>5s} "
              f"{fmt(s['ignorada']):>9s} {fmt(s['folga'], 1, 2):>6s} {fmt(s['afastou'], 1, 2):>7s} "
              f"{fmt(s['ign_rec']):>6s}")
    print(f"(medido no meio do sinal. parada = o punho anda menos de {OUTRA_FAIXA} largura de ombro;\n"
          f" longe = as duas maos nunca chegam a menos de {OUTRA_LONGE} uma da outra (nao se tocam);\n"
          f" ignorada = (a) parada, longe e mais de {OUTRA_ALTA} largura de ombro abaixo dos ombros, ou\n"
          f" (b) o video comeca com ela mais de {OUTRA_DESCANSO} abaixo dos ombros e, no meio do sinal, ela sai\n"
          f" menos de {OUTRA_FICOU} desse lugar, sem toque. folga e afastou: medianas nos sinais com ela parada;\n"
          f" c/rec = ignorada com --recortar --sem-mao-parada juntos, no sinal ja recortado)")

    # alertas: comparando cada pessoa com as outras da MESMA base
    print("\nALERTAS")
    alertas = 0
    for prefixo in sorted({sid[0] for sid in resumo}):
        grupo = {sid: s for sid, s in resumo.items() if sid[0] == prefixo}
        nome = T.BASE_DO_PREFIXO.get(prefixo, prefixo)
        arg = {"i": "eqt", "s": "minds"}.get(prefixo, "coleta")
        tipico = {k: float(np.median([s[k] for s in grupo.values()]))
                  for k in ("mao_ombro", "aspecto", "ativo", "ignorada")}
        corte = float(np.mean([s["corte"] for s in grupo.values()]))
        if corte >= 0.10:
            alertas += 1
            print(f"  {nome}: o --recortar tiraria {corte * 100:.0f}% dos quadros (mao parada no colo). Compare: "
                  f"python avaliar_v3.py --dados {arg} --recortar")
        for sid, s in sorted(grupo.items()):
            msgs = []
            if s["corpo"] < 0.8:
                msgs.append(f"corpo detectado em so {s['corpo'] * 100:.0f}% dos quadros (posicao em relacao ao "
                            "corpo fica estimada)")
            if s["mao"] < 0.8:
                msgs.append(f"mao perdida em {(1 - s['mao']) * 100:.0f}% dos quadros no meio do sinal")
            if len(grupo) > 2 and np.isfinite(s["mao_ombro"]) and abs(s["mao_ombro"] / tipico["mao_ombro"] - 1) > 0.25:
                msgs.append(f"mao/ombro {s['mao_ombro']:.2f} x {tipico['mao_ombro']:.2f} das outras pessoas "
                            "(enquadramento ou deteccao diferentes)")
            if len(grupo) > 2 and abs(s["aspecto"] - tipico["aspecto"]) > 0.15:
                msgs.append(f"imagem {s['aspecto']:.2f} x {tipico['aspecto']:.2f} das outras (camera/orientacao diferente)")
            if len(grupo) > 2 and s["ativo"] < tipico["ativo"] - 0.15:
                msgs.append(f"so {s['ativo'] * 100:.0f}% do sinal com movimento x {tipico['ativo'] * 100:.0f}% das "
                            "outras (mais tempo parada)")
            if s["ignorada"] >= 0.15 and s["ignorada"] >= tipico["ignorada"] + 0.10:
                msgs.append(f"deixa a outra mao parada a vista em {s['ignorada'] * 100:.0f}% dos sinais (as outras "
                            f"pessoas: {tipico['ignorada'] * 100:.0f}%); o modelo pode achar que sao sinais de duas "
                            f"maos. Compare: python avaliar_v3.py --dados {arg} --sem-mao-parada")
            for msg in msgs:
                alertas += 1
                print(f"  {sid}: {msg}")
    if not alertas:
        print("  nenhum: as pessoas de cada base foram gravadas de forma parecida.")


if __name__ == "__main__":
    main()
