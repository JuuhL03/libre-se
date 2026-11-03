#!/usr/bin/env python3
"""
Reproduz um trecho salvo com a tecla S do libras_camera.py (so os pontos da
mao, sem imagem) e mostra o que o reconhecimento decidiu em cada momento.
Serve para descobrir por que uma letra ou palavra falhou e ajustar os
limiares com maos reais, sem precisar gravar dataset.

Uso:
  python reproduzir_trecho.py debug/trecho_20261024_153012.npz
  python reproduzir_trecho.py debug/trecho_20261024_153012.npz --quadros
  python reproduzir_trecho.py debug/trecho_20261024_153012.npz --modo PALAVRAS

Trechos salvos no modo PALAVRAS com o modelo v3 (duas maos + corpo) sao
reproduzidos com o v3 (se modelo_palavras_v3.pkl existir). Trechos do modo
CALIBRADO usam a calibracao da pessoa (calibracao/<usuario>/):
  python reproduzir_trecho.py debug/trecho_....npz --usuario julia

Saida (modo ALFABETO), um bloco por trecho de comportamento igual:
   0.00-0.47s  sem mao
   0.47-1.20s  parada     U 88%  forma=U  sep=+0.12
   1.20-1.83s  MOVIMENTO  forma=DOIS  vel max 3.1
     >> 1.93s  entrou H (trocou U -> H)
"""

import argparse
from pathlib import Path

import numpy as np

from libres_features import FEAT_DIM, eh_y, forma_mao, separacao_pontas
from libres_realtime import (LetterRecognizer, MovimentoMao, SignSegmenter, SignSegmenterV3,
                             Soletrador, WordRecognizer, WordRecognizerV3, accept_word, isotropico)

PROJECT_DIR = Path(__file__).parent


def carregar(caminho: Path) -> tuple[np.ndarray, np.ndarray, str, str, dict]:
    d = np.load(caminho, allow_pickle=False)
    t = d["t"].astype(np.float64)
    lm = d["lm"].astype(np.float64)
    if lm.ndim != 2 or lm.shape[1] != FEAT_DIM:
        raise SystemExit(f"{caminho.name}: formato inesperado {lm.shape}")
    # trechos antigos nao tem o aspecto: a camera abre em 1280x720
    extras = {"aspecto": float(d["aspecto"]) if "aspecto" in d.files else 16 / 9}
    if "maos" in d.files:
        extras.update(maos=d["maos"].astype(np.float64), lado=d["lado"], pose=d["pose"].astype(np.float64))
    return t - t[0], lm, str(d["modo"]), str(d["texto"]), extras


def rodar_alfabeto(t: np.ndarray, lm: np.ndarray, por_quadro: bool, aspecto: float = 16 / 9) -> str:
    try:
        rec = LetterRecognizer(PROJECT_DIR / "modelo_alfabeto.pkl")
    except Exception as e:  # noqa: BLE001
        print(f"  [erro] modelo do alfabeto: {e}")
        return ""
    sol = Soletrador(rec, aspecto=aspecto)
    vel = MovimentoMao()
    txt: list[str] = []
    bloco = None                       # [chave, t0, t1, n, confs, seps, vmax]

    def fechar(b):
        if b is None:
            return
        chave, t0, t1, n, confs, seps, vmax = b
        if chave[0] == "sem":
            desc = "sem mao"
        elif chave[0] == "mov":
            desc = f"MOVIMENTO  forma={chave[1]}  vel max {vmax:.1f}"
        else:
            desc = f"parada     {chave[1]} {np.mean(confs)*100:3.0f}%  forma={chave[2]}"
            if chave[1] in ("U", "R", "V") or chave[2] in ("U", "R", "V", "DOIS"):
                desc += f"  sep={np.mean(seps):+.2f}"
        print(f"  {t0:5.2f}-{t1:5.2f}s  {desc}  ({n} quadros)")

    for ti, raw in zip(t, lm):
        tem = not np.isnan(raw).any()
        info = sol.update(raw if tem else None, ti)
        v = vel.update(isotropico(raw, aspecto) if tem else None, ti)
        if tem:
            forma = forma_mao(raw) or "-"
            sep = separacao_pontas(raw)
            chave = ("mov", forma) if info["movendo"] else ("par", info["letra"], forma)
        else:
            forma, sep, chave = "-", 0.0, ("sem",)

        if por_quadro:
            if tem:
                top = " ".join(f"{l}{c*100:.0f}" for l, c in info["top"])
                print(f"  {ti:5.2f}s  {info['letra']:>2} {info['conf']*100:3.0f}%  "
                      f"{'MOV' if info['movendo'] else '   '}  vel={v:4.1f}  forma={forma:<9} "
                      f"sep={sep:+.2f}  Y={'s' if eh_y(raw) else 'n'}  top: {top}")
            else:
                print(f"  {ti:5.2f}s  sem mao")
        else:
            if bloco is None or bloco[0] != chave:
                fechar(bloco)
                bloco = [chave, ti, ti, 0, [], [], 0.0]
            bloco[2], bloco[3] = ti, bloco[3] + 1
            bloco[4].append(info["conf"])
            bloco[5].append(sep)
            bloco[6] = max(bloco[6], v)

        if info["acao"]:
            if info["acao"] == "troca" and txt:
                antiga, txt[-1] = txt[-1], info["letra"]
                msg = f"entrou {info['letra']} (trocou {antiga} -> {info['letra']})"
            else:
                txt.append(info["letra"])
                msg = f"entrou {info['letra']}"
            if not por_quadro:
                fechar(bloco)
                bloco = None
            print(f"    >> {ti:5.2f}s  {msg}")
    if not por_quadro:
        fechar(bloco)
    fim = sol.finalizar(float(t[-1]) + 0.05 if len(t) else 0.0)
    if fim["acao"]:
        if fim["acao"] == "troca" and txt:
            txt[-1] = fim["letra"]
        else:
            txt.append(fim["letra"])
        print(f"    >> fim do trecho: entrou {fim['letra']} (gesto que ainda estava aberto)")
    return "".join(txt)


def rodar_palavras(t: np.ndarray, lm: np.ndarray) -> str:
    try:
        words = WordRecognizer(PROJECT_DIR / "modelo_rotulado_xgb.pkl")
    except Exception as e:  # noqa: BLE001
        print(f"  [erro] modelo de palavras: {e}")
        return ""
    seg = SignSegmenter()
    saida: list[str] = []
    gravando_desde = None
    for ti, raw in zip(t, lm):
        tem = not np.isnan(raw).any()
        sinal = seg.update(raw if tem else None, ti)
        if seg.recording and gravando_desde is None:
            gravando_desde = ti
        if sinal is not None:
            top = words.predict(sinal)
            ok = accept_word(top)
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in top)
            ini = f"{gravando_desde:5.2f}" if gravando_desde is not None else "  ?  "
            print(f"  sinal {ini}-{ti:5.2f}s ({len(sinal)} quadros): {resumo}"
                  f"  -> {'ENTROU ' + ok.upper() if ok else 'incerto (nao entra)'}")
            if ok:
                saida.append(ok.upper())
        if not seg.recording:
            gravando_desde = None
    if seg.recording:
        print("  (o trecho acabou com um sinal ainda sendo gravado: abaixe a mao no fim)")
    return " ".join(saida)


def rodar_palavras_v3(t: np.ndarray, extras: dict) -> str:
    try:
        words = WordRecognizerV3(PROJECT_DIR / "modelo_palavras_v3.pkl")
    except Exception as e:  # noqa: BLE001
        print(f"  [erro] modelo de palavras v3: {e}")
        return ""
    seg = SignSegmenterV3()
    saida: list[str] = []
    gravando_desde = None
    for k, ti in enumerate(t):
        maos = extras["maos"][k]
        tem = not np.isnan(maos[:, 0, 0]).all()
        quadro = {"maos": maos, "lado": extras["lado"][k], "pose": extras["pose"][k]} if tem else None
        sinal = seg.update(quadro, ti)
        if seg.recording and gravando_desde is None:
            gravando_desde = ti
        if sinal is not None:
            top = words.predict(sinal, extras["aspecto"])
            ok = accept_word(top)
            duas = int((~np.isnan(sinal["maos"][:, :, 0, 0])).all(axis=1).sum())
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in top)
            ini = f"{gravando_desde:5.2f}" if gravando_desde is not None else "  ?  "
            print(f"  sinal {ini}-{ti:5.2f}s ({len(sinal['maos'])} quadros, {duas} com as duas maos): "
                  f"{resumo}  -> {'ENTROU ' + ok.upper() if ok else 'incerto (nao entra)'}")
            if ok:
                saida.append(ok.upper())
        if not seg.recording:
            gravando_desde = None
    if seg.recording:
        print("  (o trecho acabou com um sinal ainda sendo gravado: abaixe a mao no fim)")
    return " ".join(saida)


def rodar_calibrado(t: np.ndarray, extras: dict, usuario: str, vocab: Path) -> str:
    from calibracao import Calibracao, segmentador_calibrado
    rec = Calibracao(usuario, vocab).reconhecedor()
    if rec is None:
        print(f"  [erro] sem calibracao em calibracao/{usuario}/")
        return ""
    seg = segmentador_calibrado()
    saida: list[str] = []
    gravando_desde = None
    for k, ti in enumerate(t):
        maos = extras["maos"][k]
        tem = not np.isnan(maos[:, 0, 0]).all()
        quadro = {"maos": maos, "lado": extras["lado"][k], "pose": extras["pose"][k]} if tem else None
        sinal = seg.update(quadro, ti)
        if seg.recording and gravando_desde is None:
            gravando_desde = ti
        if sinal is not None:
            r = rec.predict(sinal, extras["aspecto"])
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in r["top"])
            ini = f"{gravando_desde:5.2f}" if gravando_desde is not None else "  ?  "
            print(f"  sinal {ini}-{ti:5.2f}s ({len(sinal['maos'])} quadros): {resumo}  distancia/limite "
                  f"{r['razao']:.2f} -> {'ENTROU ' + r['aceito'].upper() if r['aceito'] else 'NAO RECONHECIDO'}")
            if r["aceito"]:
                saida.append(r["aceito"].upper())
        if not seg.recording:
            gravando_desde = None
    if seg.recording:
        print("  (o trecho acabou com um sinal ainda sendo gravado: abaixe a mao no fim)")
    return " ".join(saida)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arquivos", nargs="+", help="arquivos .npz salvos com a tecla S")
    ap.add_argument("--modo", choices=["ALFABETO", "PALAVRAS", "CALIBRADO"],
                    help="padrao: o modo em que foi salvo")
    ap.add_argument("--usuario", default="padrao", help="calibracao usada no modo CALIBRADO")
    ap.add_argument("--vocabulario", default=str(PROJECT_DIR / "vocabulario.txt"))
    ap.add_argument("--quadros", action="store_true", help="mostra todos os quadros (alfabeto)")
    args = ap.parse_args()

    for arq in args.arquivos:
        caminho = Path(arq)
        t, lm, modo, texto_app, extras = carregar(caminho)
        modo = args.modo or modo
        com_mao = int((~np.isnan(lm).any(axis=1)).sum())
        fps = (len(t) - 1) / t[-1] if t[-1] > 0 else 0.0
        print(f"\n== {caminho.name}: {t[-1]:.1f} s, {len(t)} quadros (~{fps:.0f} fps), "
              f"mao em {com_mao}, modo {modo}")
        if texto_app:
            print(f"   texto na tela quando salvou: {texto_app}")
        if modo == "ALFABETO":
            final = rodar_alfabeto(t, lm, args.quadros, extras["aspecto"])
        elif modo == "CALIBRADO" and "maos" in extras:
            final = rodar_calibrado(t, extras, args.usuario, Path(args.vocabulario))
        elif "maos" in extras and (PROJECT_DIR / "modelo_palavras_v3.pkl").exists():
            final = rodar_palavras_v3(t, extras)
        else:
            final = rodar_palavras(t, lm)
        print(f"   reconhecido neste trecho: {final or '(nada)'}")


if __name__ == "__main__":
    main()
