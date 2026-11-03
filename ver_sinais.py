#!/usr/bin/env python3
"""
Ver os sinais como o modelo aprendeu: o exemplo mais tipico de cada palavra
nas bases (MINDS-Libras, LIBRAS-EQT-UECE, coleta do grupo), lado a lado, so
com os pontos que o MediaPipe ve (maos e corpo, como num espelho, igual ao
app). Serve para:

  - entender por que o modelo confunde duas palavras (ex.: RAIO e CINCO): a
    tabela no terminal mostra o que muda entre elas (movimento, altura da mao,
    dedos esticados, uma ou duas maos);
  - aprender a fazer o sinal do jeito que o modelo aprendeu;
  - comparar com o SEU sinal: no app (modo PALAVRAS), logo depois do erro,
    aperte S e passe o arquivo salvo em debug/ com --trecho;
  - gerar uma figura para o TCC (--figura), so com os pontos (sem imagem de
    ninguem).

Uso:
  python ver_sinais.py raio cinco digital
  python ver_sinais.py raio cinco --trecho debug/trecho_20261101_171200.npz
  python ver_sinais.py raio cinco digital --figura figura_raio_cinco.png
  python ver_sinais.py --listar                  # as palavras de cada base

Na janela: N / P = o exemplo de outra pessoa (do mais tipico para o menos),
ESPACO = pausa, F = salva a tela em debug/, Q = sai.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import treino_v3 as T
from calibracao import exemplos_por_tipicidade, slug
from libres_features import geometria_mao
from libres_v3 import (OMBRO_D, OMBRO_E, _referencia_corpo, escolher_dominante, medidas_outra_mao,
                       outra_mao_parada, rastrear, trecho_sem_repouso)

PROJECT_DIR = Path(__file__).parent
DEBUG_DIR = PROJECT_DIR / "debug"
FONTE = cv2.FONT_HERSHEY_SIMPLEX
LIGACOES_MAO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                (17, 18), (18, 19), (19, 20)]
LIGACOES_CORPO = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
PONTOS_CORPO = [0, 11, 12, 13, 14, 15, 16, 23, 24]
DEDOS = ["polegar", "indicador", "medio", "anelar", "minimo"]
PAUSA = 0.6                     # s parado no fim de cada passada
# cores (BGR): tela escura como o app; figura clara para o TCC
ESCURO = {"fundo": (32, 28, 26), "texto": (240, 240, 240), "suave": (168, 160, 154), "dom": (0, 230, 0),
          "outra": (255, 200, 0), "ignorada": (95, 95, 95), "corpo": (150, 150, 150), "linha": (63, 56, 52),
          "barra": (200, 110, 190), "seu": (60, 200, 255)}
CLARO = {"fundo": (255, 255, 255), "texto": (25, 25, 25), "suave": (105, 105, 105), "dom": (30, 140, 30),
         "outra": (190, 110, 20), "ignorada": (185, 185, 185), "corpo": (130, 130, 130), "linha": (225, 225, 225),
         "barra": (200, 110, 190), "seu": (0, 120, 220)}


def asc(s: str) -> str:
    """As fontes do OpenCV nao tem acento."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


# ------------------------------------------------------------------ dados
def pastas_de_dados(texto: str | None) -> list[Path]:
    if texto:
        return [Path(p.strip()) for p in texto.split(",") if p.strip()]
    return list(T.DADOS.values())


def espelhar_base(pasta: Path) -> bool:
    """MINDS e EQT vem de videos (sem espelho); a coleta do grupo vem da camera do
    app, que ja e espelhada. Para comparar com o que voce ve no app, os videos
    sao mostrados espelhados."""
    return "coleta" not in pasta.name.lower()


def quem(arq: Path) -> tuple[str, str]:
    """(base, pessoa) pelo nome do arquivo: Sinalizador03 -> MINDS-Libras, sinalizador 3."""
    m = re.search(r"(Sinalizador|Pessoa|Informante)(\d+)", arq.stem, re.I)
    if not m:
        return arq.parent.parent.name, arq.stem[:20]
    base = T.BASE_DO_PREFIXO.get(T.PREFIXO_PESSOA[m.group(1).lower()], "?")
    return base, f"{m.group(1).lower()} {int(m.group(2))}"


def arquivos_da_palavra(palavra: str, pastas: list[Path]) -> list[tuple[Path, bool]]:
    alvo, out = slug(palavra), []
    for base in pastas:
        if not base.is_dir():
            continue
        for pasta in sorted(base.iterdir()):
            if pasta.is_dir() and slug(pasta.name) == alvo:
                out += [(a, espelhar_base(base)) for a in sorted(pasta.glob("*.npz"))]
    return out


# ------------------------------------------------------------------ preparo e medidas
def preparar_exibicao(bruto: dict, espelhar: bool = False, recortar: bool = True) -> dict | None:
    """Quadros crus de um sinal -> o que desenhar: maos em lugares estaveis, sem o
    repouso no comeco e no fim (como o --recortar do modelo), qual e a mao
    dominante, se a outra mao fica parada (o --sem-mao-parada a ignora), o
    retangulo do desenho e as medidas do sinal."""
    asp = float(bruto.get("aspecto") or 16 / 9)
    fps = float(bruto.get("fps") or 30.0)
    fps = fps if 5.0 <= fps <= 120.0 else 30.0
    maos = rastrear(np.asarray(bruto["maos"], np.float64), np.asarray(bruto["lado"]))
    pose = np.asarray(bruto["pose"], np.float64) if bruto.get("pose") is not None else np.empty(0)
    alguma = (~np.isnan(maos[:, :, 0, 0])).any(axis=1)
    if not alguma.any():
        return None
    i0, i1 = int(np.argmax(alguma)), len(alguma) - int(np.argmax(alguma[::-1]))
    m = maos[i0:i1].copy()
    p = pose[i0:i1].copy() if pose.ndim == 3 and len(pose) >= i1 else None
    if espelhar:
        m[..., 0] = 1.0 - m[..., 0]
        if p is not None:
            p[..., 0] = 1.0 - p[..., 0]
    iso = m.copy()
    iso[..., [0, 2]] *= asp                       # x, y e z na mesma escala
    centro, larg, _ = _referencia_corpo(p if p is not None else np.empty(0), asp)
    if recortar and p is not None and np.isfinite(p[:, [OMBRO_E, OMBRO_D], :2]).all(axis=(1, 2)).mean() >= 0.5:
        j0, j1 = trecho_sem_repouso(iso, centro, larg, fps)
        if j1 - j0 >= 3:
            m, iso, p = m[j0:j1], iso[j0:j1], p[j0:j1]
    tem = ~np.isnan(iso[:, :, 0, 0])
    dom = escolher_dominante(iso, sem_tremor=True)
    if tem[:, dom].sum() < 3 and tem[:, 1 - dom].sum() >= 3:
        dom = 1 - dom
    try:
        ignorada = bool(outra_mao_parada(medidas_outra_mao(iso, dom, centro, larg)))
    except Exception:  # noqa: BLE001  (sinal curto demais para medir)
        ignorada = False
    return {"maos": m, "pose": p, "aspecto": asp, "fps": fps, "dom": int(dom), "ignorada": ignorada,
            "caixa": caixa(m, p, asp), "medidas": medir(iso, dom, ignorada, centro, larg, fps)}


def caixa(maos: np.ndarray, pose: np.ndarray | None, asp: float) -> tuple[float, float, float, float]:
    """Retangulo com maos, corpo e o topo da cabeca desenhada (x * aspecto, y)."""
    mm = maos.reshape(-1, 3)
    mm = mm[np.isfinite(mm[:, 0])]
    pts = [np.c_[mm[:, 0] * asp, mm[:, 1]]]
    if pose is not None:
        sel = pose[:, PONTOS_CORPO]
        ok = np.isfinite(sel[..., 0]) & (np.nan_to_num(sel[..., 3]) > 0.5) & (sel[..., 1] < 1.05)
        pts.append(np.c_[sel[ok][:, 0] * asp, sel[ok][:, 1]])
        cab = ok[:, 0] & ok[:, 1] & ok[:, 2]
        if cab.any():
            lo = np.hypot((sel[cab, 1, 0] - sel[cab, 2, 0]) * asp, sel[cab, 1, 1] - sel[cab, 2, 1])
            pts.append(np.c_[sel[cab, 0, 0] * asp, sel[cab, 0, 1] - 0.28 * lo])
    P = np.vstack(pts) if sum(len(x) for x in pts) else np.array([[0.0, 0.0], [asp, 1.0]])
    (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    mx, my = 0.08 * max(x1 - x0, 0.2), 0.08 * max(y1 - y0, 0.2)
    return float(x0 - mx), float(y0 - my), float(x1 + mx), float(y1 + my)


def medir(iso: np.ndarray, dom: int, ignorada: bool, centro: np.ndarray, larg: float, fps: float) -> dict:
    """Numeros que explicam uma confusao: duracao, quanto da para ver as duas
    maos, quanto o punho dominante anda (larguras de ombro), a altura dele
    (larguras de ombro abaixo da linha dos ombros; negativo = acima) e os dedos
    esticados no meio do sinal."""
    n = len(iso)
    tem = ~np.isnan(iso[:, :, 0, 0])
    duas = 0.0 if ignorada else float((tem[:, 0] & tem[:, 1]).mean())
    w = iso[tem[:, dom], dom, 0, :2] / larg
    if len(w) >= 5:                               # sem o tremor da deteccao
        w = np.stack([np.convolve(w[:, j], np.ones(3) / 3, mode="valid") for j in range(2)], axis=1)
    anda = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0
    alt = float(np.median((iso[tem[:, dom], dom, 0, 1] - centro[1]) / larg)) if tem[:, dom].any() else np.nan
    formas = []
    for k in range(int(0.3 * n), max(int(0.7 * n), int(0.3 * n) + 1)):
        if k < n and tem[k, dom]:
            g = geometria_mao(iso[k, dom])
            formas.append(tuple(d for d in DEDOS if g["esticado"][d]))
    dedos = Counter(formas).most_common(1)[0][0] if formas else None
    return {"dur": n / fps, "duas": duas, "anda": anda, "altura": alt, "dedos": dedos}


FORMAS = {(): "fechada", ("indicador", "medio", "anelar", "minimo"): "4 dedos (sem polegar)",
          ("polegar", "minimo"): "polegar+minimo (Y)", ("polegar", "indicador"): "polegar+indicador (L)"}


def nome_dedos(t) -> str:
    if t is None:
        return "?"
    if len(t) == 5:
        return "aberta (5 dedos)"
    return FORMAS.get(tuple(t), "+".join(t))


def onde(alt: float) -> str:
    if not np.isfinite(alt):
        return "?"
    lugar = "rosto" if alt < -0.35 else "peito" if alt < 0.45 else "barriga" if alt < 1.0 else "abaixo"
    return f"{lugar} ({alt:+.1f})"


def resumo_medidas(lista: list[dict]) -> dict:
    """Mediana das medidas de varios exemplos (dedos: a forma mais comum)."""
    if not lista:
        return {}
    med = {k: float(np.nanmedian([d[k] for d in lista])) for k in ("dur", "duas", "anda", "altura")}
    formas = [d["dedos"] for d in lista if d["dedos"] is not None]
    med["dedos"] = Counter(formas).most_common(1)[0][0] if formas else None
    return med


# ------------------------------------------------------------------ o SEU sinal (tecla S do app)
def sinal_do_trecho(arq: Path, qual: int = -1) -> tuple[dict, list]:
    """O sinal (o ultimo, ou o de numero `qual`) de um trecho salvo com S no modo
    PALAVRAS, cortado pelo mesmo segmentador do app, e o que o modelo achou dele."""
    from libres_realtime import SignSegmenterV3, WordRecognizerV3
    with np.load(arq, allow_pickle=False) as d:
        if "maos" not in d.files:
            raise SystemExit(f"{arq.name}: o trecho nao tem as duas maos e o corpo (salve no modo PALAVRAS "
                             "ou CALIBRADO do app atual)")
        t = d["t"].astype(np.float64)
        maos, lado, pose = d["maos"].astype(np.float64), d["lado"], d["pose"].astype(np.float64)
        asp = float(d["aspecto"]) if "aspecto" in d.files else 16 / 9
    seg, sinais = SignSegmenterV3(), []
    for k, ti in enumerate(t):
        tem = not np.isnan(maos[k, :, 0, 0]).all()
        s = seg.update({"maos": maos[k], "lado": lado[k], "pose": pose[k]} if tem else None, ti)
        if s is not None:
            sinais.append(s)
    for _ in range(90):                           # um sinal ainda aberto no fim do trecho
        if not seg.recording:
            break
        t = np.r_[t, t[-1] + 1 / 30]
        s = seg.update(None, t[-1])
        if s is not None:
            sinais.append(s)
    if not sinais:
        dt = np.diff(t)
        sinais = [{"maos": maos, "lado": lado, "pose": pose,
                   "fps": float(1 / np.median(dt)) if len(dt) and np.median(dt) > 0 else 30.0}]
        print("  (o segmentador nao achou um sinal inteiro no trecho: mostrando o trecho todo)")
    print(f"  trecho {arq.name}: {len(sinais)} sinal(is); mostrando o "
          f"{'ultimo' if qual == -1 else f'numero {qual}'}")
    sinal = sinais[qual if -len(sinais) <= qual < len(sinais) else -1]
    sinal["aspecto"] = asp
    top = []
    modelo = PROJECT_DIR / "modelo_palavras_v3.pkl"
    if modelo.exists():
        try:
            top = WordRecognizerV3(modelo).predict(sinal, asp)
        except Exception as e:  # noqa: BLE001
            print(f"  (modelo de palavras nao respondeu: {e})")
    return sinal, top


# ------------------------------------------------------------------ desenho
def desenhar_esqueleto(img: np.ndarray, area: tuple, ex: dict, k: int, cores: dict, esp: float = 1.0) -> None:
    x, y, w, h = area
    bx0, by0, bx1, by1 = ex["caixa"]
    esc = min(w / max(bx1 - bx0, 1e-6), h / max(by1 - by0, 1e-6))
    ox, oy = x + (w - esc * (bx1 - bx0)) / 2, y + (h - esc * (by1 - by0)) / 2
    asp = ex["aspecto"]
    lw = max(1, int(round(2 * esp)))

    def pt(px, py):
        return int(ox + (px * asp - bx0) * esc), int(oy + (py - by0) * esc)

    if ex["pose"] is not None:
        q = ex["pose"][k]

        def ok(i):
            return np.isfinite(q[i, 0]) and np.nan_to_num(q[i, 3]) > 0.5 and q[i, 1] < 1.05

        for a, b in LIGACOES_CORPO:
            if ok(a) and ok(b):
                cv2.line(img, pt(*q[a, :2]), pt(*q[b, :2]), cores["corpo"], lw, cv2.LINE_AA)
        if ok(0):
            raio = 12 * esp
            if ok(11) and ok(12):
                raio = 0.28 * esc * np.hypot((q[11, 0] - q[12, 0]) * asp, q[11, 1] - q[12, 1])
            cv2.circle(img, pt(*q[0, :2]), max(int(raio), 6), cores["corpo"], lw, cv2.LINE_AA)
    for slot in range(2):
        mao = ex["maos"][k, slot]
        if np.isnan(mao[0, 0]):
            continue
        cor = cores["dom"] if slot == ex["dom"] else cores["ignorada"] if ex["ignorada"] else cores["outra"]
        for a, b in LIGACOES_MAO:
            cv2.line(img, pt(*mao[a, :2]), pt(*mao[b, :2]), cor, lw, cv2.LINE_AA)
        for q2 in mao:
            cv2.circle(img, pt(*q2[:2]), max(2, int(3 * esp)), cor, -1, cv2.LINE_AA)


def quadro_no_tempo(ex: dict, t: float) -> tuple[int, float]:
    n = len(ex["maos"])
    dur = n / ex["fps"]
    no_ciclo = t % (dur + PAUSA)
    return min(n - 1, int(no_ciclo * ex["fps"])), min(1.0, no_ciclo / dur) if dur > 0 else 1.0


def linha_medidas(md: dict) -> str:
    if not md:
        return ""
    return (f"{md['dur']:.1f} s | punho anda {md['anda']:.1f} | {onde(md['altura'])} | "
            f"{nome_dedos(md['dedos'])}" + (f" | 2 maos {md['duas'] * 100:.0f}%" if md["duas"] >= 0.2 else ""))


def painel(ex: dict, titulo: str, sub: str, t: float, tam: tuple, cores: dict, destaque: bool = False) -> np.ndarray:
    W, H = tam
    img = np.full((H, W, 3), cores["fundo"], np.uint8)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), cores["seu"] if destaque else cores["linha"], 2 if destaque else 1)
    cv2.putText(img, asc(titulo), (12, 30), FONTE, 0.8, cores["seu"] if destaque else cores["texto"], 2, cv2.LINE_AA)
    cv2.putText(img, asc(sub)[:64], (12, 52), FONTE, 0.45, cores["suave"], 1, cv2.LINE_AA)
    if ex is None:
        cv2.putText(img, "sem exemplo", (12, H // 2), FONTE, 0.6, cores["suave"], 1, cv2.LINE_AA)
        return img
    k, frac = quadro_no_tempo(ex, t)
    desenhar_esqueleto(img, (10, 62, W - 20, H - 62 - 48), ex, k, cores)
    cv2.putText(img, asc(linha_medidas(ex["medidas"]))[:70], (12, H - 26), FONTE, 0.42, cores["suave"], 1,
                cv2.LINE_AA)
    if ex["ignorada"]:
        cv2.putText(img, "outra mao parada: o modelo ignora (cinza)", (12, 74), FONTE, 0.4, cores["suave"], 1,
                    cv2.LINE_AA)
    cv2.rectangle(img, (12, H - 14), (W - 12, H - 9), cores["linha"], -1)
    cv2.rectangle(img, (12, H - 14), (12 + int((W - 24) * frac), H - 9), cores["barra"], -1)
    return img


def grade(imagens: list[np.ndarray], colunas: int, cores: dict) -> np.ndarray:
    h, w = imagens[0].shape[:2]
    linhas = (len(imagens) + colunas - 1) // colunas
    tela = np.full((h * linhas, w * colunas, 3), cores["fundo"], np.uint8)
    for i, im in enumerate(imagens):
        r, c = divmod(i, colunas)
        tela[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return tela


def figura(itens: list[dict], arq: Path, n_quadros: int = 6) -> None:
    """Uma linha por sinal, n_quadros colunas do comeco ao fim (fundo branco)."""
    cw, ch, rot = 210, 190, 270
    cores = CLARO
    linhas = []
    for it in itens:
        ex = it["ex"]
        img = np.full((ch, rot + cw * n_quadros, 3), 255, np.uint8)
        cv2.putText(img, asc(it["titulo"]), (12, 40), FONTE, 0.85, cores["seu"] if it.get("seu") else cores["texto"],
                    2, cv2.LINE_AA)
        for j, ln in enumerate(quebrar(asc(it["sub"]), 34)[:3]):
            cv2.putText(img, ln, (12, 68 + 20 * j), FONTE, 0.45, cores["suave"], 1, cv2.LINE_AA)
        md = ex["medidas"] if ex else {}
        if md:
            extra = [f"punho anda {md['anda']:.1f}", onde(md["altura"]), f"dedos: {nome_dedos(md['dedos'])}"]
            for j, ln in enumerate(extra):
                cv2.putText(img, asc(ln), (12, 130 + 18 * j), FONTE, 0.42, cores["suave"], 1, cv2.LINE_AA)
        if ex is not None:
            n = len(ex["maos"])
            for c, k in enumerate(np.linspace(0, n - 1, n_quadros).round().astype(int)):
                x0 = rot + c * cw
                cv2.rectangle(img, (x0 + 4, 4), (x0 + cw - 4, ch - 4), cores["linha"], 1)
                desenhar_esqueleto(img, (x0 + 10, 10, cw - 20, ch - 34), ex, int(k), cores)
                cv2.putText(img, f"{k / ex['fps']:.1f} s", (x0 + cw - 60, ch - 12), FONTE, 0.42, cores["suave"],
                            1, cv2.LINE_AA)
        linhas.append(img)
    topo = np.full((34, linhas[0].shape[1], 3), 255, np.uint8)
    cv2.putText(topo, "comeco do sinal  ->  fim (sem o repouso)", (rot + 10, 24), FONTE, 0.5, cores["suave"], 1,
                cv2.LINE_AA)
    sep = np.full((2, linhas[0].shape[1], 3), 225, np.uint8)
    partes = [topo]
    for im in linhas:
        partes += [im, sep]
    arq.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(arq), np.vstack(partes[:-1]))


def quebrar(txt: str, n: int) -> list[str]:
    out, linha = [], ""
    for p in txt.split():
        if len(linha) + len(p) + 1 > n and linha:
            out.append(linha)
            linha = p
        else:
            linha = f"{linha} {p}".strip()
    return out + ([linha] if linha else [])


# ------------------------------------------------------------------ programa
def listar(pastas: list[Path]) -> None:
    for base in pastas:
        if not base.is_dir():
            print(f"\n{base.name}/: nao encontrada")
            continue
        palavras = sorted(p for p in base.iterdir() if p.is_dir())
        print(f"\n{base.name}/ ({len(palavras)} palavras): palavra (exemplos/pessoas)")
        itens = []
        for p in palavras:
            arqs = list(p.glob("*.npz"))
            itens.append(f"{p.name} ({len(arqs)}/{len({quem(a)[1] for a in arqs})})")
        for ln in quebrar(", ".join(itens), 100):
            print("  " + ln)


def comparar(itens: list[dict]) -> None:
    """Tabela das medidas e o que mais muda entre a 1a palavra e as outras."""
    print("\nO QUE O MODELO VE (mediana dos exemplos de cada palavra; alturas e caminhos em larguras de ombro)")
    print(f"  {'sinal':14s} {'exemplos':>8s} {'pessoas':>7s} {'duracao':>7s} {'2 maos':>6s} {'punho anda':>10s} "
          f"{'altura do punho':>16s}  dedos esticados (meio do sinal)")
    for it in itens:
        md = it["medidas"]
        if not md:
            print(f"  {asc(it['titulo']):14s} sem dados")
            continue
        print(f"  {asc(it['titulo'])[:14]:14s} {it.get('n', 1):8d} {it.get('pessoas', 1):7d} {md['dur']:6.1f}s "
              f"{md['duas'] * 100:5.0f}% {md['anda']:10.1f} {onde(md['altura']):>16s}  {nome_dedos(md['dedos'])}")
    base = next((it for it in itens if it["medidas"] and not it.get("seu")), None)
    if base is None:
        return
    print()
    for it in itens:
        if it is base or not it["medidas"]:
            continue
        a, b = base["medidas"], it["medidas"]
        dif = []
        if abs(a["anda"] - b["anda"]) >= 0.5:
            dif.append(f"movimento ({a['anda']:.1f} x {b['anda']:.1f})")
        if np.isfinite(a["altura"]) and np.isfinite(b["altura"]) and abs(a["altura"] - b["altura"]) >= 0.4:
            dif.append(f"altura da mao ({onde(a['altura'])} x {onde(b['altura'])})")
        if a["dedos"] != b["dedos"]:
            dif.append(f"dedos ({nome_dedos(a['dedos'])} x {nome_dedos(b['dedos'])})")
        if abs(a["duas"] - b["duas"]) >= 0.3:
            dif.append(f"duas maos ({a['duas'] * 100:.0f}% x {b['duas'] * 100:.0f}%)")
        if abs(a["dur"] - b["dur"]) >= 0.5:
            dif.append(f"duracao ({a['dur']:.1f} x {b['dur']:.1f} s)")
        nome_a, nome_b = asc(base["titulo"]), asc(it["titulo"])
        if dif:
            print(f"  {nome_a} x {nome_b}: muda {'; '.join(dif)}")
        else:
            print(f"  {nome_a} x {nome_b}: quase iguais nesses numeros -> a diferenca esta em detalhes "
                  "(trajetoria, orientacao da palma): sinais faceis de confundir")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("palavras", nargs="*", help="as palavras (como as pastas das bases: raio, cinco, bom_dia...)")
    ap.add_argument("--trecho", type=Path, default=None,
                    help="trecho salvo com S no app (debug/trecho_....npz): mostra o SEU sinal junto")
    ap.add_argument("--sinal", type=int, default=-1, help="qual sinal do trecho (padrao: o ultimo)")
    ap.add_argument("--figura", type=Path, default=None, help="salva uma figura (PNG) para o TCC e sai")
    ap.add_argument("--quadros", type=int, default=6, help="colunas da figura (padrao 6)")
    ap.add_argument("--listar", action="store_true", help="lista as palavras de cada base")
    ap.add_argument("--pastas", default=None, help="pastas das bases, separadas por virgula "
                    "(padrao: minds_v3, eqt_v3 e coleta_v3)")
    ap.add_argument("--sem-recorte", action="store_true", help="mostra o video inteiro (com o repouso)")
    ap.add_argument("--sem-espelho", action="store_true", help="mostra os videos das bases como foram gravados")
    args = ap.parse_args()

    pastas = pastas_de_dados(args.pastas)
    if args.listar:
        listar(pastas)
        return
    if not args.palavras and args.trecho is None:
        ap.error("diga as palavras (ex.: python ver_sinais.py raio cinco) ou use --listar")
    if not any(p.is_dir() for p in pastas):
        sys.exit("Nenhuma base encontrada (" + ", ".join(p.name + "/" for p in pastas) + "). Rode antes "
                 "extrair_minds_v3.py e/ou extrair_eqt_v3.py.")

    recortar = not args.sem_recorte
    itens = []                # um por painel: titulo, sub, ex, medidas, lista de exemplos (para N/P)
    for palavra in args.palavras:
        arqs = arquivos_da_palavra(palavra, pastas)
        espelho = {a: e and not args.sem_espelho for a, e in arqs}
        print(f"  {palavra}: {len(arqs)} exemplo(s); escolhendo o mais tipico...", flush=True)
        lista = exemplos_por_tipicidade([a for a, _ in arqs])
        exibicoes = []
        for _, a, bruto in lista:
            ex = preparar_exibicao(bruto, espelho[a], recortar)
            if ex is not None:
                base, pessoa = quem(a)
                exibicoes.append((ex, base, pessoa))
        it = {"titulo": palavra.replace("_", " ").upper(), "lista": exibicoes, "i": 0,
              "n": len(exibicoes), "pessoas": len({p for _, _, p in exibicoes}),
              "medidas": resumo_medidas([e["medidas"] for e, _, _ in exibicoes])}
        if not exibicoes:
            print(f"  ATENCAO: '{palavra}' nao esta nas bases (veja --listar)")
        itens.append(it)
    if args.trecho is not None:
        sinal, top = sinal_do_trecho(args.trecho, args.sinal)
        ex = preparar_exibicao(sinal, espelhar=False, recortar=recortar)
        sub = "modelo: " + "  ".join(f"{asc(p).upper()} {c * 100:.0f}%" for p, c in top) if top else "seu sinal"
        if top:
            print("  o modelo achou no seu sinal: " + ", ".join(f"{p} {c * 100:.0f}%" for p, c in top))
        itens.append({"titulo": "SEU SINAL", "seu": True, "lista": [(ex, "trecho", sub)] if ex else [], "i": 0,
                      "n": 1, "pessoas": 1, "medidas": ex["medidas"] if ex else {}, "sub_fixo": sub})

    comparar(itens)

    def atual(it):
        if not it["lista"]:
            return None, "sem exemplo nas bases"
        ex, base, pessoa = it["lista"][it["i"]]
        if it.get("seu"):
            return ex, it["sub_fixo"]
        return ex, f"{base} - {pessoa} (exemplo {it['i'] + 1} de {it['n']})"

    if not any(it["lista"] for it in itens):
        sys.exit("\nNenhuma dessas palavras esta nas bases. Veja os nomes com: python ver_sinais.py --listar")
    if args.figura is not None:
        figura([{"titulo": it["titulo"], "sub": atual(it)[1], "ex": atual(it)[0], "seu": it.get("seu")}
                for it in itens], args.figura, max(2, args.quadros))
        print(f"\nFigura salva: {args.figura}")
        fontes = sorted({it["lista"][it["i"]][1] for it in itens if it["lista"] and not it.get("seu")})
        print("Sugestao de legenda: Figura X - Quadros dos sinais " + ", ".join(it["titulo"] for it in itens)
              + " (pontos do MediaPipe)." + (" Fonte: elaborado pelos autores a partir de "
                                             + " e ".join(fontes) + "." if fontes else ""))
        return

    colunas = min(3, len(itens))
    tam = (460, 420)
    janela = "Libre-se - como o modelo aprendeu os sinais"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print("\nN/P = outra pessoa | ESPACO = pausa | F = salva a tela | Q = sai")
    t0, pausado, t_pausa = time.time(), False, 0.0
    while True:
        t = (t_pausa if pausado else time.time()) - t0
        imgs = []
        for it in itens:
            ex, sub = atual(it)
            imgs.append(painel(ex, it["titulo"], sub, t, tam, ESCURO, it.get("seu", False)))
        tela = grade(imgs, colunas, ESCURO)
        cv2.imshow(janela, tela)
        tecla = cv2.waitKey(15) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord(" "):
            if pausado:
                t0 += time.time() - t_pausa          # continua de onde parou
            else:
                t_pausa = time.time()
            pausado = not pausado
        elif tecla in (ord("n"), ord("p")):
            for it in itens:
                if not it.get("seu") and it["lista"]:
                    it["i"] = (it["i"] + (1 if tecla == ord("n") else -1)) % len(it["lista"])
            t0 = t_pausa = time.time()               # o novo exemplo comeca do inicio
        elif tecla == ord("f"):
            DEBUG_DIR.mkdir(exist_ok=True)
            arq = DEBUG_DIR / f"ver_sinais_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(arq), tela)
            print(f"  tela salva: {arq}")
        try:
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
