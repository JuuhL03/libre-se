#!/usr/bin/env python3
"""
Teste de ponta a ponta das PALAVRAS v3 (duas maos + corpo) SEM os videos do MINDS.

Gera "sinais" sinteticos com uma mao procedural e um corpo (nariz e ombros) e
passa por todo o caminho de verdade, numa pasta temporaria (nao mexe em
minds_v3/ nem nos modelos):

  1. extracao: um .mp4 dentro de um ZIP com nome de sinalizador do MINDS, com o
     MediaPipe trocado por um detector falso (confere ZIP, video e .npz);
  2. avaliacao com pessoa fora do treino (avaliar_v3.py --comparar). Os sinais
     foram montados para que a v2 NAO consiga separar dois pares:
       rosto x peito   - mesma mao e mesmo movimento, perto do rosto ou do peito;
       juntas x aberta - mesma mao dominante, com ou sem a outra mao;
     a v3 tem de separar (local em relacao ao corpo e a segunda mao);
  3. treino do modelo final (treino_v3.py);
  4. camera simulada: os quadros de pessoas novas entram um a um no
     SignSegmenterV3 e cada sinal recortado vai para o WordRecognizerV3.

Uso:  python teste_v3.py            (5 a 15 min, conforme o computador)
      python teste_v3.py --rapido   (menos repeticoes)

Os numeros daqui NAO sao resultado do TCC (os sinais sao sinteticos); servem
para provar que o codigo faz o que deveria antes de gastar horas na extracao.
"""

import argparse
import contextlib
import io
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

PALAVRAS = ["rosto", "peito", "juntas", "aberta", "circulo", "fecha"]
FPS = 30


# ------------------------------------------------------------------ mao e corpo
def mao_procedural(flex: np.ndarray) -> np.ndarray:
    """21 pontos de uma mao com a palma para a camera, punho na origem,
    punho -> base do medio = 1, y para baixo. flex[d] = 0 (esticado) a 1
    (dobrado) para polegar, indicador, medio, anelar e minimo."""
    pts = np.zeros((21, 3))
    bases = {1: (-0.32, -0.28), 5: (-0.33, -0.93), 9: (-0.06, -1.0), 13: (0.19, -0.95), 17: (0.40, -0.84)}
    comp = {1: (0.36, 0.30, 0.26), 5: (0.43, 0.26, 0.21), 9: (0.48, 0.29, 0.23),
            13: (0.44, 0.27, 0.21), 17: (0.34, 0.21, 0.19)}
    for d, b in enumerate((1, 5, 9, 13, 17)):
        p = np.array([*bases[b], 0.0])
        pts[b] = p
        direc = np.array([-0.8, -0.6, 0.0]) if d == 0 else np.array([bases[b][0] * 0.25, -1.0, 0.0])
        direc /= np.linalg.norm(direc)
        ang = 0.0
        for j, L in enumerate(comp[b]):
            ang += flex[d] * np.radians(75 if j == 0 else 85)
            v = direc * np.cos(ang) + np.array([0.0, 0.0, -1.0]) * np.sin(ang)
            p = p + L * v
            pts[b + 1 + j] = p
    return pts


INDICADOR = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
ABERTA = np.array([0.05, 0.05, 0.05, 0.05, 0.05])


class Pessoa:
    """Uma pessoa sintetica: onde esta na imagem, tamanho, rapidez, tremor,
    canhota ou nao, e a camera (16:9 ou 4:3)."""

    def __init__(self, sid: int, rng: np.random.Generator, canhota: bool = False, quatro_tres: bool = False):
        self.sid = sid
        self.W, self.H = (640, 480) if quatro_tres else (1280, 720)
        self.cx = self.W * rng.uniform(0.42, 0.58)
        self.sw = self.W * rng.uniform(0.22, 0.30) * (1.25 if quatro_tres else 1.0)   # largura dos ombros
        self.sy = self.H * rng.uniform(0.66, 0.78)
        self.esc_mao = rng.uniform(0.9, 1.1)
        self.rapidez = rng.uniform(0.8, 1.25)
        self.tremor = rng.uniform(1.0, 3.0) * self.W / 1280
        self.inclina = np.radians(rng.uniform(-15, 15))
        self.canhota = canhota
        self.desvio = {p: rng.uniform(-0.06, 0.06, 2) for p in PALAVRAS}

    @property
    def aspecto(self) -> float:
        return self.W / self.H

    def pose(self) -> np.ndarray:
        """33 pontos do MediaPipe Pose (so nariz e ombros importam aqui)."""
        p = np.zeros((33, 4))
        nariz = (self.cx, self.sy - 0.75 * self.sw)
        oe, od = (self.cx + self.sw / 2, self.sy), (self.cx - self.sw / 2, self.sy)
        p[:, :2] = nariz
        p[11, :2], p[12, :2] = oe, od
        p[:, 0] /= self.W
        p[:, 1] /= self.H
        p[:, 3] = 1.0
        return p

    def mao_na_imagem(self, flex, punho_corpo, outra=False, rng=None) -> np.ndarray:
        """Mao em coordenadas do MediaPipe (x/largura, y/altura, z/largura).
        punho_corpo: punho em larguras de ombro a partir do centro dos ombros."""
        r = mao_procedural(flex) * 0.30 * self.sw * self.esc_mao
        if outra:                                  # a outra mao e o espelho
            r[:, 0] *= -1
        c, s = np.cos(self.inclina), np.sin(self.inclina)
        r[:, :2] = r[:, :2] @ np.array([[c, -s], [s, c]]).T
        px = np.array([punho_corpo[0], punho_corpo[1]])
        if self.canhota:                           # canhota: tudo espelhado
            r[:, 0] *= -1
            px[0] *= -1
        r[:, 0] += self.cx + px[0] * self.sw
        r[:, 1] += self.sy + px[1] * self.sw
        if rng is not None:
            r += rng.normal(0, self.tremor, r.shape)
        r[:, 0] /= self.W
        r[:, 1] /= self.H
        r[:, 2] /= self.W
        return r


def trajeto(palavra: str, t: np.ndarray, desvio: np.ndarray):
    """-> lista de (flex, punho) por quadro para a mao dominante e para a outra
    (None quando o sinal e de uma mao). t vai de 0 a 1."""
    dom, outra = [], []
    for u in t:
        if palavra in ("rosto", "peito"):
            y0 = -0.55 if palavra == "rosto" else 0.35
            toque = 0.09 * abs(np.sin(2 * np.pi * 2 * u))
            dom.append((INDICADOR, (0.05 + desvio[0], y0 + desvio[1] - toque)))
            outra.append(None)
        elif palavra in ("juntas", "aberta"):
            x = 0.75 - 0.6 * u
            dom.append((ABERTA, (x + desvio[0], 0.35 + desvio[1])))
            outra.append((ABERTA, (-x + desvio[0], 0.35 + desvio[1])) if palavra == "juntas" else None)
        elif palavra == "circulo":
            a = 2 * np.pi * u
            dom.append((INDICADOR, (0.35 + 0.18 * np.cos(a) + desvio[0], -0.1 + 0.18 * np.sin(a) + desvio[1])))
            outra.append(None)
        elif palavra == "fecha":
            dom.append((ABERTA + (1 - ABERTA) * min(1.0, 1.4 * u), (0.3 + desvio[0], 0.05 + 0.35 * u + desvio[1])))
            outra.append(None)
        else:
            raise ValueError(palavra)
    return dom, outra


def gerar_video(pessoa: Pessoa, palavra: str, rng: np.random.Generator, vazio: float = 0.2) -> dict:
    """Quadros de um video do sinal (com quadros sem mao antes e depois), no
    formato de minds_v3/*.npz."""
    n = int(round(1.2 / pessoa.rapidez * rng.uniform(0.9, 1.1) * FPS))
    t = np.linspace(0, 1, n)
    dom, outra = trajeto(palavra, t, pessoa.desvio[palavra] + rng.normal(0, 0.02, 2))
    nv = int(vazio * FPS)
    M, L, P = [], [], []
    pose = pessoa.pose()
    nada = np.full((2, 21, 3), np.nan)
    for _ in range(nv):
        M.append(nada); L.append([-1, -1]); P.append(pose)
    lado_dom, lado_out = (0, 1) if pessoa.canhota else (1, 0)
    for k in range(n):
        maos = np.full((2, 21, 3), np.nan)
        lado = np.array([-1, -1])
        itens = [(dom[k], False, lado_dom)]
        if outra[k] is not None:
            itens.append((outra[k], True, lado_out))
        if rng.random() < 0.5:                      # o MediaPipe devolve em qualquer ordem
            itens = itens[::-1]
        for slot, ((flex, punho), eh_outra, rot) in enumerate(itens):
            if rng.random() < 0.03:                 # falha de deteccao
                continue
            maos[slot] = pessoa.mao_na_imagem(flex, punho, eh_outra, rng)
            lado[slot] = rot if rng.random() > 0.05 else 1 - rot    # rotulo as vezes troca
        M.append(maos); L.append(lado); P.append(pose + rng.normal(0, 0.002, pose.shape))
    for _ in range(nv):
        M.append(nada); L.append([-1, -1]); P.append(pose)
    return {"maos": np.array(M, np.float32), "lado": np.array(L, np.int8),
            "conf_lado": np.ones((len(M), 2), np.float32), "pose": np.array(P, np.float32),
            "aspecto": np.float32(pessoa.aspecto), "fps": np.float32(FPS)}


def pessoas(ids, rng) -> list[Pessoa]:
    out = []
    for k, sid in enumerate(ids):
        out.append(Pessoa(sid, rng, canhota=(k % 3 == 2), quatro_tres=(k % 4 == 1)))
    return out


# ------------------------------------------------------------------ partes
def parte_extracao(tmp: Path) -> bool:
    """ZIP com nome do MINDS -> listar_tarefas -> processar -> .npz."""
    import cv2
    import extrair_minds_v3 as E

    pasta = tmp / "minds_videos"
    pasta.mkdir()
    video = tmp / "video.mp4"
    W, H, n = 320, 180, 40
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(img, (60 + 4 * k, 90), 12, (200, 200, 200), -1)
        vw.write(img)
    vw.release()
    with zipfile.ZipFile(pasta / "Sinalizador07.zip", "w") as z:
        z.write(video, "Sinalizador07/07AcontecerSn07-1.mp4")
        z.write(video, "__MACOSX/Sinalizador07/._07AcontecerSn07-1.mp4")     # lixo do macOS

    rng = np.random.default_rng(5)
    p = Pessoa(7, rng)
    chamadas = {"n": 0}

    def detector_falso(_m, _p, rgb):
        k = chamadas["n"]
        chamadas["n"] += 1
        maos = np.full((2, 21, 3), np.nan, np.float32)
        lado = np.full(2, -1, np.int8)
        if 5 <= k < 35:
            maos[0] = p.mao_na_imagem(INDICADOR, (0.1 + 0.01 * k, 0.2), rng=rng)
            lado[0] = 1
        return maos, lado, np.ones(2, np.float32), p.pose().astype(np.float32)

    E.OUTPUT_DIR = tmp / "minds_v3_extraido"
    E.detectar_quadro = detector_falso
    tarefas, ignorados = E.listar_tarefas(pasta)
    ok = len(tarefas) == 1 and tarefas[0][2] == "acontecer"
    status, saida, nq = E.processar(tarefas[0]) if tarefas else ("sem tarefa", "", 0)
    ok = ok and status == "ok" and nq == 30
    if ok:
        with np.load(saida) as d:
            ok = (d["maos"].shape == (n, 2, 21, 3) and d["pose"].shape == (n, 33, 4)
                  and abs(float(d["aspecto"]) - W / H) < 1e-3 and "Sinalizador07" in Path(saida).name)
    print(f"  tarefas={len(tarefas)} ignorados={len(ignorados)} status={status} quadros_com_mao={nq}")
    return ok


def gerar_base(tmp: Path, ids, reps: int, rng) -> Path:
    base = tmp / "minds_v3"
    for p in pessoas(ids, rng):
        for palavra in PALAVRAS:
            (base / palavra).mkdir(parents=True, exist_ok=True)
            for r in range(reps):
                np.savez_compressed(base / palavra / f"minds_Sinalizador{p.sid:02d}_{palavra}_{r}.npz",
                                    **gerar_video(p, palavra, rng))
    return base


def parte_avaliacao(base: Path) -> tuple[bool, str]:
    import avaliar_v3
    import treino_v3 as T
    T.DADOS_DIRS = [base]
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["avaliar_v3.py", "--comparar"]
    try:
        with contextlib.redirect_stdout(buf):
            avaliar_v3.main()
    finally:
        sys.argv = argv
    saida = buf.getvalue()
    linha = [l for l in saida.splitlines() if l.startswith("v2 (uma mao)")]
    resumo = linha[0] if linha else "(sem linha de comparacao)"
    import re
    m = re.search(r"v2 \(uma mao\): ([\d.]+)%.*v3 \(duas maos \+ corpo\): ([\d.]+)%", resumo)
    if not m:
        print(saida)
        return False, resumo
    v2, v3 = float(m.group(1)), float(m.group(2))
    # "Acerto por palavra" da v3 (vem antes da comparacao com a v2)
    trecho = saida.split("Acerto por palavra", 1)[-1].split("COMPARACAO", 1)[0]
    for l in trecho.splitlines():
        if l.startswith("  ") and "sinais)" in l:
            print("   ", l.strip())
    return v3 >= 90.0 and v3 >= v2 + 15.0, resumo


def parte_treino(base: Path, tmp: Path) -> bool:
    import treino_v3 as T
    T.DADOS_DIRS = [base]
    T.MODELO_V3 = tmp / "modelo_palavras_v3.pkl"
    argv = sys.argv
    sys.argv = ["treino_v3.py"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            T.main()
    finally:
        sys.argv = argv
    return T.MODELO_V3.exists()


def parte_camera(modelo: Path, rng) -> tuple[bool, str]:
    """Pessoas novas, quadro a quadro: mao some entre os sinais (como abaixar a mao)."""
    from libres_realtime import SignSegmenterV3, WordRecognizerV3
    rec = WordRecognizerV3(modelo)
    certos = total = recortes = 0
    erros = []
    for p in pessoas([21, 22, 23], rng):
        seg = SignSegmenterV3()
        t = 0.0
        ordem = list(PALAVRAS) * 2
        rng.shuffle(ordem)
        for palavra in ordem:
            v = gerar_video(p, palavra, rng, vazio=0.5)
            obtidos = []
            for k in range(len(v["maos"])):
                quadro = {"maos": v["maos"][k], "lado": v["lado"][k], "pose": v["pose"][k]}
                if np.isnan(v["maos"][k][:, 0, 0]).all():
                    quadro = None
                sinal = seg.update(quadro, t)
                t += 1 / FPS
                if sinal is not None:
                    obtidos.append(sinal)
            total += 1
            recortes += len(obtidos)
            if len(obtidos) == 1:
                top = rec.predict(obtidos[0], p.aspecto)
                if top and top[0][0] == palavra:
                    certos += 1
                else:
                    erros.append(f"{palavra}->{top[0][0] if top else '?'}")
            else:
                erros.append(f"{palavra}: {len(obtidos)} recortes")
    resumo = f"{certos}/{total} sinais certos ({recortes} recortes)" + (f"; erros: {', '.join(erros[:6])}" if erros else "")
    return certos >= 0.9 * total, resumo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="menos repeticoes por palavra")
    ap.add_argument("--guardar-modelo", default=None, metavar="ARQ",
                    help="copia o modelo sintetico para ARQ (para testes da camera)")
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    resultados = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        print("[1/4] Extracao (ZIP do MINDS, detector falso)...")
        ok = parte_extracao(tmp)
        resultados.append(("extracao", ok))

        print("[2/4] Avaliacao com pessoa fora do treino (v2 x v3)...")
        base = gerar_base(tmp, [1, 2, 3, 4, 5, 6], 3 if args.rapido else 4, rng)
        ok, resumo = parte_avaliacao(base)
        print(f"  {resumo}")
        resultados.append(("avaliacao v3 > v2", ok))

        print("[3/4] Treino do modelo final...")
        ok = parte_treino(base, tmp)
        resultados.append(("treino", ok))

        print("[4/4] Camera simulada (pessoas novas, quadro a quadro)...")
        if ok:
            if args.guardar_modelo:
                import shutil
                shutil.copy(tmp / "modelo_palavras_v3.pkl", args.guardar_modelo)
            ok, resumo = parte_camera(tmp / "modelo_palavras_v3.pkl", rng)
            print(f"  {resumo}")
        resultados.append(("camera", ok))

    print("\n" + "=" * 60)
    for nome, ok in resultados:
        print(f"  {'OK   ' if ok else 'FALHOU'}  {nome}")
    print(f"  ({(time.time() - t0) / 60:.1f} min)")
    print("=" * 60)
    sys.exit(0 if all(ok for _, ok in resultados) else 1)


if __name__ == "__main__":
    main()
