#!/usr/bin/env python3
"""
Teste do soletrador (modo ALFABETO) sem camera: monta sequencias de letras
com poses REAIS dos datasets (coleta_alfabeto/), adiciona tremor de MediaPipe,
falhas de deteccao, transicoes suaves entre letras e o movimento de H, J, K, X
e Z, e confere o texto que o app montaria.

Uso:
  python teste_soletrador.py                       # tremor 1, 2 e 3 px; 30 e 15 fps
  python teste_soletrador.py --sementes 0,11,22,33 --tremor 1,2,3,4
  python teste_soletrador.py --falhas 0.08         # 8% dos quadros sem mao + rajadas no gesto

O tremor e o desvio-padrao do ruido por coordenada, em pixels de uma imagem de
1280 px de largura (MediaPipe em modo IMAGE treme ~1-3 px com a mao parada).
Os movimentos seguem a descricao das letras na Libras: J desenha o J com o
minimo; Z desenha o Z com o indicador; X com o indicador em gancho indo para
tras (e a variante com o indicador esticado); K com indicador e medio
levantados subindo e descendo; H com a mesma forma, girando a mao para dentro.
"""

import argparse
from pathlib import Path

import sys

import numpy as np

import train_alfabeto as T
from libres_features import familia_forma, forma_mao
from libres_realtime import LetterRecognizer, Soletrador

PROJECT_DIR = Path(__file__).parent
CENTRO = np.array([0.5, 0.55, 0.0])
ESC = 0.12                      # tamanho da mao na imagem
LARGURA_PX = 1280


def para_imagem(pose: np.ndarray, estado=(0.0, 0.0, 1.0, 0.0)) -> np.ndarray:
    """estado = (dx, dy em maos, escala, giro em graus em torno do eixo vertical)."""
    dx, dy, esc, giro = estado
    r = pose.reshape(21, 3).copy()
    if giro:
        th = np.radians(giro)
        x, z = r[:, 0].copy(), r[:, 2].copy()
        r[:, 0], r[:, 2] = x * np.cos(th) + z * np.sin(th), -x * np.sin(th) + z * np.cos(th)
    r = r * ESC * esc + CENTRO
    r[:, :2] += np.array([dx, dy]) * ESC
    return r.reshape(-1)


def dobrar_indicador(raw: np.ndarray, ang_pip: float = 80, ang_dip: float = 45) -> np.ndarray:
    """Indicador em gancho a partir de uma pose com o indicador esticado."""
    r = np.asarray(raw, dtype=np.float64).reshape(21, 3).copy()
    normal = np.cross(r[5] - r[0], r[17] - r[0])
    normal /= np.linalg.norm(normal)

    def girar(p, centro, eixo, graus):
        th = np.radians(graus)
        v = p - centro
        return centro + v * np.cos(th) + np.cross(eixo, v) * np.sin(th) + eixo * np.dot(eixo, v) * (1 - np.cos(th))

    eixo = np.cross(r[6] - r[5], normal)
    eixo /= np.linalg.norm(eixo)
    r[7], r[8] = girar(r[7], r[6], eixo, ang_pip), girar(r[8], r[6], eixo, ang_pip)
    r[8] = girar(r[8], r[7], eixo, ang_dip)
    return r.reshape(-1)


def trajeto(nome: str, dur: float, fps: int, ini) -> np.ndarray:
    """Estados (dx, dy, escala, giro) quadro a quadro durante o gesto."""
    n = max(int(dur * fps), 2)
    t = np.linspace(0, 1, n)
    dx = np.zeros(n)
    dy = np.zeros(n)
    esc = np.ones(n)
    giro = np.zeros(n)
    if nome == "J":       # desce e faz a curva
        dx, dy = -np.sin(t * np.pi) * 0.9 * (t > 0.4), t * 1.6
    elif nome == "JL":    # J pequeno e devagar (~1 mao/s)
        dx, dy = -np.sin(t * np.pi) * 0.5 * (t > 0.4), t * 1.0
    elif nome == "Z":     # direita, diagonal para baixo-esquerda, direita
        pts = np.array([[0, 0], [1.6, 0], [0, 1.4], [1.6, 1.4]])
        seg = np.minimum((t * 3).astype(int), 2)
        f = t * 3 - seg
        c = pts[seg] + (pts[seg + 1] - pts[seg]) * f[:, None]
        dx, dy = c[:, 0], c[:, 1]
    elif nome == "X":     # uma curva pequena para o lado
        dx, dy = np.sin(t * np.pi) * 0.8, -t * 0.7
    elif nome == "X2":    # X sacudido 2 vezes na horizontal (quase nao desce)
        dx, dy = np.sin(t * 4 * np.pi) * 0.6, t * 0.15
    elif nome == "XT":    # X indo para tras: a mao diminui na imagem
        esc, dy = 1 - 0.22 * t, t * 0.15
    elif nome == "H":     # movimento lateral
        dx = np.sin(t * 2 * np.pi) * 0.9
    elif nome == "HG":    # H girando a mao para dentro
        giro, dx = 75 * t, 0.15 * t
    elif nome == "K":     # sobe e desce
        dy = -np.sin(t * np.pi) * 0.8
    else:
        raise ValueError(nome)
    x0, y0, e0, g0 = ini
    return np.c_[x0 + dx, y0 + dy, e0 * esc, g0 + giro]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sementes", default="0", help="sementes do ruido, ex.: 0,11,22")
    ap.add_argument("--tremor", default="1,2,3", help="tremor em px, ex.: 1,2,3,4")
    ap.add_argument("--fps", default="30,15")
    ap.add_argument("--falhas", default="0", help="fracao de quadros sem mao, ex.: 0,0.08")
    ap.add_argument("--casos", default="", help="so os casos que comecam com estes nomes (virgula)")
    ap.add_argument("--detalhe", action="store_true", help="mostra as medidas de cada gesto")
    ap.add_argument("--z", type=float, default=1.0,
                    help="multiplica a profundidade (z) dos pontos: 0.5 simula o MediaPipe "
                         "subestimando a profundidade (afeta a mao girando)")
    args = ap.parse_args()
    sementes = [int(x) for x in args.sementes.split(",")]
    tremores = [float(x) for x in args.tremor.split(",")]
    fpss = [int(x) for x in args.fps.split(",")]
    falhas = [float(x) for x in args.falhas.split(",")]

    rec = LetterRecognizer(PROJECT_DIR / "modelo_alfabeto.pkl")
    X, y, _ = T.carregar_coleta()
    if not X:
        sys.exit("Este teste usa poses reais do alfabeto (pasta coleta_alfabeto/, de coletar_alfabeto.py ou "
                 "importar_imagens_alfabeto.py), que nao vai para o GitHub: rode no computador que tem a pasta.")
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
    # K/H: indicador e medio levantados, polegar no medio (forma das duas letras)
    pose_K = None
    for k in np.where(y == "K")[0]:
        raw = para_imagem(X[k])
        top = {l for l, _ in rec.predict_top(raw, 3)}
        if familia_forma(forma_mao(raw)) == "DOIS" and top & {"K", "H"}:
            pose_K = X[k]
            break
    pose_gancho = next(g for g in (dobrar_indicador(X[k]) for k in np.where(np.isin(y, ["L", "G"]))[0])
                       if forma_mao(para_imagem(g)) == "GANCHO")
    faltam = set("ABCDEILORSUVWY") - set(B)
    if faltam or pose_K is None:
        raise SystemExit(f"sem pose reconhecida para: {''.join(sorted(faltam))} {'K' if pose_K is None else ''}")

    H_ = 0.6
    casos = [
        ("UVA", [(B["U"], H_, None), (B["V"], H_, None), (B["A"], H_, None)], "UVA"),
        ("RUA", [(B["R"], H_, None), (B["U"], H_, None), (B["A"], H_, None)], "RUA"),
        ("BOLA", [(B["B"], H_, None), (B["O"], H_, None), (B["L"], H_, None), (B["A"], H_, None)], "BOLA"),
        ("SOL", [(B["S"], H_, None), (B["O"], H_, None), (B["L"], H_, None)], "SOL"),
        ("LIVRE", [(B["L"], H_, None), (B["I"], H_, None), (B["V"], H_, None), (B["R"], H_, None),
                   (B["E"], H_, None)], "LIVRE"),
        ("Y parado", [(B["Y"], 1.0, None)], "Y"),
        ("J (I + movimento)", [(pose_I, .3, None), (pose_I, .8, "J"), (pose_I, .5, None)], "J"),
        ("J segurando I antes", [(pose_I, .6, None), (pose_I, .8, "J"), (pose_I, .5, None)], "J"),
        ("J pequeno e lento", [(pose_I, .6, None), (pose_I, 1.0, "JL"), (pose_I, .5, None)], "J"),
        ("J depois de 2 s de I", [(pose_I, 2.0, None), (pose_I, .8, "J"), (pose_I, .5, None)], "J"),
        ("I parado", [(pose_I, 1.0, None)], "I"),
        ("Z (zigue-zague)", [(pose_ind, .3, None), (pose_ind, 1.0, "Z"), (pose_ind, .5, None)], "Z"),
        ("X (indicador de lado)", [(pose_ind, .3, None), (pose_ind, .6, "X"), (pose_ind, .7, None)], "X"),
        ("X sacudido 2x", [(pose_ind, .3, None), (pose_ind, .8, "X2"), (pose_ind, .7, None)], "X"),
        ("X (gancho para tras)", [(pose_gancho, .3, None), (pose_gancho, .5, "XT"), (pose_gancho, .5, None)], "X"),
        ("X (gancho de lado)", [(pose_gancho, .3, None), (pose_gancho, .6, "X"), (pose_gancho, .5, None)], "X"),
        ("H (U + lateral)", [(pose_U, .3, None), (pose_U, .6, "H"), (pose_U, .7, None)], "H"),
        ("H (gira a mao)", [(pose_K, .3, None), (pose_K, .6, "HG"), (pose_K, .5, None)], "H"),
        ("K (sobe e desce)", [(pose_K, .3, None), (pose_K, .8, "K"), (pose_K, .5, None)], "K"),
        ("U parado", [(pose_U, 1.0, None)], "U"),
        ("OSSO", [(B["O"], H_, None), (B["S"], H_, None), ("fora", .4), (B["S"], H_, None),
                  (B["O"], H_, None)], "OSSO"),
        ("H e abaixa a mao", [(pose_U, .3, None), (pose_U, .6, "H"), ("fora", .4)], "H"),
        ("Z e abaixa a mao", [(pose_ind, .3, None), (pose_ind, 1.0, "Z"), ("fora", .4)], "Z"),
        ("JUIZ", [(pose_I, .3, None), (pose_I, .8, "J"), (pose_I, .3, None), (pose_U, H_, None),
                  (pose_I, H_, None), (pose_ind, .3, None), (pose_ind, 1.0, "Z"), (pose_ind, .5, None)], "JUIZ"),
        ("HOJE", [(pose_K, .3, None), (pose_K, .6, "HG"), (pose_K, .3, None), (B["O"], H_, None),
                  (pose_I, .3, None), (pose_I, .8, "J"), (pose_I, .3, None), (B["E"], H_, None)], "HOJE"),
        ("KIWI", [(pose_K, .3, None), (pose_K, .8, "K"), (pose_K, .3, None), (pose_I, H_, None),
                  (B["W"], H_, None), (pose_I, H_, None)], "KIWI"),
        ("AXE", [(B["A"], H_, None), (pose_gancho, .3, None), (pose_gancho, .5, "XT"), (pose_gancho, .3, None),
                 (B["E"], H_, None)], "AXE"),
    ]
    if args.casos:
        prefixos = [c.strip() for c in args.casos.split(",")]
        casos = [c for c in casos if any(c[0].startswith(p) for p in prefixos)]

    def rodar(roteiro, fps, ruido, p_falha, rng, transicao=0.25) -> str:
        """roteiro: (pose, segundos, movimento|None) ou ("fora", segundos).
        Entre poses diferentes, a mao passa suavemente de uma para a outra
        (e volta a posicao/escala/giro normais)."""
        s = Soletrador(rec)
        txt: list[str] = []
        t = 0.0
        anterior, estado = None, np.array([0.0, 0.0, 1.0, 0.0])
        rajada = [0]

        def quadro(raw, em_gesto=False):
            nonlocal t
            if raw is not None and p_falha > 0:      # falhas de deteccao do MediaPipe
                if rajada[0] > 0:
                    rajada[0] -= 1
                    raw = None
                elif em_gesto and rng.random() < 2 * p_falha:
                    rajada[0] = int(rng.integers(1, 4))   # gesto rapido: 2 a 4 quadros seguidos
                    raw = None
                elif rng.random() < p_falha:
                    raw = None
            info = s.update(raw, t)
            t += 1 / fps
            if info["acao"] == "troca" and txt:
                txt[-1] = info["letra"]
            elif info["acao"] in ("add", "troca"):
                txt.append(info["letra"])

        def tremido(pose, est):
            r = para_imagem(pose, est)
            if args.z != 1.0:
                r[2::3] *= args.z
            return r + rng.normal(0, ruido, 63)

        for passo in roteiro:
            if isinstance(passo[0], str):            # ("fora", segundos)
                for _ in range(int(passo[1] * fps)):
                    quadro(None)
                anterior, estado = None, np.array([0.0, 0.0, 1.0, 0.0])
                continue
            pose, dur, mov = passo
            if anterior is not None and not np.array_equal(anterior, pose):
                n = int(transicao * fps)
                alvo = np.array([estado[0], estado[1], 1.0, 0.0])
                for k in range(1, n + 1):
                    a = k / n
                    quadro(tremido(anterior * (1 - a) + pose * a, estado * (1 - a) + alvo * a))
                estado = alvo
            if mov:
                caminho = trajeto(mov, dur, fps, estado)
                for est in caminho:
                    quadro(tremido(pose, est), em_gesto=True)
                estado = caminho[-1]
            else:
                for _ in range(int(dur * fps)):
                    quadro(tremido(pose, estado))
            anterior = pose
        if anterior is not None:          # a pessoa continua com a mao ali mais um pouco
            for _ in range(int(0.4 * fps)):
                quadro(tremido(anterior, estado))
        for _ in range(int(1.0 * fps)):   # e no fim abaixa a mao (a camera continua rodando)
            quadro(None)
        return "".join(txt)

    if args.detalhe:                      # mostra as medidas dos gestos (para ajustar limiares)
        import libres_realtime as LR
        original = LR._Segmento.letra

        def com_log(self):
            r = original(self)
            i, j = getattr(self, "nucleo_ij", None) or (0, len(self.formas) - 1)
            m = self.medidas(i, j) if self.formas else {}
            print(f"      gesto -> {r}  forma={self.predominio(0.6, i, j)}  nucleo={i}-{j}/{len(self.formas)}  " +
                  "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
            return r
        LR._Segmento.letra = com_log

    total = certos = 0
    for semente in sementes:
        rng = np.random.default_rng(semente)
        for pf in falhas:
            for px in tremores:
                for fps in fpss:
                    erros = []
                    for nome, rot, esperado in casos:
                        if args.detalhe:
                            print(f"  [{nome}]")
                        saida = rodar(rot, fps, px / LARGURA_PX, pf, rng)
                        if saida != esperado:
                            erros.append(f"{nome} -> {saida!r}")
                    ok = len(casos) - len(erros)
                    total += len(casos)
                    certos += ok
                    print(f"semente {semente:>2} | falhas {pf:.2f} | tremor {px:.0f} px | {fps} fps: "
                          f"{ok}/{len(casos)}  " + ("; ".join(erros) if erros else "tudo certo"), flush=True)
    print(f"\nTOTAL: {certos}/{total} sequencias corretas ({100 * certos / total:.1f}%)")


if __name__ == "__main__":
    main()
