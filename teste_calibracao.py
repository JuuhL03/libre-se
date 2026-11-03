#!/usr/bin/env python3
"""
Teste da CALIBRACAO (calibracao.py) sem camera, com uma pessoa sintetica.

Os "sinais" sao aproximacoes feitas a mao (mesma mao procedural do teste_v3.py)
para ter os tipos de sinal que importam na cena da padaria:
  - parados (numeros 1 a 5: so a forma da mao muda);
  - com troca de forma (oi: O -> I);
  - compostos que comecam igual (bom dia, boa tarde, boa noite: mesmo BOM);
  - de uma e de duas maos, perto do rosto ou do peito, com e sem repeticao.
Cada exemplo e um fluxo de camera: a mao entra por baixo, faz o sinal e sai,
quadro a quadro, pelo MESMO segmentador do app (modo CALIBRADO).

O que mede:
  1. acerto com 3, 5 e 8 exemplos por palavra (mesma pessoa, sinais novos);
  2. sinais que NAO foram calibrados: quantos o app recusa ("nao reconhecido");
  3. outra pessoa usando a calibracao da primeira (por que calibrar).

Uso:  python teste_calibracao.py            (~1 a 3 min)
      python teste_calibracao.py --rapido

Os numeros daqui NAO sao resultado do TCC (sinais sinteticos); provam que o
codigo faz o que deveria e ajudam a escolher os limites.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import teste_v3 as TV
from calibracao import Calibracao, ReconhecedorCalibrado, segmentador_calibrado
from libres_realtime import SignSegmenterV3

FPS = 30

# formas de mao (0 = dedo esticado, 1 = dobrado): polegar, indicador, medio, anelar, minimo
F = {
    "O": np.array([0.55, 0.6, 0.6, 0.6, 0.6]),
    "I": np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    "flor": np.array([0.7, 0.75, 0.75, 0.75, 0.75]),
    "aberta": np.full(5, 0.05),
    "S": np.array([0.6, 1.0, 1.0, 1.0, 1.0]),
    "1": np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
    "2": np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
    "3": np.array([1.0, 0.0, 0.0, 0.0, 1.0]),
    "4": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 1.0, 1.0, 0.0]),
}


def reta(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return lambda u: a + (b - a) * u


def parado(p):
    p = np.array(p, float)
    return lambda u: p


def onda(p, eixo, amp, ciclos):
    p = np.array(p, float)

    def f(u):
        q = p.copy()
        q[eixo] += amp * np.sin(2 * np.pi * ciclos * u)
        return q
    return f


def arco(c, r, a0, a1):
    c = np.array(c, float)
    return lambda u: c + r * np.array([np.cos(a0 + (a1 - a0) * u), np.sin(a0 + (a1 - a0) * u)])


BOM = (0.45, "flor", "aberta", reta((0.05, -0.30), (0.15, -0.15)), None)
# fase: (duracao em s, forma inicial, forma final, trajeto do punho, outra mao (forma, trajeto) | None)
SINAIS = {
    "oi": [(0.35, "O", "O", parado((0.30, -0.05)), None),
           (0.35, "O", "I", reta((0.30, -0.05), (0.38, -0.12)), None)],
    "bom dia": [BOM, (0.6, "1", "1", arco((0.30, -0.05), 0.25, 0.3, -1.6), None)],
    "boa tarde": [BOM, (0.6, "aberta", "aberta", arco((0.25, -0.05), 0.30, -1.6, 0.3), None)],
    "boa noite": [BOM, (0.6, "aberta", "aberta", reta((0.20, -0.20), (0.20, 0.15)),
                        ("aberta", parado((-0.15, 0.25))))],
    "prazer": [(0.9, "aberta", "aberta", lambda u: np.array([-0.05, 0.30]) + 0.10 * np.array(
        [np.cos(2 * np.pi * u), np.sin(2 * np.pi * u)]), None)],
    "obrigado": [(0.7, "aberta", "aberta", reta((0.05, -0.45), (0.25, -0.05)), None)],
    "tchau": [(0.9, "aberta", "aberta", onda((0.50, -0.30), 0, 0.12, 2), None)],
    "por favor": [(0.8, "aberta", "aberta", onda((0.08, 0.20), 1, 0.06, 1.5),
                   ("aberta", onda((-0.08, 0.20), 1, 0.06, 1.5)))],
    "pão": [(0.7, "S", "S", reta((0.05, 0.15), (0.35, 0.15)), ("S", reta((-0.05, 0.15), (-0.35, 0.15))))],
    "café": [(0.7, "Y", "Y", onda((0.05, -0.30), 1, 0.05, 1), None)],
    "leite": [(0.9, "S", "S", onda((0.12, 0.10), 1, 0.10, 2),
               ("S", onda((-0.12, 0.10), 1, -0.10, 2)))],
    "1": [(0.8, "1", "1", parado((0.30, 0.00)), None)],
    "2": [(0.8, "2", "2", parado((0.30, 0.00)), None)],
    "3": [(0.8, "3", "3", parado((0.30, 0.00)), None)],
    "4": [(0.8, "4", "4", parado((0.30, 0.00)), None)],
    "5": [(0.8, "aberta", "aberta", parado((0.30, 0.00)), None)],
    "sim": [(0.8, "S", "S", onda((0.30, -0.05), 1, 0.07, 2), None)],
    "não": [(0.9, "1", "1", onda((0.30, -0.15), 0, 0.09, 3), None)],
}
PARADOS = {"1", "2", "3", "4", "5"}
ARGS_FPS = [FPS]                  # fps do TESTE (a calibracao e sempre a 30)
ARGS_FALHAS = [0.03]
VARIACAO = 1.0                    # escala da variacao de um exemplo para o outro (--variacao)
FORA_Y = 1.7                      # punho abaixo do quadro (em larguras de ombro)


def fluxo(pessoa: TV.Pessoa, palavra: str, rng: np.random.Generator, fps: int = FPS,
          falhas: float = 0.03) -> list:
    """Quadros (maos, lado, pose) de: sem mao -> mao entra -> sinal -> mao sai -> sem mao."""
    fases = SINAIS[palavra]
    v = VARIACAO
    rap = pessoa.rapidez * np.exp(rng.uniform(np.log(0.85), np.log(1.2)) * v)
    amp = 1 + rng.uniform(-0.15, 0.15) * v
    desl = pessoa.desvio.get(palavra, rng.uniform(-0.05, 0.05, 2)) + rng.normal(0, 0.035 * v, 2)
    ruido_f = rng.normal(0, 0.04 * v, 5)
    linha = []                    # (forma, punho, outra|None) por quadro

    def ponto(fun, u, p0):
        return p0 + (fun(u) - p0) * amp + desl

    p_ini = fases[0][3](0.0)
    for fase_k, (dur, fa, fb, traj, outra) in enumerate(fases):
        dur = dur / rap
        if palavra in PARADOS:
            dur = rng.uniform(0.45, 1.0)          # quanto tempo segura o numero
        n = max(2, int(round(dur * fps)))
        for k in range(n):
            u = k / (n - 1)
            m = min(1.0, u / 0.35) if fa != fb else 1.0
            forma = np.clip(F[fa] + (F[fb] - F[fa]) * m + ruido_f, 0, 1)
            o = None
            if outra is not None:
                o = (np.clip(F[outra[0]] + ruido_f, 0, 1), ponto(outra[1], u, p_ini))
            linha.append((forma, ponto(traj, u, p_ini), o))
    # entrada e saida (a mao vem de baixo e volta para baixo)
    def ida(de, para, n):
        return [de + (para - de) * (k / max(1, n - 1)) ** 0.8 for k in range(n)]
    n_ent = int(rng.uniform(0.25, 0.4) * fps)
    n_sai = int(rng.uniform(0.25, 0.4) * fps)
    a0, a1 = linha[0], linha[-1]
    entrada = [(a0[0], p, None if a0[2] is None else (a0[2][0], a0[2][1] + (p - a0[1])))
               for p in ida(a0[1] + np.array([0.1, FORA_Y]), a0[1], n_ent)]
    saida = [(a1[0], p, None if a1[2] is None else (a1[2][0], a1[2][1] + (p - a1[1])))
             for p in ida(a1[1], a1[1] + np.array([0.1, FORA_Y]), n_sai)]
    linha = entrada + linha + saida

    pose = pessoa.pose()
    nada = (np.full((2, 21, 3), np.nan), np.array([-1, -1]), pose)
    quadros = [nada] * int(0.3 * fps)
    lado_dom, lado_out = (0, 1) if pessoa.canhota else (1, 0)
    for forma, punho, outra in linha:
        maos = np.full((2, 21, 3), np.nan)
        lado = np.array([-1, -1])
        itens = [(forma, punho, False, lado_dom)]
        if outra is not None:
            itens.append((outra[0], outra[1], True, lado_out))
        if rng.random() < 0.5:
            itens = itens[::-1]
        for slot, (fm, pw, eh_outra, rot) in enumerate(itens):
            if rng.random() < falhas:
                continue
            m = pessoa.mao_na_imagem(fm, pw, eh_outra, rng)
            if m[0, 1] > 0.98:                    # punho fora da imagem: o MediaPipe nao ve
                continue
            maos[slot] = m
            lado[slot] = rot if rng.random() > 0.05 else 1 - rot
        quadros.append((maos, lado, pose + rng.normal(0, 0.002, pose.shape)))
    quadros += [nada] * int(0.5 * fps)
    return quadros


def segmentador() -> SignSegmenterV3:
    """O mesmo do modo CALIBRADO do app."""
    return segmentador_calibrado()


def sinais_do_fluxo(quadros: list, fps: int = FPS) -> list:
    seg = segmentador()
    saida = []
    for k, (maos, lado, pose) in enumerate(quadros):
        q = None if np.isnan(maos[:, 0, 0]).all() else {"maos": maos, "lado": lado, "pose": pose}
        s = seg.update(q, 1000.0 + k / fps)
        if s is not None:
            saida.append(s)
    return saida


def calibrar(pessoa, palavras, n, rng, pasta: Path, fps=FPS) -> tuple[Calibracao, int]:
    pasta.mkdir(parents=True, exist_ok=True)
    vocab = pasta / "vocab.txt"
    vocab.write_text("\n".join(palavras), encoding="utf-8")
    cal = Calibracao("teste", vocab, pasta, repeticoes=n)
    perdidos = 0
    for p in palavras:
        feitos = 0
        while feitos < n:
            sinais = sinais_do_fluxo(fluxo(pessoa, p, rng, fps), fps)
            if len(sinais) != 1:
                perdidos += 1
                if not sinais:
                    continue
            cal.salvar(p, sinais[0], pessoa.aspecto, fps)
            feitos += 1
    return cal, perdidos


def testar(rec: ReconhecedorCalibrado, pessoa, palavras, m, rng, fps=None, falhas=None) -> dict:
    fps = fps or ARGS_FPS[0]
    falhas = ARGS_FALHAS[0] if falhas is None else falhas
    r = {"certos": 0, "errados": 0, "recusados": 0, "total": 0, "erros": {}, "segmentacao": 0}
    for p in palavras:
        for _ in range(m):
            sinais = sinais_do_fluxo(fluxo(pessoa, p, rng, fps, falhas), fps)
            if len(sinais) != 1:
                r["segmentacao"] += 1
                if not sinais:
                    r["total"] += 1
                    r["recusados"] += 1
                    continue
            res = rec.predict(sinais[0], pessoa.aspecto)
            r["total"] += 1
            if res["aceito"] == p:
                r["certos"] += 1
            elif res["aceito"] is None:
                r["recusados"] += 1
                r["erros"][(p, "(recusado)")] = r["erros"].get((p, "(recusado)"), 0) + 1
            else:
                r["errados"] += 1
                r["erros"][(p, res["aceito"])] = r["erros"].get((p, res["aceito"]), 0) + 1
    return r


def pct(a, b):
    return f"{100 * a / max(b, 1):.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--semente", type=int, default=1)
    ap.add_argument("--pessoas", type=int, default=3, help="pessoas sinteticas testadas")
    ap.add_argument("--fps-teste", type=int, default=FPS, help="fps da camera no uso (calibracao a 30)")
    ap.add_argument("--falhas", type=float, default=0.03, help="fracao de quadros sem a mao no uso")
    ap.add_argument("--variacao", type=float, default=1.0,
                    help="multiplica a variacao entre um exemplo e outro (2 = pessoa bem menos constante)")
    args = ap.parse_args()
    global VARIACAO
    VARIACAO = args.variacao
    ARGS_FPS[0] = args.fps_teste
    ARGS_FALHAS[0] = args.falhas
    m = 6 if args.rapido else 12
    reps = [5] if args.rapido else [3, 5, 8]
    palavras = list(SINAIS)
    t0 = time.time()
    ok_geral = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        resumo = []
        for k in range(args.pessoas):
            rng = np.random.default_rng(args.semente * 100 + k)
            pessoa = TV.Pessoa(40 + k, rng, canhota=(k == 2), quatro_tres=(k == 1))
            pessoa.desvio = {p: rng.uniform(-0.04, 0.04, 2) for p in palavras}
            for n in reps:
                cal, perdidos = calibrar(pessoa, palavras, n, rng, tmp / f"p{k}_n{n}")
                rec = cal.reconhecedor()
                conf = rec.conferencia()
                r = testar(rec, pessoa, palavras, m, rng)
                linha = (f"pessoa {k} ({'canhota' if pessoa.canhota else 'destra'}, "
                         f"{'4:3' if k == 1 else '16:9'}) | {n} exemplos: acerto {r['certos']}/{r['total']} "
                         f"({pct(r['certos'], r['total'])}), errados {r['errados']}, recusados {r['recusados']}"
                         f" | conferencia {conf['certos']}/{conf['total']}")
                print(linha)
                if r["erros"]:
                    print("    erros:", ", ".join(f"{a}->{b} x{v}" for (a, b), v in
                                                sorted(r["erros"].items(), key=lambda x: -x[1])[:8]))
                if conf["parecidos"]:
                    print("    parecidos:", ", ".join(f"{a}~{b} ({x:.2f})" for a, b, x in conf["parecidos"][:5]))
                resumo.append((k, n, r))
            # 2) sinais fora da calibracao
            fora = ["boa noite", "café", "4", "tchau"]
            dentro = [p for p in palavras if p not in fora]
            cal, _ = calibrar(pessoa, dentro, 5, rng, tmp / f"p{k}_fora")
            rec = cal.reconhecedor()
            aceitos, total, como = 0, 0, {}
            for p in fora:
                for _ in range(m):
                    for s in sinais_do_fluxo(fluxo(pessoa, p, rng)):
                        res = rec.predict(s, pessoa.aspecto)
                        total += 1
                        if res["aceito"] is not None:
                            aceitos += 1
                            como[(p, res["aceito"])] = como.get((p, res["aceito"]), 0) + 1
            r_dentro = testar(rec, pessoa, dentro, m // 2, rng)
            print(f"    fora da calibracao ({', '.join(fora)}): recusados {total - aceitos}/{total} "
                  f"({pct(total - aceitos, total)})" + ("; aceitos como: " + ", ".join(
                      f"{a}->{b} x{v}" for (a, b), v in como.items()) if como else "")
                  + f" | as calibradas continuam: {r_dentro['certos']}/{r_dentro['total']}")
        # 3) outra pessoa com a calibracao da pessoa 0
        rng = np.random.default_rng(args.semente * 100 + 99)
        p0 = TV.Pessoa(40, np.random.default_rng(args.semente * 100))
        outra = TV.Pessoa(77, rng)
        outra.desvio = {p: rng.uniform(-0.04, 0.04, 2) for p in palavras}
        cal, _ = calibrar(p0, palavras, 5, np.random.default_rng(5), tmp / "cruzado")
        r = testar(cal.reconhecedor(), outra, palavras, m // 2, rng)
        print(f"outra pessoa com a calibracao da pessoa 0: acerto {r['certos']}/{r['total']} "
              f"({pct(r['certos'], r['total'])}), recusados {r['recusados']}, errados {r['errados']}")

        certos = sum(r["certos"] for _, n, r in resumo if n >= 5)
        total = sum(r["total"] for _, n, r in resumo if n >= 5)
        errados = sum(r["errados"] for _, n, r in resumo if n >= 5)
        ok_geral = certos / max(total, 1) >= 0.95 and errados / max(total, 1) <= 0.02
    print(f"\nRESUMO (5+ exemplos): {certos}/{total} certos ({pct(certos, total)}), {errados} errados "
          f"| {time.time() - t0:.0f} s | {'OK' if ok_geral else 'ABAIXO DO ESPERADO'}")
    return 0 if ok_geral else 1


if __name__ == "__main__":
    sys.exit(main())
