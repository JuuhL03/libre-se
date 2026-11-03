#!/usr/bin/env python3
"""
Extracao v3 (duas maos + pose) dos videos da LIBRAS-EQT-UECE (UECE, 2026).

A base tem sinais ISOLADOS (cada pasta e uma glosa, com uma subpasta por
informante) e frases (parte continua). So os sinais isolados entram aqui.

1. Baixe "Libras-EQT-UECE (RGB).zip" em https://zenodo.org/records/21398895
   (licenca CC BY-NC 4.0: pesquisa e ensino, sem uso comercial; nao
   redistribua os videos nem os pontos extraidos - nao suba para o GitHub)
2. Coloque o .zip em eqt_videos/ (nao precisa extrair; a pasta extraida tambem serve)
3. Confira o que foi encontrado:   python extrair_eqt_v3.py --listar
4. Extraia:                         python extrair_eqt_v3.py
   (pode parar e rodar de novo: o que ja foi feito e pulado)
5. Avalie, uma pessoa fora do treino (5 informantes):   python avaliar_v3.py --dados eqt

Saida: eqt_v3/<sinal>/eqt_InformanteNN_<video>.npz, no mesmo formato do minds_v3/,
entao treino_v3.py e avaliar_v3.py leem as duas bases juntas ou separadas (--dados).

Como o script separa sinal isolado de frase, pelo caminho de cada video:
  - pasta com "contin" no nome -> frase; com "isol" -> sinal isolado;
  - senao, pelo nome da pasta: UMA palavra ("Energia.", "e-mail") ou glosa em
    MAIUSCULAS ("ANTERIOR") -> sinal isolado; o resto ("Casa toda.") -> frase.
O --listar mostra a decisao para cada pasta; corrija com --incluir / --excluir.

Opcoes:
  --pasta CAMINHO       pasta com o .zip ou os videos (padrao: eqt_videos)
  --listar              so mostra os sinais, informantes e videos encontrados
  --incluir A,B         forca estas glosas como sinal isolado
  --excluir A,B         deixa estas glosas de fora
  --processos N         videos em paralelo (padrao: metade dos nucleos)
  --teste N             processa so N videos e mostra o que saiu (para conferir)
"""

import argparse
import hashlib
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np

import extrair_minds_v3 as E
from extrair_minds_videos import VIDEO_EXT, eh_lixo_mac

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "eqt_v3"
EXTENSOES = tuple(sorted(set(VIDEO_EXT) | {".m4v"}))
RE_INFORMANTE = re.compile(r"inform[a-z]*[\s_-]*0*(\d+)", re.I)
RE_NUMERACAO = re.compile(r"^\s*\d+\s*[-_. ]\s*")      # "01_", "3 - ", "12."


def sem_acento(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def glosa_de(nome_pasta: str) -> str:
    """'01_ANTERIOR' -> 'ANTERIOR'; 'A__luz_caiu.' fica como esta."""
    return RE_NUMERACAO.sub("", nome_pasta).strip(" _-")


def slug(glosa: str) -> str:
    """Nome da pasta de saida (e da classe do modelo): 'BOM_DIA' -> 'bom_dia'."""
    s = re.sub(r"[^a-z0-9]+", "_", sem_acento(glosa).lower()).strip("_")
    return s or "sinal"


def parece_frase(glosa: str, caminho: str) -> bool:
    """Na base, os sinais isolados sao pastas com UMA palavra ("Energia.", "Luz.",
    "e-mail") ou com a glosa em MAIUSCULAS ("ANTERIOR", "SOMENTE ALGUNS COMODOS");
    o resto sao frases ("Eu quero pagar a conta.", "Casa toda.")."""
    baixo = sem_acento(caminho).lower()
    if "contin" in baixo:
        return True
    if "isol" in baixo:
        return False
    texto = re.sub(r"[\s.!?\u2026]+$", "", glosa.replace("_", " ").strip())   # "Energia." -> "Energia"
    if not texto or re.search(r"[.?!,;:\u2026]", texto):                      # pontuacao no meio
        return True
    letras = [c for c in texto if c.isalpha()]
    if letras and all(c.isupper() for c in letras):                          # glosa em maiusculas
        return False
    return len(texto.split()) >= 2


def identificar(caminho: str) -> tuple[str | None, int | None]:
    """Caminho do video -> (glosa, informante). A glosa e a pasta logo acima da
    pasta do informante ("01_ANTERIOR/INFORMANTE 4/IMG_7004.mp4"); se o
    informante so aparece no nome do arquivo, e a pasta do arquivo, ou o comeco
    do nome ("ANTERIOR_INFORMANTE 4_IMG_7004.mp4")."""
    partes = PurePosixPath(caminho.replace("\\", "/")).parts
    for i, parte in enumerate(partes):
        m = RE_INFORMANTE.search(parte)
        if not m:
            continue
        informante = int(m.group(1))
        if i < len(partes) - 1:                       # pasta do informante
            pai = partes[i - 1] if i >= 1 else ""
        else:                                         # informante so no nome do arquivo
            prefixo = parte[:m.start()].strip(" _-")
            pai = prefixo if prefixo else (partes[i - 1] if i >= 1 else "")
        glosa = glosa_de(pai)
        if not glosa or "libras-eqt" in glosa.lower():
            return None, informante
        return glosa, informante
    return None, None


def nome_saida(glosa: str, informante: int, caminho_logico: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", PurePosixPath(caminho_logico.replace("\\", "/")).stem)[:40]
    marca = hashlib.md5(caminho_logico.encode("utf-8")).hexdigest()[:6]   # nomes repetidos (video_1.mp4)
    return OUTPUT_DIR / slug(glosa) / f"eqt_Informante{informante:02d}_{stem}_{marca}.npz"


def listar(pasta: Path, incluir: set, excluir: set):
    """-> (tarefas, por_glosa, nao_identificados). tarefa = (origem, membro, palavra, saida)."""
    videos = []                                        # (origem, membro, caminho_logico)
    for arq in sorted(pasta.rglob("*")):
        if arq.suffix.lower() == ".zip":
            with zipfile.ZipFile(arq) as z:
                for membro in z.namelist():
                    if membro.lower().endswith(EXTENSOES) and not eh_lixo_mac(membro):
                        videos.append((arq, membro, membro))
        elif arq.suffix.lower() in EXTENSOES and not eh_lixo_mac(str(arq)):
            videos.append((arq, None, str(arq.relative_to(pasta))))
    # chave (glosa, frase?): a mesma glosa pode existir na parte isolada e na continua
    por_glosa = defaultdict(lambda: {"videos": 0, "informantes": set(), "usar": False})
    tarefas, sem_id = [], []
    for origem, membro, logico in videos:
        glosa, inf = identificar(logico)
        if glosa is None or inf is None:
            sem_id.append(logico)
            continue
        frase = parece_frase(glosa, logico)
        usar = (slug(glosa) in incluir) or (not frase and slug(glosa) not in excluir)
        g = por_glosa[(glosa, frase)]
        g["videos"] += 1
        g["informantes"].add(inf)
        g["usar"] = usar
        if usar:
            tarefas.append((str(origem), membro, slug(glosa), str(nome_saida(glosa, inf, logico))))
    return tarefas, por_glosa, sem_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", default=str(PROJECT_DIR / "eqt_videos"))
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--incluir", default="", help="glosas forcadas como sinal isolado (virgula)")
    ap.add_argument("--excluir", default="", help="glosas que ficam de fora (virgula)")
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--teste", type=int, default=0)
    args = ap.parse_args()

    pasta = Path(args.pasta)
    if not pasta.exists():
        sys.exit(f"Pasta {pasta} nao encontrada. Crie e coloque nela o 'Libras-EQT-UECE (RGB).zip'.")
    incluir = {slug(x) for x in args.incluir.split(",") if x.strip()}
    excluir = {slug(x) for x in args.excluir.split(",") if x.strip()}

    print("=" * 60)
    print("EXTRACAO v3 (duas maos + pose) DA LIBRAS-EQT-UECE")
    print("=" * 60)
    tarefas, por_glosa, sem_id = listar(pasta, incluir, excluir)
    isoladas = {g: v for g, v in por_glosa.items() if v["usar"]}
    frases = {g: v for g, v in por_glosa.items() if not v["usar"]}
    infs = sorted({i for v in isoladas.values() for i in v["informantes"]})
    print(f"Sinais isolados: {len(isoladas)} | videos: {len(tarefas)} | informantes: "
          f"{', '.join(map(str, infs)) or '-'}")
    print(f"Fora (frases ou excluidos): {len(frases)} pastas, {sum(v['videos'] for v in frases.values())} videos")
    if sem_id:
        print(f"Videos sem informante ou glosa no caminho (ignorados): {len(sem_id)}")
        for s in sem_id[:5]:
            print(f"   ex.: {s}")
    if args.listar:
        print("\nSINAIS ISOLADOS (entram):")
        # nomes sem acento: no Windows, "> arquivo.txt" estraga os acentos
        for (g, _), v in sorted(isoladas.items()):
            nome = sem_acento(g).replace("\n", " ")
            print(f"   {nome:28s} -> {slug(g):24s} {v['videos']:3d} videos, informantes {sorted(v['informantes'])}")
        print("\nFORA (frases da parte continua ou --excluir):")
        for (g, _), v in sorted(frases.items()):
            print(f"   {sem_acento(g).replace(chr(10), ' ')[:60]:60s} {v['videos']:3d} videos")
        print("\nSe algum sinal isolado ficou na lista de fora: --incluir NOME; o contrario: --excluir NOME")
        return

    E.garantir_modelo_pose()
    pendentes = [t for t in tarefas if not Path(t[3]).exists()]
    print(f"Ja feitos: {len(tarefas) - len(pendentes)} | a fazer: {len(pendentes)}")
    if args.teste:
        pendentes = pendentes[:args.teste]
        args.processos = 1
    if not pendentes:
        print("Nada a fazer.")
        return

    print(f"Processando com {args.processos} processo(s)...\n")
    t0 = time.time()
    ok = falhas = 0
    with ProcessPoolExecutor(max_workers=args.processos, initializer=E._init_worker,
                             initargs=(str(E.HAND_MODEL), str(E.POSE_MODEL))) as ex:
        futs = [ex.submit(E.processar, t) for t in pendentes]
        for i, fut in enumerate(as_completed(futs), 1):
            status, saida, n = fut.result()
            if status == "ok":
                ok += 1
                if args.teste:
                    with np.load(saida) as d:
                        duas = int((~np.isnan(d["maos"][:, :, 0, 0])).all(axis=1).sum())
                        com_pose = int((~np.isnan(d["pose"][:, 0, 0])).sum())
                        print(f"   {Path(saida).parent.name}/{Path(saida).name}: {len(d['maos'])} quadros, "
                              f"{n} com mao, {duas} com as duas, {com_pose} com pose, "
                              f"imagem {float(d['aspecto']):.2f} (largura/altura), {float(d['fps']):.0f} fps")
                    if com_pose < 0.5 * len(d["maos"]):
                        print("      ATENCAO: pose em menos da metade dos quadros. Se o video estiver "
                              "deitado (celular de pe), me avise.")
            else:
                falhas += 1
                print(f"   [{status}] {Path(saida).name} ({n} quadros com mao)")
            if i % 20 == 0 or i == len(pendentes):
                gasto = time.time() - t0
                print(f"  {i}/{len(pendentes)} videos | {gasto / 60:.1f} min | "
                      f"faltam ~{gasto / i * (len(pendentes) - i) / 60:.0f} min", flush=True)

    print(f"\nPronto: {ok} salvos, {falhas} descartados/erros.")
    print("Proximo passo:  python avaliar_v3.py --dados eqt   (e, para o app, python treino_v3.py)")


if __name__ == "__main__":
    main()
