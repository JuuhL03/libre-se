"""
calibracao.py
-------------
CALIBRACAO POR USUARIO: a pessoa faz cada sinal do vocabulario algumas vezes
(5, por padrao) e o app passa a reconhecer esses sinais comparando cada sinal
novo com os exemplos DELA.

Por que existe: o modelo geral (treino_v3.py) so conhece as 20 palavras do
MINDS e, com uma pessoa que ele nunca viu, acerta ~88%. Palavras do dia a dia
(oi, bom dia, obrigado, numeros...) nao estao em nenhuma base publica com
varias pessoas. Com exemplos da propria pessoa, o reconhecimento passa a ser
"dependente do usuario": mesmo jeito de sinalizar, mesma variante regional,
mesma camera.

Como compara: cada sinal vira a mesma sequencia normalizada da v3 (30 quadros:
forma das duas maos, posicao em relacao aos ombros e ao nariz; libres_v3.py)
e e comparado com todos os exemplos por DTW (Dynamic Time Warping, que alinha
as duas sequencias no tempo: o mesmo sinal feito mais rapido ou mais devagar
no meio continua parecido). Vence a palavra do exemplo mais proximo. Se o
sinal estiver longe de TODOS os exemplos, a resposta e "nao reconhecido": o
limite de cada palavra sai da propria calibracao (o quanto os exemplos dela
variam entre si).

Arquivos (so pontos, nenhuma imagem):
  vocabulario.txt                          uma palavra por linha (editavel)
  calibracao/<usuario>/<palavra>/*.npz     maos, lado, pose, aspecto, fps
"""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np

from libres_v3 import (C_DEDO_D, C_FORMA_D, C_FORMA_O, C_LOC_D, C_LOC_O, C_PRES_O, C_REL_O,
                       preparar)

PROJECT_DIR = Path(__file__).parent
PASTA_CALIBRACAO = PROJECT_DIR / "calibracao"
VOCAB_ARQ = PROJECT_DIR / "vocabulario.txt"
REPETICOES = 5
N_QUADROS = 30

VOCAB_PADRAO = ["oi", "bom dia", "boa tarde", "boa noite", "prazer", "obrigado", "tchau",
                "por favor", "pão", "café", "leite", "quanto custa",
                "1", "2", "3", "4", "5", "sim", "não"]

# peso de cada parte do quadro na distancia (forma em "tamanhos de mao",
# posicoes em "larguras de ombro"); cada grupo pesa igual, qualquer que seja
# o numero de colunas dele
GRUPOS = [  # (colunas, peso)
    (C_FORMA_D, 1.0),       # forma da mao dominante
    (C_LOC_D, 1.0),         # punho dominante x ombros
    (C_DEDO_D, 0.7),        # ponta do indicador x nariz
    (slice(C_PRES_O, C_PRES_O + 1), 0.7),   # a outra mao aparece?
    (C_FORMA_O, 0.7),       # forma da outra mao
    (C_LOC_O, 0.7),         # punho da outra mao x ombros
    (C_REL_O, 0.5),         # outra mao x mao dominante
]
PESO_MOV = 0.5              # velocidade do punho dominante (parado x em movimento)
SUAVIZAR = 0                # passadas de media movel no tempo antes de comparar
BANDA = 6                   # DTW: no maximo 6 quadros (20%) de desalinhamento
PONTAS = 3                  # DTW: pode pular ate 3 quadros no comeco e no fim (a mao entrando
                            # e saindo do quadro, que a camera pega em pontos diferentes)
FATOR_MAX = 1.35            # limite da palavra = 1,35 x a maior distancia entre os exemplos dela,
FATOR_MEDIANA = 2.2         # ... mas no maximo 2,2 x a distancia mediana deles (um exemplo
                            #     estranho nao abre a porta para qualquer sinal)
FATOR_MED = 1.6             # ... e nunca menos que 1,6 x a distancia tipica de todas
DESTOANTE = 2.5             # exemplo 2,5 x mais longe dos outros que o normal: avisa para refazer
LIMITE_MIN = 0.10           # piso absoluto do limite (exemplos quase identicos)
LIMITE_1 = 0.45             # palavra com um exemplo so (sem como medir a variacao)
INCLINACAO = 6.0            # so para o painel: distancia/limite -> "confianca" (50% no limite)


# ------------------------------------------------------------------ vocabulario
def slug(palavra: str) -> str:
    """'Bom dia' -> 'bom_dia'; 'pão' -> 'pao' (nome de pasta seguro no Windows)."""
    s = unicodedata.normalize("NFKD", str(palavra)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "sinal"


def ler_vocabulario(arq: Path = VOCAB_ARQ) -> list[str]:
    """Uma palavra (ou expressao) por linha; # comeca comentario. Se o arquivo
    nao existe, cria com o vocabulario da cena da padaria."""
    arq = Path(arq)
    if not arq.exists():
        arq.write_text(
            "# Vocabulario da calibracao: uma palavra ou expressao por linha.\n"
            "# Pode mudar a vontade; depois, no app (modo CALIBRADO), calibre as novas.\n"
            + "\n".join(VOCAB_PADRAO) + "\n", encoding="utf-8")
    palavras, vistos = [], set()
    for linha in arq.read_text(encoding="utf-8-sig").splitlines():
        p = linha.split("#", 1)[0].strip()
        if p and slug(p) not in vistos:
            vistos.add(slug(p))
            palavras.append(p)
    return palavras


# ------------------------------------------------------------------ como se faz o sinal
# A calibracao grava como a PROPRIA pessoa faz cada sinal. Quem ainda nao sabe um
# sinal aprende antes: no video de um dicionario de Libras (tecla H / "Como se faz?")
# ou no exemplo do perfil de REFERENCIA (alguem que conhece os sinais calibrou com
# esse nome; a tecla V mostra o exemplo dele quando a pessoa ainda nao gravou o seu).
PERFIL_REFERENCIA = "referencia"
DICIONARIO_BUSCA = "https://spreadthesign.com/pt.br/search/?q="   # Spread the Sign, Libras (pt.br)
DICIONARIO_INES = "https://dicionario.ines.gov.br/"                # Dicionario da Libras do INES
DICIONARIO_ALFABETO = "https://spreadthesign.com/pt.br/alphabet/"  # alfabeto manual no Spread the Sign
NUMEROS_EXTENSO = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"]


def termo_dicionario(palavra: str) -> str:
    """O que procurar no dicionario: 'BOM_DIA' -> 'bom dia'; '2' -> 'dois'."""
    t = " ".join(str(palavra).replace("_", " ").split()).lower()
    return NUMEROS_EXTENSO[int(t)] if t.isdecimal() and int(t) < len(NUMEROS_EXTENSO) else t


def link_dicionario(palavra: str) -> str:
    """Busca da palavra no Spread the Sign (videos de sinais em Libras)."""
    return DICIONARIO_BUSCA + quote(termo_dicionario(palavra))


def segmentador_calibrado():
    """O segmentador do modo CALIBRADO (gravar exemplos e reconhecer usam o
    MESMO): o sinal comeca quando uma mao aparece e termina quando as maos saem
    do quadro (abaixar a mao). Se a mao ficar parada no ar por 2,5 s, o sinal
    termina tambem, guardando 0,6 s da mao parada: num numero, a mao parada E o
    sinal (no modelo geral esse trecho e cortado)."""
    from libres_realtime import SignSegmenterV3
    return SignSegmenterV3(still_time=2.5, min_motion=0.0, cauda_parada=0.6, max_duration=6.0)


# ------------------------------------------------------------------ distancia
def suavizar(seq: np.ndarray, passadas: int) -> np.ndarray:
    """Media movel [1/4, 1/2, 1/4] no tempo, `passadas` vezes (pontas repetidas)."""
    for _ in range(passadas):
        ext = np.vstack([seq[:1], seq, seq[-1:]])
        seq = 0.25 * ext[:-2] + 0.5 * ext[1:-1] + 0.25 * ext[2:]
    return seq


def vetor_dtw(seq: np.ndarray) -> np.ndarray:
    """(T, 135) da v3 -> (T, D) em que a distancia euclidiana entre dois quadros
    ja e a soma ponderada das diferencas de cada grupo."""
    seq = suavizar(np.asarray(seq, dtype=np.float64), SUAVIZAR)
    partes = []
    for cols, peso in GRUPOS:
        bloco = seq[:, cols]
        partes.append(bloco * (peso / np.sqrt(bloco.shape[1])))
    loc = seq[:, C_LOC_D]
    vel = np.vstack([np.zeros((1, 2)), np.diff(loc, axis=0)]) * (len(seq) / 3.0)
    partes.append(vel * (PESO_MOV / np.sqrt(2)))
    return np.hstack(partes)


def dtw_lote(a: np.ndarray, B: np.ndarray, banda: int = BANDA, pontas: int = PONTAS) -> np.ndarray:
    """DTW (passos simetricos, banda de Sakoe-Chiba) de a (T, D) contra cada
    B[k] (T, D), todos de uma vez, podendo comecar e terminar ate `pontas`
    quadros "para dentro" de qualquer uma das duas. Distancia media por passo
    do caminho: (K,)."""
    K, T, _ = B.shape
    n2a = (a ** 2).sum(1)
    n2b = (B ** 2).sum(2)
    C = n2a[None, :, None] + n2b[:, None, :] - 2.0 * np.einsum("id,ktd->kit", a, B)
    C = np.sqrt(np.maximum(C, 0.0))
    D = np.full((K, T + 1, T + 1), np.inf)
    L = np.zeros((K, T + 1, T + 1))            # peso acumulado do caminho (normalizacao)
    D[:, 0, 0] = 0.0
    for k in range(1, pontas + 1):             # comeco livre: pular quadros iniciais
        D[:, k, 0] = 0.0
        D[:, 0, k] = 0.0
    for i in range(1, T + 1):
        j0, j1 = max(1, i - banda), min(T, i + banda)
        for j in range(j0, j1 + 1):
            c = C[:, i - 1, j - 1]
            d1 = D[:, i - 1, j - 1] + 2 * c
            d2 = D[:, i - 1, j] + c
            d3 = D[:, i, j - 1] + c
            m = d1 <= d2
            melhor = np.where(m, d1, d2)
            peso = np.where(m, L[:, i - 1, j - 1] + 2, L[:, i - 1, j] + 1)
            m = d3 < melhor
            D[:, i, j] = np.where(m, d3, melhor)
            L[:, i, j] = np.where(m, L[:, i, j - 1] + 1, peso)
    fins = [(T, T)] + [(T - k, T) for k in range(1, pontas + 1)] + [(T, T - k) for k in range(1, pontas + 1)]
    melhor = np.full(K, np.inf)
    for i, j in fins:                          # fim livre: parar antes do ultimo quadro
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(L[:, i, j] > 0, D[:, i, j] / np.maximum(L[:, i, j], 1e-9), np.inf)
        melhor = np.minimum(melhor, v)
    return melhor


def sequencia(sinal: dict, aspecto: float) -> Optional[np.ndarray]:
    """Sinal do segmentador (maos, lado, pose) -> sequencia normalizada (30, 135)."""
    return preparar(sinal["maos"], sinal["lado"], sinal["pose"], aspecto, N_QUADROS,
                    simetrico_pela_posicao=True)


# ------------------------------------------------------------------ reconhecedor
class ReconhecedorCalibrado:
    """Vizinho mais proximo por DTW sobre os exemplos da calibracao, com um
    limite por palavra para responder "nao reconhecido"."""

    def __init__(self, exemplos: dict[str, list[np.ndarray]]) -> None:
        self.classes = [p for p, seqs in exemplos.items() if seqs]
        self.rotulos = np.array([p for p in self.classes for _ in exemplos[p]])
        self.vetores = np.stack([vetor_dtw(s) for p in self.classes for s in exemplos[p]])
        K = len(self.rotulos)
        self.dist = np.zeros((K, K))
        for k in range(K):
            self.dist[k] = dtw_lote(self.vetores[k], self.vetores)
        self.dist = (self.dist + self.dist.T) / 2
        np.fill_diagonal(self.dist, np.inf)
        self.limite = self._limites()
        self.fps_calibracao: Optional[float] = None

    def adicionar(self, palavra: str, seqs: list) -> "ReconhecedorCalibrado":
        """Inclui novos exemplos sem recalcular tudo (so as distancias novas):
        durante a calibracao, a cada palavra pronta."""
        if not len(seqs):
            return self
        novos = np.stack([vetor_dtw(s) for s in seqs])
        todos = np.concatenate([self.vetores, novos])
        K0, K = len(self.vetores), len(todos)
        dist = np.full((K, K), np.inf)
        dist[:K0, :K0] = self.dist
        for k in range(K0, K):
            d = dtw_lote(todos[k], todos)
            dist[k, :] = d
            dist[:, k] = np.minimum(dist[:, k], d) if k else d
        dist[K0:, K0:] = (dist[K0:, K0:] + dist[K0:, K0:].T) / 2
        np.fill_diagonal(dist, np.inf)
        self.vetores, self.dist = todos, dist
        self.rotulos = np.concatenate([self.rotulos, [palavra] * len(seqs)])
        if palavra not in self.classes:
            self.classes.append(palavra)
        self.limite = self._limites()
        return self

    # distancia de cada exemplo ao exemplo mais proximo da MESMA palavra
    def _intra(self) -> dict[str, np.ndarray]:
        out = {}
        for p in self.classes:
            idx = np.where(self.rotulos == p)[0]
            if len(idx) >= 2:
                out[p] = self.dist[np.ix_(idx, idx)].min(axis=1)
        return out

    def _limites(self) -> dict[str, float]:
        intra = self._intra()
        todos = np.concatenate(list(intra.values())) if intra else np.array([])
        tipico = float(np.median(todos)) if len(todos) else LIMITE_1 / FATOR_MED
        self.tipico = tipico
        lim = {}
        for p in self.classes:
            if p in intra:
                v = intra[p]
                proprio = min(FATOR_MAX * float(v.max()), FATOR_MEDIANA * float(np.median(v)))
                lim[p] = max(proprio, FATOR_MED * tipico, LIMITE_MIN)
            else:
                lim[p] = max(FATOR_MED * tipico, LIMITE_1)
        return lim

    def destoantes(self) -> dict[str, int]:
        """Palavras com exemplo muito diferente dos outros da mesma palavra
        (gesto errado, mao saiu no meio...): {palavra: quantos}."""
        intra = self._intra()
        todos = np.concatenate(list(intra.values())) if intra else np.array([])
        tipico = float(np.median(todos)) if len(todos) else 0.0
        out = {}
        for p, v in intra.items():
            if len(v) >= 3:
                n = int(((v > DESTOANTE * np.median(v)) & (v > 2 * tipico)).sum())
                if n:
                    out[p] = n
        return out

    def _por_classe(self, d: np.ndarray) -> dict[str, float]:
        return {p: float(d[self.rotulos == p].min()) for p in self.classes
                if np.isfinite(d[self.rotulos == p]).any()}

    def _decidir(self, dc: dict[str, float], top_k: int = 3) -> dict:
        ordem = sorted(dc, key=dc.get)        # vizinho mais proximo (distancia crua)
        s = np.array([dc[p] / self.limite[p] for p in ordem])
        # "confianca" para o painel: 50% exatamente no limite da palavra, ~95% na
        # metade dele, ~0% no dobro (nao e probabilidade; nao soma 100%)
        conf = 1.0 / (1.0 + np.exp(INCLINACAO * (s - 1.0)))
        melhor = ordem[0] if ordem else None
        aceito = melhor if melhor is not None and dc[melhor] <= self.limite[melhor] else None
        return {"top": [(p, float(c)) for p, c in zip(ordem[:top_k], conf[:top_k])],
                "aceito": aceito, "melhor": melhor,
                "dist": dc.get(melhor, np.inf) if melhor else np.inf,
                "razao": float(s[0]) if len(s) else np.inf}

    def sequencia(self, sinal: dict, aspecto: float) -> Optional[np.ndarray]:
        return sequencia(sinal, aspecto)

    def predict(self, sinal: dict, aspecto: float, top_k: int = 3) -> dict:
        """sinal do SignSegmenterV3 -> {"top": [(palavra, conf)], "aceito": palavra|None, ...}"""
        seq = self.sequencia(sinal, aspecto)
        if seq is None:
            return {"top": [], "aceito": None, "melhor": None, "dist": np.inf, "razao": np.inf}
        return self.predict_seq(seq, top_k)

    def predict_seq(self, seq: np.ndarray, top_k: int = 3) -> dict:
        d = dtw_lote(vetor_dtw(seq), self.vetores)
        return self._decidir(self._por_classe(d), top_k)

    def conferencia(self) -> dict:
        """Cada exemplo contra os OUTROS (como se fosse um sinal novo).
        -> {"total", "certos", "errados", "recusados", "por_palavra": {p: [certos, n]},
            "confusoes": [(p, q, vezes)], "parecidos": [(p, q, razao)]}"""
        certos = errados = recusados = 0
        por = {p: [0, 0] for p in self.classes}
        conf: dict[tuple[str, str], int] = {}
        for k, p in enumerate(self.rotulos):
            if (self.rotulos == p).sum() < 2:
                continue
            r = self._decidir(self._por_classe(self.dist[k]))
            por[p][1] += 1
            if r["aceito"] == p:
                certos += 1
                por[p][0] += 1
            elif r["aceito"] is None:
                recusados += 1
                if r["melhor"] != p:
                    conf[(p, r["melhor"])] = conf.get((p, r["melhor"]), 0) + 1
            else:
                errados += 1
                conf[(p, r["aceito"])] = conf.get((p, r["aceito"]), 0) + 1
        # pares de palavras perto demais: distancia entre elas / limite
        parecidos = []
        for i, p in enumerate(self.classes):
            for q in self.classes[i + 1:]:
                d = float(self.dist[np.ix_(self.rotulos == p, self.rotulos == q)].min())
                r = d / min(self.limite[p], self.limite[q])
                if r < 1.0:
                    parecidos.append((p, q, r))
        parecidos.sort(key=lambda x: x[2])
        return {"total": certos + errados + recusados, "certos": certos, "errados": errados,
                "recusados": recusados, "por_palavra": por, "destoantes": self.destoantes(),
                "confusoes": sorted(((p, q, n) for (p, q), n in conf.items()), key=lambda x: -x[2]),
                "parecidos": parecidos}


# ------------------------------------------------------------------ exemplos em disco
class Calibracao:
    """Os exemplos de UM usuario, salvos em calibracao/<usuario>/<palavra>/."""

    def __init__(self, usuario: str = "padrao", vocab_arq: Path = VOCAB_ARQ,
                 pasta_base: Path = PASTA_CALIBRACAO, repeticoes: int = REPETICOES) -> None:
        self.usuario = usuario
        self.pasta = Path(pasta_base) / slug(usuario)
        self.vocab_arq = Path(vocab_arq)
        self.repeticoes = repeticoes
        self.palavras = ler_vocabulario(self.vocab_arq)

    def pasta_palavra(self, palavra: str) -> Path:
        return self.pasta / slug(palavra)

    def arquivos(self, palavra: str) -> list[Path]:
        return sorted(self.pasta_palavra(palavra).glob("*.npz"))

    def n(self, palavra: str) -> int:
        return len(self.arquivos(palavra))

    def completa(self) -> bool:
        return all(self.n(p) >= self.repeticoes for p in self.palavras)

    def proxima_incompleta(self, a_partir: int = 0) -> Optional[int]:
        n = len(self.palavras)
        for k in range(n):
            i = (a_partir + k) % n
            if self.n(self.palavras[i]) < self.repeticoes:
                return i
        return None

    def salvar(self, palavra: str, sinal: dict, aspecto: float, fps: float = 30.0) -> Path:
        pasta = self.pasta_palavra(palavra)
        pasta.mkdir(parents=True, exist_ok=True)
        base = pasta / f"{slug(palavra)}_{time.strftime('%Y%m%d_%H%M%S')}"
        nome, k = Path(f"{base}.npz"), 2
        while nome.exists():
            nome, k = Path(f"{base}_{k}.npz"), k + 1
        np.savez_compressed(nome, maos=np.asarray(sinal["maos"], np.float32),
                            lado=np.asarray(sinal["lado"], np.int8),
                            pose=np.asarray(sinal["pose"], np.float32),
                            aspecto=np.float32(aspecto), fps=np.float32(fps), palavra=str(palavra))
        return nome

    def apagar_ultimo(self, palavra: str) -> bool:
        arqs = self.arquivos(palavra)
        if not arqs:
            return False
        max(arqs, key=lambda a: a.stat().st_mtime).unlink()
        return True

    def apagar_todos(self, palavra: str) -> int:
        arqs = self.arquivos(palavra)
        for a in arqs:
            a.unlink()
        return len(arqs)

    def sequencias(self, palavra: str) -> list[np.ndarray]:
        seqs = []
        for a in self.arquivos(palavra):
            try:
                with np.load(a) as d:
                    s = preparar(d["maos"], d["lado"], d["pose"], float(d["aspecto"]), N_QUADROS,
                                 simetrico_pela_posicao=True)
            except Exception:  # noqa: BLE001  (arquivo corrompido: ignora)
                s = None
            if s is not None:
                seqs.append(s)
        return seqs

    def fps(self) -> Optional[float]:
        """fps tipico da camera durante a calibracao (mediana dos exemplos)."""
        v = []
        for p in self.palavras:
            for a in self.arquivos(p):
                try:
                    with np.load(a) as d:
                        v.append(float(d["fps"]))
                except Exception:  # noqa: BLE001
                    pass
        return float(np.median(v)) if v else None

    def exemplo_tipico(self, palavra: str) -> Optional[dict]:
        """O exemplo mais "do meio" da palavra (o mais parecido com os outros
        exemplos dela), com os quadros crus (maos, lado, pose, aspecto, fps),
        para a tecla V mostrar como a pessoa fez o sinal na calibracao.
        None se a palavra ainda nao tem exemplo."""
        lista = exemplos_por_tipicidade(self.arquivos(palavra))
        if not lista:
            return None
        k, _, bruto = lista[0]
        ex = dict(bruto)
        ex.update(palavra=palavra, indice=k, n=len(lista))
        return ex

    def reconhecedor(self) -> Optional[ReconhecedorCalibrado]:
        """Com as palavras que tem pelo menos 1 exemplo (None se nenhuma)."""
        ex = {p: self.sequencias(p) for p in self.palavras}
        ex = {p: s for p, s in ex.items() if s}
        if not ex:
            return None
        rec = ReconhecedorCalibrado(ex)
        rec.fps_calibracao = self.fps()
        return rec


def exemplos_por_tipicidade(arquivos: list[Path]) -> list[tuple[int, Path, dict]]:
    """Os exemplos (indice na lista, arquivo, quadros crus: maos, lado, pose,
    aspecto, fps) do mais tipico ao menos tipico: o mais tipico e o mais parecido
    com os outros (soma das distancias DTW do reconhecimento calibrado). Arquivos
    ilegiveis ou sem mao ficam de fora. Com menos de 3 exemplos, a ordem dos arquivos."""
    brutos, seqs = [], []
    for a in arquivos:
        try:
            with np.load(a) as d:
                bruto = {"maos": d["maos"].astype(np.float64), "lado": d["lado"],
                         "pose": d["pose"].astype(np.float64), "aspecto": float(d["aspecto"]),
                         "fps": float(d["fps"]) if "fps" in d.files else 30.0}
        except Exception:  # noqa: BLE001  (arquivo corrompido: ignora)
            continue
        s = preparar(bruto["maos"], bruto["lado"], bruto["pose"], bruto["aspecto"], N_QUADROS,
                     simetrico_pela_posicao=True)
        if s is not None:
            brutos.append((len(brutos), Path(a), bruto))
            seqs.append(s)
    if len(seqs) < 3:
        return brutos
    V = np.stack([vetor_dtw(s) for s in seqs])
    soma = [float(dtw_lote(V[i], np.delete(V, i, axis=0)).sum()) for i in range(len(V))]
    return [brutos[i] for i in np.argsort(soma, kind="stable")]


def exemplo_referencia(palavra: str, vocab_arq: Path, pasta_base: Path = PASTA_CALIBRACAO,
                       usuario: str = "", repeticoes: int = REPETICOES,
                       perfil: str = PERFIL_REFERENCIA) -> Optional[dict]:
    """O exemplo tipico da palavra no perfil de referencia, para quem ainda nao
    gravou a palavra ver como se faz (com "referencia": True). None se a referencia
    nao tem a palavra ou se o usuario e a propria referencia. Nao cria nada em disco
    (vocab_arq deve ser um vocabulario que ja existe, o do proprio usuario)."""
    if slug(usuario) == slug(perfil) or not (Path(pasta_base) / slug(perfil) / slug(palavra)).is_dir():
        return None
    ex = Calibracao(perfil, vocab_arq, pasta_base=pasta_base, repeticoes=repeticoes).exemplo_tipico(palavra)
    if ex is not None:
        ex["referencia"] = True
    return ex


def avaliar_exemplo(rec: Optional[ReconhecedorCalibrado], palavra: str, seq: np.ndarray,
                    mesmos: list = ()) -> str:
    """Logo depois de gravar um exemplo: ele ficou parecido com OUTRA palavra
    ja calibrada, ou muito diferente dos outros exemplos da mesma palavra?
    mesmos: exemplos desta palavra gravados agora (ainda fora do `rec`).
    "" = tudo certo."""
    if seq is None:
        return ""
    v = vetor_dtw(seq)
    d_mesma = None
    if len(mesmos):
        d_mesma = float(dtw_lote(v, np.stack([vetor_dtw(s) for s in mesmos])).min())
    tipico = None
    if rec is not None:
        dc = rec._por_classe(dtw_lote(v, rec.vetores))
        if palavra in dc:
            d_mesma = dc[palavra] if d_mesma is None else min(d_mesma, dc[palavra])
        outras = {p: d for p, d in dc.items() if p != palavra}
        if outras:
            q = min(outras, key=outras.get)
            if outras[q] <= rec.limite[q] and (d_mesma is None or outras[q] < d_mesma):
                return f"ficou parecido com '{q}'"
        tipico = rec.tipico
    if d_mesma is not None and tipico and d_mesma > 3.0 * tipico:
        return f"ficou diferente dos outros '{palavra}'"
    return ""
