"""
Motor da versao web do Libre-se.

E a mesma logica do libras_camera.py (modos ALFABETO, PALAVRAS e CALIBRADO,
com calibracao, conferencia e exemplo da tecla V), sem camera e sem janela:
o navegador roda o MediaPipe, manda os pontos das maos e do corpo de cada
quadro e recebe o estado para desenhar o painel. Os modelos e o codigo de
reconhecimento sao os do app de computador (libres_realtime.py, libres_v3.py,
calibracao.py), entao os numeros da avaliacao valem para a versao web.

Nenhuma imagem chega aqui: so os pontos (RNF06 do TCC).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from calibracao import PERFIL_REFERENCIA as REFERENCIA_PADRAO  # noqa: E402
from calibracao import (Calibracao, avaliar_exemplo, exemplo_referencia,  # noqa: E402
                        segmentador_calibrado, sequencia, slug)
from libres_realtime import (LetterRecognizer, SignSegmenter, SignSegmenterV3,  # noqa: E402
                             Soletrador, WordRecognizer, WordRecognizerV3, accept_word)
from libres_v3 import rastrear  # noqa: E402

MODOS = ["ALFABETO", "PALAVRAS", "CALIBRADO"]
# perfil de REFERENCIA: alguem que conhece os sinais calibra com este nome; quem ainda nao gravou
# um sinal ve, na tecla V, como essa pessoa fez (para aprender antes de gravar o seu)
PERFIL_REFERENCIA = os.environ.get("LIBRES_REFERENCIA", REFERENCIA_PADRAO)
PONTOS_CORPO = [0, 11, 12, 13, 14, 15, 16, 23, 24]   # nariz, ombros, cotovelos, punhos, quadris
MAX_PALAVRAS_VOCAB = 60
MAX_TAM_PALAVRA = 40


def _num(x: float, casas: int = 4) -> Optional[float]:
    """float JSON (NaN/inf -> None)."""
    x = float(x)
    return round(x, casas) if np.isfinite(x) else None


def _lista(a: np.ndarray, casas: int = 4) -> list:
    """Array -> listas aninhadas para JSON, com NaN -> None."""
    a = np.asarray(a, dtype=np.float64)
    r = np.round(a, casas).astype(object)
    r[~np.isfinite(a)] = None
    return r.tolist()


class Recursos:
    """Modelos carregados uma vez e usados por todas as sessoes (so leitura)."""

    def __init__(self, raiz: Path = RAIZ, pasta_dados: Optional[Path] = None) -> None:
        self.raiz = Path(raiz)
        self.pasta_dados = Path(pasta_dados or os.environ.get("LIBRES_DADOS") or self.raiz / "dados_web")
        self.pasta_dados.mkdir(parents=True, exist_ok=True)
        self.avisos: list[str] = []
        self.letters = self.words = self.words_v3 = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                self.letters = LetterRecognizer(self.raiz / "modelo_alfabeto.pkl")
            except Exception as e:  # noqa: BLE001
                self.avisos.append(f"alfabeto indisponivel: {e}")
            v3 = self.raiz / "modelo_palavras_v3.pkl"
            if v3.exists():
                try:
                    self.words_v3 = self.words = WordRecognizerV3(v3)
                except Exception as e:  # noqa: BLE001
                    self.avisos.append(f"palavras v3 indisponiveis: {e}")
            if self.words is None:
                try:
                    self.words = WordRecognizer(self.raiz / "modelo_rotulado_xgb.pkl")
                except Exception as e:  # noqa: BLE001
                    self.avisos.append(f"palavras indisponiveis: {e}")
        # modelo salvo com outra versao do scikit-learn/xgboost: pode dar resultado errado
        versoes = sorted({str(x.message).split(" when using")[0].replace("Trying to unpickle estimator ", "")
                          for x in w if "version" in str(x.message).lower()})
        if versoes:
            self.avisos.append("modelos salvos com outras versoes das bibliotecas (rode servidor/fixar_versoes.py "
                               "e publique de novo): " + "; ".join(versoes[:4]))
            print("[aviso]", self.avisos[-1], flush=True)
        self.vocab_padrao = self.raiz / "vocabulario.txt"

    def modos(self) -> list[str]:
        disp = []
        for m in MODOS:
            if m == "ALFABETO" and self.letters is None:
                continue
            if m == "PALAVRAS" and self.words is None:
                continue
            disp.append(m)
        return disp

    def deteccao(self) -> dict:
        """O que o navegador detecta em cada modo: "1mao" (uma mao, como o
        landmarker do app no alfabeto) ou "v3" (duas maos + corpo, como o
        DetectorV3 da extracao dos videos de treino)."""
        return {"ALFABETO": "1mao", "PALAVRAS": "v3" if self.words_v3 is not None else "1mao",
                "CALIBRADO": "v3"}

    def info(self) -> dict:
        return {"modos": self.modos(), "deteccao": self.deteccao(),
                "palavras": [str(c) for c in self.words.classes] if self.words is not None else [],
                "palavras_v3": self.words_v3 is not None,
                "opcoes_modelo": {"recortar": bool(getattr(self.words_v3, "recortar", False)),
                                  "sem_mao_parada": bool(getattr(self.words_v3, "sem_mao_parada", False))},
                "avisos": self.avisos}


def preparar_exemplo(ex: dict) -> dict:
    """O exemplo da tecla V pronto para o navegador desenhar: maos sempre no
    mesmo lugar, qual e a dominante e o retangulo com corpo e maos (unidades
    de imagem, x * aspecto). Mesmo calculo do libras_camera.preparar_exemplo."""
    asp = float(ex["aspecto"])
    maos = rastrear(np.asarray(ex["maos"], np.float64), np.asarray(ex["lado"]))
    caminho = []
    for k in range(2):
        w = maos[~np.isnan(maos[:, k, 0, 0]), k, 0, :2] * [asp, 1.0]
        caminho.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if len(w) > 1 else 0.0)
    pose = np.asarray(ex["pose"], np.float64)
    m = maos.reshape(-1, 3)
    m = m[np.isfinite(m[:, 0])]
    pts = [np.c_[m[:, 0] * asp, m[:, 1]]]
    corpo = None
    if pose.ndim == 3:
        sel = pose[:, PONTOS_CORPO]
        ok = np.isfinite(sel[..., 0]) & (np.nan_to_num(sel[..., 3]) > 0.5) & (sel[..., 1] < 1.05)
        pts.append(np.c_[sel[ok][:, 0] * asp, sel[ok][:, 1]])
        cab = ok[:, 0] & ok[:, 1] & ok[:, 2]           # nariz e ombros: inclui o topo da cabeca desenhada
        if cab.any():                                  # (circulo no nariz, raio 0,28 x largura dos ombros)
            larg = np.hypot((sel[cab, 1, 0] - sel[cab, 2, 0]) * asp, sel[cab, 1, 1] - sel[cab, 2, 1])
            pts.append(np.c_[sel[cab, 0, 0] * asp, sel[cab, 0, 1] - 0.28 * larg])
        corpo = np.where(ok[..., None], sel[..., [0, 1]], np.nan)          # (T, 9, 2)
    P = np.vstack(pts) if len(np.vstack(pts)) else np.array([[0.0, 0.0], [asp, 1.0]])
    (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    mx, my = 0.08 * max(x1 - x0, 0.2), 0.08 * max(y1 - y0, 0.2)
    fps_ex = float(ex.get("fps", 30.0))
    return {"tipo": "exemplo", "palavra": str(ex["palavra"]), "indice": int(ex["indice"]), "n": int(ex["n"]),
            "referencia": bool(ex.get("referencia")),
            "aspecto": asp, "fps": fps_ex if 8.0 <= fps_ex <= 60.0 else 30.0,
            "dom": int(np.argmax(caminho)),
            "caixa": [_num(x0 - mx), _num(y0 - my), _num(x1 + mx), _num(y1 + my)],
            "maos": _lista(maos[..., :2]),                                   # (T, 2, 21, 2)
            "corpo": _lista(corpo) if corpo is not None else None,           # (T, 9, 2): PONTOS_CORPO
            "pontos_corpo": PONTOS_CORPO}


class Sessao:
    """Um usuario conectado: o estado do libras_camera.py inteiro, quadro a quadro."""

    def __init__(self, recursos: Recursos, usuario: str = "padrao", repeticoes: int = 5,
                 modo: Optional[str] = None) -> None:
        self.r = recursos
        self.usuario = (usuario or "padrao").strip()[:40] or "padrao"
        self.modos = recursos.modos()
        if not self.modos:
            raise RuntimeError("nenhum modelo carregado no servidor")
        self.segmenter = SignSegmenterV3() if recursos.words_v3 is not None else SignSegmenter()
        self.soletrador = Soletrador(recursos.letters) if recursos.letters is not None else None
        self.texto: list[str] = []
        self.letra_atual: list[str] = []
        self.ultimo_top: list = []
        self.ultimo_ok: Optional[str] = None
        self.ultimo_melhor: Optional[str] = None
        self.info_alfa: Optional[dict] = None
        self.fps = 0.0
        self.t_ant: Optional[float] = None
        self.t = 0.0                           # relogio do NAVEGADOR (ultimo quadro)
        self.aspecto = 16 / 9
        self.historico: list[tuple[float, str]] = []
        self.entrou, self.entrou_ate = "", -1e9
        self.aviso, self.aviso_ate = "", -1e9
        self.saidas: list[dict] = []          # mensagens extras para o navegador (exemplo da tecla V)
        self.n_quadros = 0
        # ------------------------------------------------ calibracao (por usuario)
        base = recursos.pasta_dados / "calibracao"
        pasta = base / slug(self.usuario)
        pasta.mkdir(parents=True, exist_ok=True)
        self.vocab_arq = pasta / "vocabulario.txt"
        if not self.vocab_arq.exists() and recursos.vocab_padrao.exists():
            shutil.copyfile(recursos.vocab_padrao, self.vocab_arq)    # comeca com o do projeto
        self.cal = Calibracao(self.usuario, self.vocab_arq, pasta_base=base, repeticoes=max(2, repeticoes))
        self._contagem: dict[str, int] = {}
        self._ref: dict[str, bool] = {}       # a referencia tem exemplo da palavra? (mesmo cache de 2 s)
        self._t_contagem = 0.0
        self.rec_cal = None
        self.seg_cal = segmentador_calibrado()
        self.calibrando = False
        self.cal_idx = 0
        self.cal_msg, self.cal_msg_ate, self.cal_msg_ruim = "", -1e9, False
        self.cal_resumo: list[str] = []
        self.cal_detalhes: list[str] = []
        self.cal_sessao: dict = {}
        self.ultimo_salvo = None
        self.ver_palavra: Optional[str] = None
        self.exemplo_palavra: Optional[str] = None
        self.exemplo_ref = False              # o ultimo exemplo mostrado era o da referencia
        self.exemplo_ate = -1e9
        self.modo = modo if modo in self.modos else self.modos[0]
        if self.modo == "CALIBRADO":
            self.entrar_calibrado()

    # ------------------------------------------------------------ texto
    def texto_completo(self) -> str:
        partes = self.texto + (["".join(self.letra_atual)] if self.letra_atual else [])
        return " ".join(partes)

    def fechar_soletracao(self) -> None:
        if self.letra_atual:
            self.texto.append("".join(self.letra_atual))
            self.letra_atual.clear()

    def registrar(self, item: str, t: float) -> None:
        self.historico.append((t, item))
        del self.historico[:-8]
        self.entrou, self.entrou_ate = item, t + 1.5

    def avisar(self, txt: str, dur: float = 4.0) -> None:
        self.aviso, self.aviso_ate = txt, self.t + dur

    # ------------------------------------------------------------ calibracao
    def n(self, palavra: str) -> int:
        """Exemplos gravados da palavra. Contagem guardada por 2 s (o painel pergunta
        varias vezes por segundo); outra aba do mesmo perfil pode ter gravado mais."""
        agora = time.monotonic()
        if agora - self._t_contagem > 2.0:
            self._contagem.clear()
            self._ref.clear()
            self._t_contagem = agora
        if palavra not in self._contagem:
            self._contagem[palavra] = self.cal.n(palavra)
        return self._contagem[palavra]

    def _mudou(self, palavra: Optional[str] = None) -> None:
        if palavra is None:
            self._contagem.clear()
        else:
            self._contagem.pop(palavra, None)

    def proxima_incompleta(self, a_partir: int = 0) -> Optional[int]:
        n = len(self.cal.palavras)
        for k in range(n):
            i = (a_partir + k) % n
            if self.n(self.cal.palavras[i]) < self.cal.repeticoes:
                return i
        return None

    def msg_cal(self, txt: str, ruim: bool = False, dur: float = 4.0) -> None:
        self.cal_msg, self.cal_msg_ate, self.cal_msg_ruim = txt, self.t + dur, ruim

    def iniciar_calibracao(self, idx: int) -> None:
        self.calibrando, self.cal_idx = True, idx
        self.seg_cal.reset()
        self.cal_sessao.clear()

    def mostrar_conferencia(self) -> None:
        """Confere a calibracao (cada exemplo contra os outros) e diz o que refazer."""
        self.cal_detalhes = []
        if self.rec_cal is None:
            self.cal_resumo = ["nenhum sinal calibrado: toque em Calibrar (K)"]
            return
        c = self.rec_cal.conferencia()
        self.cal_resumo = []
        if c["total"]:
            self.cal_resumo.append(f"conferencia: {c['certos']}/{c['total']} exemplos certos")
        for p, (ok, n) in c["por_palavra"].items():
            if n and ok < n:
                self.cal_detalhes.append(f"{p}: {ok}/{n} certos")
        for p, q, n in c["confusoes"][:5]:
            self.cal_detalhes.append(f"'{p}' saiu como '{q}' {n}x")
        for p, q, _r in c["parecidos"][:5]:
            self.cal_detalhes.append(f"parecidos: '{p}' e '{q}': refaca um deles (R), mais diferente")
            if len(self.cal_resumo) < 3:
                self.cal_resumo.append(f"parecidos: {p} ~ {q} (R refaz)")
        for p, n in c["destoantes"].items():
            self.cal_detalhes.append(f"'{p}': {n} exemplo(s) bem diferente(s) dos outros: R refaz a palavra")
            if len(self.cal_resumo) < 3:
                self.cal_resumo.append(f"{p}: exemplo estranho (R refaz)")
        faltam = [p for p in self.cal.palavras if self.n(p) == 0]
        if faltam:
            self.cal_detalhes.append("sem calibrar: " + ", ".join(faltam))

    def terminar_calibracao(self) -> None:
        self.calibrando = False
        self.seg_cal.reset()
        self.rec_cal = self.cal.reconhecedor()
        self.mostrar_conferencia()

    def entrar_calibrado(self) -> None:
        """Ao entrar no modo: carrega os exemplos; se falta calibrar alguma palavra, calibra."""
        self.rec_cal = self.cal.reconhecedor()
        falta = self.proxima_incompleta(0)
        if falta is not None:
            self.iniciar_calibracao(falta)
        else:
            self.calibrando = False
            self.mostrar_conferencia()

    def tratar_exemplo(self, sinal: dict) -> None:
        """Um sinal feito durante a calibracao: vira exemplo da palavra da vez."""
        palavra = self.cal.palavras[self.cal_idx]
        seq = sequencia(sinal, self.aspecto)
        if seq is None:
            self.msg_cal("sinal curto demais: repita", ruim=True)
            return
        problema = avaliar_exemplo(self.rec_cal, palavra, seq, self.cal_sessao.get(palavra, []))
        arq = self.cal.salvar(palavra, sinal, self.aspecto, self.fps if self.fps > 0 else 30.0)
        self._mudou(palavra)
        self.ultimo_salvo = (palavra, arq)
        self.cal_sessao.setdefault(palavra, []).append(seq)
        n = self.n(palavra)
        rep = self.cal.repeticoes
        if problema:
            self.msg_cal(f"{palavra} {n}/{rep}: {problema}. Desfazer (D) apaga", ruim=True, dur=5.0)
        else:
            self.msg_cal(f"{palavra} {n}/{rep} ok")
        if n >= rep:
            novos = self.cal_sessao.pop(palavra, [])
            if self.rec_cal is None or palavra in self.rec_cal.classes:
                self.rec_cal = self.cal.reconhecedor()
            else:
                self.rec_cal = self.rec_cal.adicionar(palavra, novos)
            prox = self.proxima_incompleta(self.cal_idx + 1)
            if prox is None:
                self.terminar_calibracao()
                self.msg_cal("calibracao pronta!", dur=5.0)
            else:
                self.cal_idx = prox
                if problema:
                    self.msg_cal(f"{palavra}: {problema} (D desfaz). Agora: {self.cal.palavras[self.cal_idx]}",
                                 ruim=True, dur=6.0)
                else:
                    self.msg_cal(f"{palavra} pronto! Agora: {self.cal.palavras[self.cal_idx]}")

    # ------------------------------------------------------------ quadros
    @staticmethod
    def _converter(q: dict) -> Optional[tuple]:
        """Um quadro do navegador -> (t, maos (2,21,3), lado (2,), pose (33,4), aspecto)."""
        try:
            t = float(q["t"])
            asp = float(q.get("a") or 16 / 9)
            if not (np.isfinite(t) and 0.2 < asp < 5.0):
                return None
            maos = np.full((2, 21, 3), np.nan)
            lado = np.full(2, -1)
            m = q.get("m") or [None, None]
            l_ = q.get("l") or [-1, -1]
            for k in range(min(2, len(m))):
                if m[k] is not None:
                    v = np.asarray(m[k], dtype=np.float64)
                    if v.size == 63 and np.isfinite(v).all():
                        maos[k] = v.reshape(21, 3)
                        lado[k] = int(l_[k]) if k < len(l_) and l_[k] in (0, 1) else -1
            pose = np.full((33, 4), np.nan)
            p = q.get("p")
            if p is not None:
                v = np.asarray([np.nan if x is None else x for x in p], dtype=np.float64)
                if v.size == 132:
                    pose = v.reshape(33, 4)
            return t, maos, lado, pose, asp
        except (KeyError, TypeError, ValueError):
            return None

    def quadros(self, lista: list) -> int:
        """Processa os quadros recebidos (em ordem). Devolve quantos valeram."""
        ok = 0
        for q in lista or []:
            c = self._converter(q)
            if c is None:
                continue
            t, maos, lado, pose, asp = c
            if self.t_ant is not None and t <= self.t_ant:
                continue                    # quadro repetido ou fora de ordem
            self.quadro(t, maos, lado, pose, asp)
            ok += 1
        return ok

    def quadro(self, t: float, maos: np.ndarray, lado: np.ndarray, pose: np.ndarray, aspecto: float) -> None:
        """Um quadro: o corpo do laco do libras_camera.py (sem desenhar)."""
        if self.t_ant is not None:
            dt = t - self.t_ant
            if 0 < dt < 1.0:
                self.fps = 1.0 / dt if self.fps <= 0 else 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.t_ant = self.t = t
        self.aspecto = aspecto
        self.n_quadros += 1
        presentes = [k for k in range(2) if not np.isnan(maos[k, 0, 0])]
        lm = maos[presentes[0]].reshape(-1).astype(np.float64) if presentes else None
        v3 = (maos, lado, pose) if presentes else None
        self.info_alfa = None

        if self.modo == "ALFABETO":
            if self.soletrador is not None:
                self.soletrador.aspecto = aspecto
                info = self.soletrador.update(lm, t)
                self.info_alfa = info
                self._aplicar_letra(info, t)

        elif self.modo == "CALIBRADO":
            quadro = None if v3 is None else {"maos": v3[0], "lado": v3[1], "pose": v3[2]}
            sinal = self.seg_cal.update(quadro, t)
            if sinal is not None and self.calibrando:
                self.tratar_exemplo(sinal)
            elif sinal is not None and self.rec_cal is not None:
                res = self.rec_cal.predict(sinal, aspecto)
                self.ultimo_top, self.ultimo_ok, self.ultimo_melhor = res["top"], res["aceito"], res["melhor"]
                self.ver_palavra = self.ultimo_ok or self.ultimo_melhor
                if self.ultimo_ok:
                    self.fechar_soletracao()
                    self.texto.append(self.ultimo_ok.upper())
                    self.registrar(self.ultimo_ok.upper(), t)

        else:   # PALAVRAS
            if self.r.words_v3 is not None:
                quadro = None if v3 is None else {"maos": v3[0], "lado": v3[1], "pose": v3[2]}
                sinal = self.segmenter.update(quadro, t)
                if sinal is not None:
                    self.ultimo_top = self.r.words_v3.predict(sinal, aspecto)
            else:
                sinal = self.segmenter.update(lm, t)
                if sinal is not None:
                    self.ultimo_top = self.r.words.predict(sinal)
            if sinal is not None:
                self.ultimo_ok = accept_word(self.ultimo_top)
                if self.ultimo_ok:
                    self.fechar_soletracao()
                    item = self.ultimo_ok.replace("_", " ").upper()
                    self.texto.append(item)
                    self.registrar(item, t)

    def _aplicar_letra(self, info: dict, t: float) -> None:
        acao, nova = info.get("acao"), info.get("letra")
        if acao == "troca" and self.letra_atual:
            antiga, self.letra_atual[-1] = self.letra_atual[-1], nova
            self.registrar(f"{antiga} -> {nova}", t)
        elif acao in ("add", "troca"):
            self.letra_atual.append(nova)
            self.registrar(nova, t)

    def gravando(self) -> bool:
        if self.modo == "CALIBRADO":
            return bool(self.seg_cal.recording)
        if self.modo == "PALAVRAS":
            return bool(self.segmenter.recording)
        return False

    # ------------------------------------------------------------ comandos (as teclas do app)
    def comando(self, cmd: str, dados: Optional[dict] = None) -> None:
        dados = dados or {}
        t = self.t
        if cmd in ("modo", "sair") and self.modo == "ALFABETO" and self.soletrador is not None:
            # um gesto (J, Z...) que ainda nao terminou entra antes de trocar de modo
            fim = self.soletrador.finalizar(t)
            self._aplicar_letra(fim, t)
        if cmd == "modo":
            alvo = dados.get("alvo")
            if alvo not in self.modos:
                i = self.modos.index(self.modo)
                alvo = self.modos[(i + 1) % len(self.modos)]
            if alvo == self.modo:
                if len(self.modos) == 1:
                    self.avisar("nenhum outro modo disponivel")
                return
            self.modo = alvo
            self.fechar_soletracao()
            self.segmenter.reset()
            self.seg_cal.reset()
            self.calibrando = False
            self.exemplo_palavra = None
            self.ultimo_top, self.ultimo_ok, self.ultimo_melhor = [], None, None
            if self.soletrador is not None:
                self.soletrador.reset()
            if self.modo == "CALIBRADO":
                self.entrar_calibrado()
        elif cmd == "limpar":
            self.texto.clear()
            self.letra_atual.clear()
            self.ultimo_top, self.ultimo_ok, self.ultimo_melhor = [], None, None
        elif cmd == "apagar":
            if self.letra_atual:
                self.letra_atual.pop()
            elif self.texto:
                self.texto.pop()
        elif cmd == "vocabulario":
            if self.calibrando:
                self.msg_cal("termine a calibracao (K) antes de mudar o vocabulario", ruim=True)
            else:
                self._trocar_vocabulario(dados.get("palavras") or [])
        elif self.modo != "CALIBRADO":
            return
        elif cmd == "k":
            if self.calibrando:
                self.terminar_calibracao()
                self.msg_cal("calibracao encerrada")
            else:
                falta = self.proxima_incompleta(0)
                self.iniciar_calibracao(0 if falta is None else falta)
        elif cmd == "v":
            self._exemplo()
        elif cmd in ("n", "p") and self.calibrando:
            self.cal_idx = (self.cal_idx + (1 if cmd == "n" else -1)) % len(self.cal.palavras)
            self.seg_cal.reset()
        elif cmd == "r" and self.calibrando:
            palavra = self.cal.palavras[self.cal_idx]
            self.cal.apagar_todos(palavra)
            self._mudou(palavra)
            self.cal_sessao.pop(palavra, None)
            self.rec_cal = self.cal.reconhecedor()
            self.seg_cal.reset()
            self.msg_cal(f"exemplos de '{palavra}' apagados: faca de novo")
        elif cmd == "d" and self.calibrando:
            if self.ultimo_salvo is not None and self.ultimo_salvo[1].exists():
                palavra, arq = self.ultimo_salvo
                arq.unlink()
                self._mudou(palavra)
                if self.cal_sessao.get(palavra):
                    self.cal_sessao[palavra].pop()
                if palavra in self.cal.palavras:
                    self.cal_idx = self.cal.palavras.index(palavra)
                self.rec_cal = self.cal.reconhecedor()
                self.ultimo_salvo = None
                self.msg_cal(f"ultimo exemplo de '{palavra}' desfeito")
            else:
                self.msg_cal("nada para desfazer", ruim=True)

    def _exemplo(self) -> None:
        """Tecla V: o exemplo tipico (como a pessoa fez o sinal na calibracao). Se a pessoa
        ainda nao gravou a palavra, o da referencia (alguem que conhece os sinais); calibrando,
        V de novo enquanto o exemplo passa alterna entre o seu e o da referencia."""
        prontas = [p for p in self.cal.palavras if self.n(p) > 0]
        repetiu = self.t < self.exemplo_ate
        if self.calibrando:
            alvo = self.cal.palavras[self.cal_idx]
        elif self.exemplo_palavra in prontas and repetiu:
            alvo = prontas[(prontas.index(self.exemplo_palavra) + 1) % len(prontas)]   # V de novo: proxima
        elif self.ver_palavra in prontas:
            alvo = self.ver_palavra
        else:
            alvo = prontas[0] if prontas else None
        ex = None
        if alvo is not None:
            quer_ref = self.calibrando and repetiu and self.exemplo_palavra == alvo and not self.exemplo_ref
            ex = None if quer_ref else self.cal.exemplo_tipico(alvo)
            if ex is None:
                ex = self.exemplo_referencia(alvo)
            if ex is None and quer_ref:
                ex = self.cal.exemplo_tipico(alvo)
        if ex is None:
            txt = (f"'{alvo}' ainda nao tem exemplo gravado: veja o video no dicionario (Como se faz? / H)"
                   if alvo else "nenhum sinal calibrado ainda")
            if self.calibrando:
                self.msg_cal(txt, ruim=True, dur=6.0)
            else:
                self.avisar(txt)
            return
        pronto = preparar_exemplo(ex)
        dur = len(ex["maos"]) / pronto["fps"]
        self.exemplo_palavra, self.exemplo_ref = alvo, pronto["referencia"]
        self.exemplo_ate = self.t + 2 * (dur + 0.6)          # 2 passadas + pausa (o navegador desenha)
        self.saidas.append(pronto)

    def _pasta_referencia(self, palavra: str) -> Optional[Path]:
        if slug(self.usuario) == slug(PERFIL_REFERENCIA):
            return None
        return self.cal.pasta.parent / slug(PERFIL_REFERENCIA) / slug(palavra)

    def tem_referencia(self, palavra: str) -> bool:
        self.n(palavra)                                       # renova o cache a cada 2 s
        if palavra not in self._ref:
            pasta = self._pasta_referencia(palavra)
            self._ref[palavra] = pasta is not None and any(pasta.glob("*.npz"))
        return self._ref[palavra]

    def exemplo_referencia(self, palavra: str) -> Optional[dict]:
        """O exemplo do perfil de referencia (se existe e tem essa palavra)."""
        return exemplo_referencia(palavra, self.vocab_arq, self.cal.pasta.parent, self.usuario,
                                  self.cal.repeticoes, PERFIL_REFERENCIA)

    def _trocar_vocabulario(self, palavras: list) -> None:
        limpas, vistos = [], set()
        for p in palavras:
            p = str(p).replace("#", " ").strip()[:MAX_TAM_PALAVRA]
            if p and slug(p) not in vistos:
                vistos.add(slug(p))
                limpas.append(p)
        if not limpas:
            self.avisar("vocabulario vazio: nada mudou")
            return
        limpas = limpas[:MAX_PALAVRAS_VOCAB]
        self.vocab_arq.write_text(
            "# Vocabulario da calibracao (versao web): uma palavra ou expressao por linha.\n"
            + "\n".join(limpas) + "\n", encoding="utf-8")
        rep = self.cal.repeticoes
        self.cal = Calibracao(self.usuario, self.vocab_arq, pasta_base=self.cal.pasta.parent, repeticoes=rep)
        self._mudou()
        self.cal_idx = 0
        self.cal_sessao.clear()
        self.ultimo_salvo = None
        if self.modo == "CALIBRADO":
            self.entrar_calibrado()
        self.avisar(f"vocabulario salvo: {len(limpas)} palavras")

    def pegar_saidas(self) -> list[dict]:
        s, self.saidas = self.saidas, []
        return s

    # ------------------------------------------------------------ estado para o painel
    def estado(self) -> dict:
        t = self.t
        e = {"modo": self.modo, "modos": self.modos, "texto": self.texto_completo(),
             "fps": round(self.fps, 1), "gravando": self.gravando(),
             "entrou": self.entrou if t < self.entrou_ate else None,
             "historico": [[round(ti, 3), item] for ti, item in self.historico],
             "aviso": self.aviso if t < self.aviso_ate else None,
             "ultimo": {"top": [[str(p), _num(c, 3)] for p, c in self.ultimo_top],
                        "ok": self.ultimo_ok, "melhor": self.ultimo_melhor}}
        if self.modo == "ALFABETO":
            a = self.info_alfa
            e["alfa"] = None if not a or not a.get("letra") else {
                "letra": a["letra"], "conf": _num(a.get("conf", 0.0), 3), "aceita": bool(a.get("aceita")),
                "movendo": bool(a.get("movendo")),
                "top": [[str(l), _num(c, 3)] for l, c in (a.get("top") or [])[:3]]}
        if self.modo == "CALIBRADO":
            palavras = self.cal.palavras
            rep = self.cal.repeticoes
            c = {"calibrando": self.calibrando, "rep": rep, "total": len(palavras),
                 "tem_reconhecedor": self.rec_cal is not None,
                 "palavras": [[p, self.n(p)] for p in palavras],
                 "resumo": self.cal_resumo[:3], "detalhes": self.cal_detalhes[:8],
                 "msg": self.cal_msg if t < self.cal_msg_ate and self.cal_msg else None,
                 "msg_ruim": self.cal_msg_ruim,
                 "fps_cal": _num(getattr(self.rec_cal, "fps_calibracao", None) or np.nan, 1)}
            if self.calibrando and palavras:
                p = palavras[self.cal_idx]
                c.update(idx=self.cal_idx, palavra=p, n=self.n(p), ref=self.tem_referencia(p))
            e["cal"] = c
        return e
