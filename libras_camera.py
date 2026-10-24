#!/usr/bin/env python3
"""
Libre-se - camera unica para ALFABETO e PALAVRAS.

Uso: python libras_camera.py

A janela tem duas partes: a camera (so com os pontos da mao) e, ao lado, um
PAINEL ESCURO com o texto reconhecido, a letra/palavra do momento, as 3
opcoes mais provaveis e o historico. O painel nao depende da luz da sala.
Tudo o que entra no texto tambem e gravado em saida_libras.txt.

Teclas:
  M          = alterna modo ALFABETO <-> PALAVRAS
  ESPACO     = limpa o texto montado
  BACKSPACE  = apaga o ultimo item
  S          = salva os ultimos 6 s de pontos da mao em debug/ (so numeros,
               sem imagem), para ajustar o reconhecimento com maos reais
  Q          = sair

Modo ALFABETO: letra parada entra depois de ~0,3 s estavel. H, J, K, X e Z
tem movimento: faca o movimento e, no fim, pare a mao ou abaixe; a letra
entra quando o movimento termina. Para repetir a mesma letra (OSSO), tire a
mao do quadro entre as duas.

Modo PALAVRAS: comece com a mao fora do quadro (ou abaixada), faca o sinal
e ABAIXE a mao. O sinal tambem termina se a mao ficar parada no ar por ~1 s
depois de se mexer. O painel mostra as 3 palavras mais provaveis do ultimo
sinal e a lista de sinais que o modelo conhece.
"""

import sys
import time
import unicodedata
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from libres_features import FEAT_DIM
from libres_realtime import (LeitorCamera, LetterRecognizer, SignSegmenter,
                             Soletrador, WordRecognizer, abrir_camera, accept_word)

PROJECT_DIR = Path(__file__).parent
WORD_MODEL = PROJECT_DIR / "modelo_rotulado_xgb.pkl"
LETTER_MODEL = PROJECT_DIR / "modelo_alfabeto.pkl"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"


# ---------------------------------------------------------------- modelos
letters = words = None
try:
    letters = LetterRecognizer(LETTER_MODEL)
    print(f"Alfabeto: {LETTER_MODEL.name} carregado")
except Exception as e:  # noqa: BLE001
    print(f"[aviso] alfabeto indisponivel: {e}")
try:
    words = WordRecognizer(WORD_MODEL)
    print(f"Palavras: {len(words.classes)} classes -> {', '.join(words.classes)}")
except Exception as e:  # noqa: BLE001
    print(f"[aviso] palavras indisponiveis: {e}")
if letters is None and words is None:
    sys.exit("Nenhum modelo carregado.")

mode = "ALFABETO" if letters is not None else "PALAVRAS"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=HAND_MODEL.read_bytes()),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))

# ---------------------------------------------------------------- camera
cap = abrir_camera()
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
         "(Teams, Discord, navegador, OBS) ou rode com LIBRES_CAMERA=1.")
leitor = LeitorCamera(cap)

segmenter = SignSegmenter()
texto: list[str] = []          # itens ja reconhecidos (letras e palavras)
letra_atual: list[str] = []    # letras sendo soletradas
soletrador = Soletrador(letters) if letters is not None else None
ultimo_top: list[tuple[str, float]] = []
ultimo_ok: str | None = None
t_prev = time.time()
fps = 0.0


def texto_completo() -> str:
    partes = texto + (["".join(letra_atual)] if letra_atual else [])
    return " ".join(partes)


def fechar_soletracao() -> None:
    if letra_atual:
        texto.append("".join(letra_atual))
        letra_atual.clear()


# ---------------------------------------------------------------- painel
ALTURA_TELA = 720              # a camera e redimensionada para esta altura
PAINEL_L = 480
FONTE = cv2.FONT_HERSHEY_SIMPLEX
COR_FUNDO = (32, 28, 26)
COR_LINHA = (70, 70, 70)
COR_TEXTO = (240, 240, 240)
COR_SUAVE = (150, 150, 150)
COR_OK = (90, 220, 90)
COR_ALFA = (230, 160, 40)      # azul (BGR)
COR_PAL = (40, 150, 240)       # laranja (BGR)
SAIDA = PROJECT_DIR / "saida_libras.txt"
DEBUG_DIR = PROJECT_DIR / "debug"
TRECHO_S = 6.0                 # segundos guardados para a tecla S

historico: list[str] = []
trecho: deque = deque()        # (t, landmarks|None) dos ultimos TRECHO_S segundos
entrou, entrou_ate = "", 0.0   # destaque do que acabou de entrar no texto
aviso, aviso_ate = "", 0.0     # mensagem no rodape do painel


def asc(txt: str) -> str:
    """A fonte do OpenCV nao tem acentos: 'maçã' -> 'maca'."""
    return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()


def registrar(item: str, tipo: str, t: float) -> None:
    """Guarda no historico do painel e no arquivo saida_libras.txt."""
    global entrou, entrou_ate
    hora = time.strftime("%H:%M:%S")
    historico.append(f"{hora}  {item}")
    del historico[:-8]
    entrou, entrou_ate = item, t + 1.5
    try:
        with open(SAIDA, "a", encoding="utf-8") as f:
            f.write(f"{hora}\t{tipo}\t{item}\n")
    except OSError:
        pass


def salvar_trecho() -> Path | None:
    """Tecla S: grava os ultimos segundos de pontos da mao (sem imagem)."""
    if not trecho:
        return None
    DEBUG_DIR.mkdir(exist_ok=True)
    ts = np.array([t for t, _ in trecho], dtype=np.float64)
    lms = np.full((len(trecho), FEAT_DIM), np.nan, dtype=np.float32)
    for i, (_, lm) in enumerate(trecho):
        if lm is not None and len(lm) == FEAT_DIM:
            lms[i] = lm
    base = DEBUG_DIR / f"trecho_{time.strftime('%Y%m%d_%H%M%S')}"
    nome, n = base.with_suffix(".npz"), 2
    while nome.exists():
        nome, n = Path(f"{base}_{n}.npz"), n + 1
    np.savez_compressed(nome, t=ts, lm=lms, modo=mode, texto=texto_completo())
    return nome


def quebrar(txt: str, largura: int, escala: float, esp: int) -> list[str]:
    """Quebra o texto em linhas que cabem na largura (corta palavras longas)."""
    def cabe(x: str) -> bool:
        return cv2.getTextSize(x, FONTE, escala, esp)[0][0] <= largura
    pedacos = []
    for palavra in txt.split(" "):
        while len(palavra) > 1 and not cabe(palavra):
            k = len(palavra) - 1
            while k > 1 and not cabe(palavra[:k]):
                k -= 1
            pedacos.append(palavra[:k])
            palavra = palavra[k:]
        pedacos.append(palavra)
    linhas, atual = [], ""
    for p in pedacos:
        teste = (atual + " " + p).strip()
        if cabe(teste) or not atual:
            atual = teste
        else:
            linhas.append(atual)
            atual = p
    if atual:
        linhas.append(atual)
    return linhas


def desenhar_painel(altura: int, info_alfa: dict | None, gravando: bool, agora: float) -> np.ndarray:
    pn = np.full((altura, PAINEL_L, 3), COR_FUNDO, np.uint8)
    x0, y = 24, 44
    larg = PAINEL_L - 2 * x0
    cv2.putText(pn, "LIBRE-SE", (x0, y), FONTE, 1.0, COR_TEXTO, 2)
    cv2.putText(pn, f"{fps:4.1f} fps", (PAINEL_L - 130, y), FONTE, 0.6, COR_SUAVE, 1)
    y += 34
    cor_modo = COR_ALFA if mode == "ALFABETO" else COR_PAL
    cv2.rectangle(pn, (x0, y - 24), (x0 + 200, y + 10), cor_modo, -1)
    cv2.putText(pn, mode, (x0 + 12, y + 2), FONTE, 0.75, (20, 20, 20), 2)
    y += 36
    cv2.line(pn, (x0, y), (PAINEL_L - x0, y), COR_LINHA, 1)

    # texto reconhecido
    y += 30
    cv2.putText(pn, "TEXTO", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
    for ln in quebrar(asc(texto_completo()) or "-", larg, 1.1, 2)[-4:]:
        y += 42
        cv2.putText(pn, ln, (x0, y), FONTE, 1.1, COR_TEXTO, 2)
    y += 22
    cv2.line(pn, (x0, y), (PAINEL_L - x0, y), COR_LINHA, 1)

    # agora
    y += 30
    if mode == "ALFABETO":
        cv2.putText(pn, "LETRA AGORA", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
        if info_alfa and info_alfa["letra"]:
            ok = info_alfa["aceita"]
            cor = COR_OK if ok else COR_SUAVE
            y += 62
            cv2.putText(pn, info_alfa["letra"] + ("" if ok else "?"), (x0, y), FONTE, 2.2, cor, 4)
            cv2.putText(pn, f"{info_alfa['conf']*100:.0f}%", (x0 + 110, y - 8), FONTE, 0.9, cor, 2)
            if info_alfa["movendo"]:
                cv2.putText(pn, "movimento...", (x0 + 220, y - 8), FONTE, 0.7, COR_PAL, 2)
            y += 34
            cv2.putText(pn, "   ".join(f"{l} {c*100:.0f}%" for l, c in info_alfa["top"]),
                        (x0, y), FONTE, 0.6, COR_SUAVE, 1)
        else:
            y += 62
            cv2.putText(pn, "mostre a mao", (x0, y - 10), FONTE, 0.8, COR_SUAVE, 2)
            y += 34
    else:
        if gravando:
            cv2.circle(pn, (x0 + 8, y - 6), 8, (60, 60, 230), -1)
            cv2.putText(pn, "GRAVANDO SINAL...", (x0 + 26, y), FONTE, 0.6, (90, 90, 240), 2)
            y += 50
            cv2.putText(pn, "abaixe a mao para terminar", (x0, y), FONTE, 0.65, COR_SUAVE, 1)
        elif ultimo_top:
            cv2.putText(pn, "ULTIMO SINAL", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
            for k, (pal, conf) in enumerate(ultimo_top):
                aceito = k == 0 and ultimo_ok
                y += 56 if k == 0 else 30
                cv2.putText(pn, f"{asc(pal).upper()}  {conf*100:.0f}%", (x0, y), FONTE,
                            1.2 if k == 0 else 0.7, COR_OK if aceito else COR_SUAVE,
                            3 if k == 0 else 1)
            if not ultimo_ok:
                y += 28
                cv2.putText(pn, "(incerto - nao entrou no texto)", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
        else:
            cv2.putText(pn, "ULTIMO SINAL", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
            y += 50
            cv2.putText(pn, "faca o sinal e abaixe a mao", (x0, y), FONTE, 0.65, COR_SUAVE, 1)
    if agora < entrou_ate:
        y += 34
        cv2.putText(pn, f"+ {asc(entrou)}", (x0, y), FONTE, 0.9, COR_OK, 2)
    y += 24
    cv2.line(pn, (x0, y), (PAINEL_L - x0, y), COR_LINHA, 1)

    # vocabulario (modo palavras)
    if mode == "PALAVRAS" and words is not None:
        y += 26
        cv2.putText(pn, "SINAIS QUE O MODELO CONHECE", (x0, y), FONTE, 0.5, COR_SUAVE, 1)
        for ln in quebrar(", ".join(asc(c).lower() for c in words.classes), larg, 0.55, 1)[:4]:
            y += 24
            cv2.putText(pn, ln, (x0, y), FONTE, 0.55, COR_TEXTO, 1)
        y += 16
        cv2.line(pn, (x0, y), (PAINEL_L - x0, y), COR_LINHA, 1)

    # historico
    y += 28
    cv2.putText(pn, "HISTORICO", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
    for item in reversed(historico):
        if y > altura - 120:
            break
        y += 28
        cv2.putText(pn, asc(item), (x0, y), FONTE, 0.65, COR_TEXTO, 1)

    # rodape
    if agora < aviso_ate:
        cv2.putText(pn, asc(aviso), (x0, altura - 70), FONTE, 0.5, COR_OK, 1)
    cv2.putText(pn, "M modo   ESPACO limpa   BACKSPACE apaga", (x0, altura - 42),
                FONTE, 0.5, COR_SUAVE, 1)
    cv2.putText(pn, "S salva trecho (debug)   Q sai", (x0, altura - 18),
                FONTE, 0.5, COR_SUAVE, 1)
    return pn


JANELA = "Libre-se"
cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
janela_ajustada = False         # tamanho inicial da janela, feito no 1o quadro
print("\nM = modo | ESPACO = limpar | BACKSPACE = apagar | S = salvar trecho | Q = sair")
print(f"Tudo o que entra no texto tambem vai para: {SAIDA}\n")

while True:
    ret, frame = leitor.read()
    if not ret:
        break
    t = time.time()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))

    lm = None
    if res and res.hand_landmarks:
        hand = res.hand_landmarks[0]
        lm = np.array([c for p in hand for c in (p.x, p.y, p.z)], dtype=np.float64)
        h, w = frame.shape[:2]
        for p in hand:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
    if lm is not None and len(lm) != FEAT_DIM:
        lm = None
    trecho.append((t, lm))
    while trecho and t - trecho[0][0] > TRECHO_S:
        trecho.popleft()

    info_alfa = None
    # ------------------------------------------------------- ALFABETO
    if mode == "ALFABETO":
        info_alfa = soletrador.update(lm, t)
        acao, nova = info_alfa["acao"], info_alfa["letra"]
        if acao == "troca" and letra_atual:
            antiga, letra_atual[-1] = letra_atual[-1], nova
            registrar(f"{antiga} -> {nova}", "letra", t)
        elif acao in ("add", "troca"):
            letra_atual.append(nova)
            registrar(nova, "letra", t)

    # ------------------------------------------------------- PALAVRAS
    else:
        sinal = segmenter.update(lm, t)
        if sinal is not None:
            ultimo_top = words.predict(sinal)
            ultimo_ok = accept_word(ultimo_top)
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in ultimo_top)
            print(f"  sinal de {len(sinal)} frames -> {resumo}"
                  f"{'' if ultimo_ok else '  (incerto)'}")
            if ultimo_ok:
                fechar_soletracao()
                texto.append(ultimo_ok.upper())
                registrar(ultimo_ok.upper(), "palavra", t)
        if segmenter.recording:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (0, 0, 255), -1)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now

    vis = frame
    if vis.shape[0] != ALTURA_TELA:
        esc = ALTURA_TELA / vis.shape[0]
        vis = cv2.resize(vis, (round(vis.shape[1] * esc), ALTURA_TELA))
    painel = desenhar_painel(ALTURA_TELA, info_alfa, segmenter.recording, t)
    tela = np.hstack([vis, painel])
    if not janela_ajustada:      # 75% do tamanho real, na proporcao certa (pode maximizar)
        cv2.resizeWindow(JANELA, int(tela.shape[1] * 0.75), int(tela.shape[0] * 0.75))
        janela_ajustada = True
    cv2.imshow(JANELA, tela)
    key = cv2.waitKey(1) & 0xFF
    tecla = chr(key).lower() if 32 <= key < 127 else ""   # aceita maiuscula (Caps Lock)
    if tecla == "q":
        break
    elif tecla == "m":
        alvo = "PALAVRAS" if mode == "ALFABETO" else "ALFABETO"
        if (alvo == "PALAVRAS" and words is None) or (alvo == "ALFABETO" and letters is None):
            print(f"  modo {alvo} indisponivel (modelo nao carregado)")
            aviso, aviso_ate = f"modo {alvo} indisponivel (modelo nao carregado)", t + 4
        else:
            mode = alvo
            fechar_soletracao()
            segmenter.reset()
            if soletrador is not None:
                soletrador.reset()
            print(f"  modo: {mode}")
    elif tecla == "s":
        nome = salvar_trecho()
        if nome:
            print(f"  trecho salvo: {nome}")
            aviso, aviso_ate = f"salvo: debug/{nome.name}", t + 4
    elif tecla == " ":
        texto.clear()
        letra_atual.clear()
        ultimo_top, ultimo_ok = [], None
    elif key == 8:
        if letra_atual:
            letra_atual.pop()
        elif texto:
            texto.pop()

cap.release()
cv2.destroyAllWindows()
print(f"Texto final: {texto_completo() or '(vazio)'}")
