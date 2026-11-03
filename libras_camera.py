#!/usr/bin/env python3
"""
Libre-se - camera unica para ALFABETO, PALAVRAS e CALIBRADO.

Uso: python libras_camera.py              (escolhe a camera sozinho)
     python libras_camera.py --camera 1   (abre a camera 1; 0, 1, 2...)
     python libras_camera.py --usuario julia   (calibracao de cada pessoa)

A janela tem duas partes: a camera (so com os pontos da mao) e, ao lado, um
PAINEL ESCURO com o texto reconhecido, a letra/palavra do momento, as 3
opcoes mais provaveis e o historico. O painel nao depende da luz da sala.
Tudo o que entra no texto tambem e gravado em saida_libras.txt.

Teclas:
  M          = troca o modo: ALFABETO -> PALAVRAS -> CALIBRADO
  ESPACO     = limpa o texto montado
  BACKSPACE  = apaga o ultimo item
  S          = salva os ultimos 6 s de pontos da mao em debug/ (so numeros,
               sem imagem), para ajustar o reconhecimento com maos reais
  C          = troca de camera (a proxima: 0, 1, 2... e volta)
  Q          = sair

Camera: sem --camera, o programa testa as cameras e pula as que mandam imagem
preta ou PARADA (camera virtual do OBS desligada, NVIDIA Broadcast...). Se a
imagem ficar parada ou preta durante o uso, aparece um aviso: aperte C.

Modo ALFABETO: letra parada entra depois de ~0,3 s estavel. H, J, K, X e Z
tem movimento: faca o movimento e, no fim, pare a mao ou abaixe; a letra
entra quando o movimento termina. Para repetir a mesma letra (OSSO), tire a
mao do quadro entre as duas.

Modo PALAVRAS: comece com a mao fora do quadro (ou abaixada), faca o sinal
e ABAIXE a mao. O sinal tambem termina se a mao ficar parada no ar por ~1 s
depois de se mexer. O painel mostra as 3 palavras mais provaveis do ultimo
sinal e a lista de sinais que o modelo conhece.

Se existir modelo_palavras_v3.pkl (treino_v3.py), o modo PALAVRAS usa a v3:
as DUAS maos e a posicao delas em relacao ao corpo (nariz e ombros, pelo
MediaPipe Pose). Sem ele, usa o modelo v2 (modelo_rotulado_xgb.pkl, uma mao).

Modo CALIBRADO: reconhece as palavras de vocabulario.txt (oi, bom dia, numeros...)
comparando cada sinal com exemplos da PROPRIA pessoa (calibracao.py). Na
primeira vez, o app pede cada sinal algumas vezes ("Faca o sinal: BOM DIA, 2
de 5"): faca o sinal e abaixe a mao. Sinal que nao parece com nenhum exemplo
vira "nao reconhecido". Teclas do modo CALIBRADO:
  K = comeca / termina a calibracao
  V = mostra, num quadro no canto da imagem, como o sinal foi feito na
      calibracao (o exemplo mais tipico, com as maos e o corpo em movimento).
      Calibrando: o da palavra da vez. Reconhecendo: o do ultimo sinal; V de
      novo passa para a proxima palavra.
      Se voce ainda nao gravou a palavra, mostra o exemplo do perfil
      "referencia" (alguem que conhece os sinais calibrou com --usuario
      referencia); calibrando, V de novo alterna entre o seu e o da referencia.
  H = "como se faz?": abre no navegador o video do sinal da vez num
      dicionario de Libras (Spread the Sign; no ALFABETO, o alfabeto manual)
  (durante a calibracao) N = proxima palavra, P = anterior,
  R = refaz a palavra (apaga os exemplos dela), D = desfaz o ultimo exemplo
"""

import argparse
import sys
import time
import unicodedata
import webbrowser
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from calibracao import (DICIONARIO_ALFABETO, DICIONARIO_INES, PERFIL_REFERENCIA, Calibracao, avaliar_exemplo,
                        exemplo_referencia, link_dicionario, segmentador_calibrado, sequencia, slug)
from libres_features import FEAT_DIM
from libres_v3 import rastrear
from libres_realtime import ExposicaoCamera, estourada
from libres_realtime import (LeitorCamera, LetterRecognizer, MonitorImagem, SignSegmenter,
                             SignSegmenterV3, Soletrador, WordRecognizer, WordRecognizerV3,
                             abrir_camera, accept_word, trocar_camera)

ap = argparse.ArgumentParser(description="Libre-se: camera para ALFABETO e PALAVRAS em Libras")
ap.add_argument("--camera", type=int, default=None, metavar="N",
                help="numero da camera (0, 1, 2...). Sem isso, o programa escolhe sozinho e "
                     "pula cameras de imagem preta ou parada (OBS...). Na janela, C troca.")
ap.add_argument("--usuario", default="padrao",
                help="perfil da calibracao (cada pessoa calibra o seu): calibracao/<usuario>/")
ap.add_argument("--vocabulario", default=str(Path(__file__).parent / "vocabulario.txt"),
                help="palavras do modo CALIBRADO, uma por linha (criado na 1a vez)")
ap.add_argument("--repeticoes", type=int, default=5,
                help="quantas vezes cada sinal e feito na calibracao (padrao 5)")
ap.add_argument("--referencia", default=PERFIL_REFERENCIA,
                help="perfil de quem conhece os sinais: a tecla V mostra o exemplo dele quando "
                     "voce ainda nao gravou a palavra (padrao: referencia)")
args = ap.parse_args()

PROJECT_DIR = Path(__file__).parent
WORD_MODEL = PROJECT_DIR / "modelo_rotulado_xgb.pkl"
WORD_MODEL_V3 = PROJECT_DIR / "modelo_palavras_v3.pkl"
LETTER_MODEL = PROJECT_DIR / "modelo_alfabeto.pkl"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"


# ---------------------------------------------------------------- modelos
letters = words = None
words_v3 = None                 # palavras v3 (duas maos + corpo)
det_v3 = None                   # detectores da v3: 2 maos + pose


def obter_det_v3():
    """Detectores da v3 (2 maos + pose), criados na primeira vez que precisar.
    Os MESMOS detectores (e limiares) da extracao dos videos de treino."""
    global det_v3
    if det_v3 is None:
        from extrair_minds_v3 import DetectorV3
        det_v3 = DetectorV3(HAND_MODEL)
    return det_v3


try:
    letters = LetterRecognizer(LETTER_MODEL)
    print(f"Alfabeto: {LETTER_MODEL.name} carregado")
except Exception as e:  # noqa: BLE001
    print(f"[aviso] alfabeto indisponivel: {e}")
if WORD_MODEL_V3.exists():
    try:
        words_v3 = WordRecognizerV3(WORD_MODEL_V3)
        obter_det_v3()
        words = words_v3
        print(f"Palavras v3 (duas maos + corpo): {len(words.classes)} classes -> "
              f"{', '.join(words.classes)}")
    except (Exception, SystemExit) as e:  # noqa: BLE001
        print(f"[aviso] palavras v3 indisponiveis ({e}); usando o modelo v2")
        words_v3 = det_v3 = None
if words is None:
    try:
        words = WordRecognizer(WORD_MODEL)
        print(f"Palavras: {len(words.classes)} classes -> {', '.join(words.classes)}")
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] palavras indisponiveis: {e}")
_calibrado_ok: bool | None = None


def calibrado_disponivel() -> bool:
    """O modo CALIBRADO so precisa dos detectores (nao depende de modelo treinado)."""
    global _calibrado_ok
    if _calibrado_ok is None:
        try:
            obter_det_v3()
            _calibrado_ok = True
        except (Exception, SystemExit) as e:  # noqa: BLE001
            print(f"[aviso] modo CALIBRADO indisponivel: {e}")
            _calibrado_ok = False
    return _calibrado_ok


MODOS = ["ALFABETO", "PALAVRAS", "CALIBRADO"]


def modo_disponivel(m: str) -> bool:
    if m == "ALFABETO":
        return letters is not None
    if m == "PALAVRAS":
        return words is not None
    return calibrado_disponivel()


if letters is None and words is None and not calibrado_disponivel():
    sys.exit("Nenhum modelo carregado.")

mode = next(m for m in MODOS if modo_disponivel(m))

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
cap = abrir_camera(indice=args.camera)
if cap is None:
    sys.exit("Nao foi possivel ler a camera. Feche outros apps que usam a webcam "
             "(Teams, Discord, navegador, OBS) ou escolha outra: python libras_camera.py --camera 1")
leitor = LeitorCamera(cap)
monitor = MonitorImagem()       # avisa se a imagem ficar parada (camera virtual) ou preta
exposicao = ExposicaoCamera(cap) if cap is not None else None   # teclas [ ] E
clara_desde, aviso_luz_em = None, -1e9

segmenter = SignSegmenterV3() if words_v3 is not None else SignSegmenter()
texto: list[str] = []          # itens ja reconhecidos (letras e palavras)
letra_atual: list[str] = []    # letras sendo soletradas
soletrador = Soletrador(letters) if letters is not None else None
ultimo_top: list[tuple[str, float]] = []
ultimo_ok: str | None = None
t_prev = time.time()
fps = 0.0
aspecto_cam = 16 / 9           # largura / altura da imagem (atualizado a cada quadro)

# ---------------------------------------------------------------- calibracao
cal = Calibracao(args.usuario, Path(args.vocabulario), repeticoes=max(2, args.repeticoes))
rec_cal = None                 # reconhecedor com os exemplos (montado ao entrar no modo)
seg_cal = segmentador_calibrado()
calibrando = False
cal_idx = 0                    # palavra sendo calibrada
cal_msg, cal_msg_ate, cal_msg_ruim = "", 0.0, False
cal_resumo: list[str] = []     # resultado da conferencia (painel)
cal_sessao: dict = {}          # exemplos gravados agora, por palavra (aviso de exemplo ruim)
ultimo_salvo = None            # (palavra, arquivo) do ultimo exemplo, para a tecla D
ultimo_melhor = None           # palavra mais parecida quando o sinal nao foi reconhecido
exemplo_vis = None             # exemplo sendo mostrado com a tecla V (dict) ou None
ver_palavra = None             # palavra que a tecla V mostra fora da calibracao


def texto_completo() -> str:
    partes = texto + (["".join(letra_atual)] if letra_atual else [])
    return " ".join(partes)


def fechar_soletracao() -> None:
    if letra_atual:
        texto.append("".join(letra_atual))
        letra_atual.clear()


def msg_cal(txt: str, ruim: bool = False, dur: float = 4.0) -> None:
    global cal_msg, cal_msg_ate, cal_msg_ruim
    cal_msg, cal_msg_ate, cal_msg_ruim = txt, time.time() + dur, ruim


def iniciar_calibracao(idx: int) -> None:
    global calibrando, cal_idx
    calibrando, cal_idx = True, idx
    seg_cal.reset()
    cal_sessao.clear()
    print(f"\nCALIBRACAO de '{cal.usuario}': faca cada sinal {cal.repeticoes} vezes e abaixe a mao "
          f"depois de cada um.\n  vocabulario ({len(cal.palavras)} palavras): {cal.vocab_arq}")
    print(f"  palavra {cal_idx + 1}: {cal.palavras[cal_idx]}")


def mostrar_conferencia() -> None:
    """Confere a calibracao (cada exemplo contra os outros) e diz o que refazer."""
    global cal_resumo
    if rec_cal is None:
        cal_resumo = ["nenhum sinal calibrado: K calibra"]
        return
    c = rec_cal.conferencia()
    cal_resumo = []
    if c["total"]:
        cal_resumo.append(f"conferencia: {c['certos']}/{c['total']} exemplos certos")
    print(f"\nCalibracao de '{cal.usuario}': {len(rec_cal.classes)} palavras, "
          f"{len(rec_cal.rotulos)} exemplos. Conferencia: {c['certos']}/{c['total']} certos, "
          f"{c['errados']} trocados, {c['recusados']} recusados.")
    for p, (ok, n) in c["por_palavra"].items():
        if n and ok < n:
            print(f"  {p}: {ok}/{n}")
    for p, q, n in c["confusoes"][:5]:
        print(f"  '{p}' saiu como '{q}' {n}x")
    for p, q, r in c["parecidos"][:5]:
        print(f"  PARECIDOS: '{p}' e '{q}' -> refaca um deles (R) mais diferente/caprichado")
        if len(cal_resumo) < 3:
            cal_resumo.append(f"parecidos: {p} ~ {q} (R refaz)")
    for p, n in c["destoantes"].items():
        print(f"  '{p}': {n} exemplo(s) bem diferente(s) dos outros -> R refaz a palavra")
        if len(cal_resumo) < 3:
            cal_resumo.append(f"{p}: exemplo estranho (R refaz)")
    faltam = [p for p in cal.palavras if cal.n(p) == 0]
    if faltam:
        print(f"  sem calibrar: {', '.join(faltam)}")


def terminar_calibracao() -> None:
    global calibrando, rec_cal
    calibrando = False
    seg_cal.reset()
    rec_cal = cal.reconhecedor()
    mostrar_conferencia()


def entrar_calibrado() -> None:
    """Ao entrar no modo: carrega os exemplos; se falta calibrar alguma palavra, calibra."""
    global rec_cal, calibrando
    rec_cal = cal.reconhecedor()
    falta = cal.proxima_incompleta(0)
    if falta is not None:
        iniciar_calibracao(falta)
    else:
        calibrando = False
        mostrar_conferencia()


def tratar_exemplo(sinal: dict) -> None:
    """Um sinal feito durante a calibracao: vira exemplo da palavra da vez."""
    global cal_idx, rec_cal, ultimo_salvo
    palavra = cal.palavras[cal_idx]
    seq = sequencia(sinal, aspecto_cam)
    if seq is None:
        msg_cal("sinal curto demais: repita", ruim=True)
        return
    problema = avaliar_exemplo(rec_cal, palavra, seq, cal_sessao.get(palavra, []))
    arq = cal.salvar(palavra, sinal, aspecto_cam, fps)
    ultimo_salvo = (palavra, arq)
    cal_sessao.setdefault(palavra, []).append(seq)
    n = cal.n(palavra)
    print(f"  {palavra}: exemplo {n}/{cal.repeticoes}" + (f"  ({problema})" if problema else ""))
    if problema:
        msg_cal(f"{palavra} {n}/{cal.repeticoes}: {problema}. D desfaz", ruim=True, dur=5.0)
    else:
        msg_cal(f"{palavra} {n}/{cal.repeticoes} ok")
    if n >= cal.repeticoes:
        # inclui os exemplos novos sem reler tudo do disco (seria 1-2 s parado)
        novos = cal_sessao.pop(palavra, [])
        if rec_cal is None:
            rec_cal = cal.reconhecedor()
        elif palavra in rec_cal.classes:
            rec_cal = cal.reconhecedor()           # palavra ja tinha exemplos antigos: recarrega
        else:
            rec_cal = rec_cal.adicionar(palavra, novos)
        prox = cal.proxima_incompleta(cal_idx + 1)
        if prox is None:
            terminar_calibracao()
            msg_cal("calibracao pronta!", dur=5.0)
        else:
            cal_idx = prox
            if problema:
                msg_cal(f"{palavra}: {problema} (D desfaz). Agora: {cal.palavras[cal_idx]}",
                        ruim=True, dur=6.0)
            else:
                msg_cal(f"{palavra} pronto! Agora: {cal.palavras[cal_idx]}")
            print(f"  palavra {cal_idx + 1}: {cal.palavras[cal_idx]}")


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
COR_CAL = (200, 110, 190)      # lilas (BGR)
COR_ATENCAO = (60, 200, 255)   # amarelo (BGR)
COR_RUIM = (90, 90, 240)       # vermelho (BGR)
SAIDA = PROJECT_DIR / "saida_libras.txt"
DEBUG_DIR = PROJECT_DIR / "debug"
TRECHO_S = 6.0                 # segundos guardados para a tecla S

historico: list[str] = []
trecho: deque = deque()        # (t, landmarks|None, (maos, lado, pose)|None) dos ultimos TRECHO_S s
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
    ts = np.array([q[0] for q in trecho], dtype=np.float64)
    lms = np.full((len(trecho), FEAT_DIM), np.nan, dtype=np.float32)
    for i, q in enumerate(trecho):
        if q[1] is not None and len(q[1]) == FEAT_DIM:
            lms[i] = q[1]
    extra = {}
    if any(q[2] is not None for q in trecho):      # modo PALAVRAS v3: as duas maos e a pose
        maos = np.full((len(trecho), 2, 21, 3), np.nan, np.float32)
        lado = np.full((len(trecho), 2), -1, np.int8)
        pose = np.full((len(trecho), 33, 4), np.nan, np.float32)
        for i, q in enumerate(trecho):
            if q[2] is not None:
                maos[i], lado[i], pose[i] = q[2]
        extra = {"maos": maos, "lado": lado, "pose": pose}
    base = DEBUG_DIR / f"trecho_{time.strftime('%Y%m%d_%H%M%S')}"
    nome, n = base.with_suffix(".npz"), 2
    while nome.exists():
        nome, n = Path(f"{base}_{n}.npz"), n + 1
    np.savez_compressed(nome, t=ts, lm=lms, modo=mode, texto=texto_completo(),
                        aspecto=np.float32(aspecto_cam), **extra)
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
    cv2.putText(pn, f"camera {getattr(cap, 'indice', '?')}", (PAINEL_L - 130, y + 30),
                FONTE, 0.5, COR_SUAVE, 1)
    y += 34
    cor_modo = {"ALFABETO": COR_ALFA, "PALAVRAS": COR_PAL}.get(mode, COR_CAL)
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
    elif mode == "CALIBRADO" and calibrando:
        y = desenhar_calibracao(pn, x0, y, larg, gravando, agora)
    elif mode == "CALIBRADO" and rec_cal is not None and ultimo_top and not gravando:
        cv2.putText(pn, "ULTIMO SINAL", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
        if ultimo_ok:
            for k, (pal, conf) in enumerate(ultimo_top):
                y += 56 if k == 0 else 30
                cv2.putText(pn, f"{asc(pal).replace('_', ' ').upper()}  {conf*100:.0f}%", (x0, y), FONTE,
                            1.2 if k == 0 else 0.7, COR_OK if k == 0 else COR_SUAVE, 3 if k == 0 else 1)
        else:
            y += 56
            cv2.putText(pn, "NAO RECONHECIDO", (x0, y), FONTE, 1.1, COR_RUIM, 3)
            if ultimo_melhor:
                y += 32
                cv2.putText(pn, f"mais parecido: {asc(ultimo_melhor).upper()}", (x0, y), FONTE, 0.65,
                            COR_SUAVE, 1)
            y += 28
            cv2.putText(pn, "(nao entrou no texto)", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
    else:
        if gravando:
            cv2.circle(pn, (x0 + 8, y - 6), 8, (60, 60, 230), -1)
            cv2.putText(pn, "GRAVANDO SINAL...", (x0 + 26, y), FONTE, 0.6, (90, 90, 240), 2)
            y += 50
            cv2.putText(pn, "abaixe a mao para terminar", (x0, y), FONTE, 0.65, COR_SUAVE, 1)
        elif mode == "CALIBRADO" and rec_cal is None:
            cv2.putText(pn, "SEM CALIBRACAO", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
            y += 50
            cv2.putText(pn, "aperte K para calibrar", (x0, y), FONTE, 0.65, COR_ATENCAO, 1)
        elif ultimo_top:
            cv2.putText(pn, "ULTIMO SINAL", (x0, y), FONTE, 0.55, COR_SUAVE, 1)
            for k, (pal, conf) in enumerate(ultimo_top):
                aceito = k == 0 and ultimo_ok
                y += 56 if k == 0 else 30
                cv2.putText(pn, f"{asc(pal).replace('_', ' ').upper()}  {conf*100:.0f}%", (x0, y), FONTE,
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
        cv2.putText(pn, f"SINAIS QUE O MODELO CONHECE ({len(words.classes)}"
                    + (", v3)" if words_v3 is not None else ")"), (x0, y), FONTE, 0.5, COR_SUAVE, 1)
        linhas = quebrar(", ".join(asc(c).lower().replace("_", " ") for c in words.classes), larg, 0.55, 1)
        for k, ln in enumerate(linhas[:4]):
            if k == 3 and len(linhas) > 4:          # nao cabe tudo: avisa quantas faltam
                mostradas = sum(len(x.split(", ")) for x in linhas[:3])
                ln = f"... e mais {len(words.classes) - mostradas}"
            y += 24
            cv2.putText(pn, ln, (x0, y), FONTE, 0.55, COR_TEXTO, 1)
        y += 16
        cv2.line(pn, (x0, y), (PAINEL_L - x0, y), COR_LINHA, 1)

    if mode == "CALIBRADO" and not calibrando:
        y += 26
        prontas = [p for p in cal.palavras if cal.n(p) > 0]
        cv2.putText(pn, f"SINAIS CALIBRADOS ({len(prontas)} de {len(cal.palavras)})   K calibra",
                    (x0, y), FONTE, 0.5, COR_SUAVE, 1)
        lista = ", ".join(asc(p).lower() + ("" if cal.n(p) >= cal.repeticoes else "*") for p in prontas)
        for ln in quebrar(lista or "-", larg, 0.55, 1)[:3]:
            y += 24
            cv2.putText(pn, ln, (x0, y), FONTE, 0.55, COR_TEXTO, 1)
        for ln in cal_resumo[:2]:
            y += 24
            cv2.putText(pn, asc(ln), (x0, y), FONTE, 0.5,
                        COR_OK if ln.startswith("conferencia") else COR_ATENCAO, 1)
        if prontas:
            y += 24
            cv2.putText(pn, "V: ver o exemplo (de novo: proxima palavra)", (x0, y), FONTE, 0.45,
                        COR_SUAVE, 1)
        fps_cal = getattr(rec_cal, "fps_calibracao", None)
        if fps_cal and fps > 3 and fps < 0.7 * fps_cal:
            y += 24
            cv2.putText(pn, f"camera mais lenta que na calibracao ({fps:.0f}/{fps_cal:.0f} fps)",
                        (x0, y), FONTE, 0.45, COR_ATENCAO, 1)
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
    cv2.putText(pn, "S salva trecho   C camera   [ ] luz   Q sai", (x0, altura - 18),
                FONTE, 0.5, COR_SUAVE, 1)
    return pn


def desenhar_calibracao(pn: np.ndarray, x0: int, y: int, larg: int, gravando: bool, agora: float) -> int:
    """Parte do painel durante a calibracao: qual sinal fazer e quantas vezes."""
    palavra = cal.palavras[cal_idx]
    n, rep = cal.n(palavra), cal.repeticoes
    cv2.putText(pn, f"CALIBRACAO - palavra {cal_idx + 1} de {len(cal.palavras)}", (x0, y), FONTE, 0.55,
                COR_CAL, 1)
    y += 32
    cv2.putText(pn, "Faca o sinal:", (x0, y), FONTE, 0.6, COR_SUAVE, 1)
    for ln in quebrar(asc(palavra).upper(), larg, 1.3, 3)[:2]:
        y += 48
        cv2.putText(pn, ln, (x0, y), FONTE, 1.3, COR_TEXTO, 3)
    if n == 0:                                  # ainda nao gravou: mostra como aprender o sinal
        y += 28
        cv2.putText(pn, "nao sabe o sinal? H = video no dicionario", (x0, y), FONTE, 0.5, COR_ATENCAO, 1)
        if tem_referencia(palavra):
            y += 22
            cv2.putText(pn, "V = como a referencia faz", (x0, y), FONTE, 0.5, COR_ATENCAO, 1)
    y += 36
    for k in range(rep):
        cx = x0 + 10 + 26 * k
        if k < n:
            cv2.circle(pn, (cx, y - 6), 9, COR_OK, -1)
        else:
            cv2.circle(pn, (cx, y - 6), 9, COR_SUAVE, 1)
    cv2.putText(pn, f"{min(n, rep)} de {rep}", (x0 + 26 * rep + 12, y), FONTE, 0.6, COR_TEXTO, 1)
    y += 34
    if gravando:
        cv2.circle(pn, (x0 + 8, y - 6), 8, (60, 60, 230), -1)
        cv2.putText(pn, "GRAVANDO... abaixe a mao no fim", (x0 + 26, y), FONTE, 0.6, (90, 90, 240), 2)
    else:
        cv2.putText(pn, "faca o sinal e abaixe a mao", (x0, y), FONTE, 0.6, COR_SUAVE, 1)
    if agora < cal_msg_ate and cal_msg:
        for ln in quebrar(asc(cal_msg), larg, 0.55, 1)[:2]:
            y += 26
            cv2.putText(pn, ln, (x0, y), FONTE, 0.55, COR_ATENCAO if cal_msg_ruim else COR_OK, 1)
    y += 30
    cv2.putText(pn, "N proxima   P anterior   R refaz a palavra", (x0, y), FONTE, 0.5, COR_SUAVE, 1)
    y += 22
    cv2.putText(pn, "D desfaz   V ver exemplo   K termina", (x0, y), FONTE, 0.5, COR_SUAVE, 1)
    y += 22
    cv2.putText(pn, "H como se faz? (video no dicionario)", (x0, y), FONTE, 0.5, COR_SUAVE, 1)
    return y


def tem_referencia(palavra: str) -> bool:
    """O perfil de referencia tem exemplo gravado desta palavra?"""
    if slug(args.usuario) == slug(args.referencia):
        return False
    pasta = cal.pasta.parent / slug(args.referencia) / slug(palavra)
    return pasta.is_dir() and any(pasta.glob("*.npz"))


LIGACOES_MAO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                (17, 18), (18, 19), (19, 20)]
LIGACOES_CORPO = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
PONTOS_CORPO = [0, 11, 12, 13, 14, 15, 16, 23, 24]
VOLTAS_EXEMPLO = 2             # quantas vezes o exemplo passa
PAUSA_EXEMPLO = 0.6            # s parado no fim de cada passada


def preparar_exemplo(ex: dict) -> dict:
    """Arruma um exemplo gravado para desenhar: maos sempre no mesmo lugar (a
    ordem do MediaPipe muda de um quadro para outro), qual e a dominante e o
    retangulo que contem corpo e maos (em unidades de imagem, x * aspecto)."""
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
    if pose.ndim == 3:
        sel = pose[:, PONTOS_CORPO]
        ok = np.isfinite(sel[..., 0]) & (np.nan_to_num(sel[..., 3]) > 0.5) & (sel[..., 1] < 1.05)
        pts.append(np.c_[sel[ok][:, 0] * asp, sel[ok][:, 1]])
        cab = ok[:, 0] & ok[:, 1] & ok[:, 2]           # nariz e ombros: inclui o topo da cabeca desenhada
        if cab.any():                                  # (circulo no nariz, raio 0,28 x largura dos ombros)
            larg = np.hypot((sel[cab, 1, 0] - sel[cab, 2, 0]) * asp, sel[cab, 1, 1] - sel[cab, 2, 1])
            pts.append(np.c_[sel[cab, 0, 0] * asp, sel[cab, 0, 1] - 0.28 * larg])
    P = np.vstack(pts)
    (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    mx, my = 0.08 * max(x1 - x0, 0.2), 0.08 * max(y1 - y0, 0.2)
    fps_ex = float(ex.get("fps", 30.0))
    return {**ex, "maos": maos, "pose": pose, "dom": int(np.argmax(caminho)),
            "caixa": (x0 - mx, y0 - my, x1 + mx, y1 + my),
            "fps": fps_ex if 8.0 <= fps_ex <= 60.0 else 30.0}


def desenhar_exemplo(img: np.ndarray, ex: dict, agora: float) -> bool:
    """Quadro no canto superior esquerdo com o exemplo em movimento (maos e
    corpo, como no espelho da camera). False quando terminou de passar."""
    T = len(ex["maos"])
    dur = T / ex["fps"]
    volta = dur + PAUSA_EXEMPLO
    dt = agora - ex["t0"]
    if dt >= VOLTAS_EXEMPLO * volta or T == 0:
        return False
    no_ciclo = dt % volta
    k = min(T - 1, int(no_ciclo * ex["fps"]))
    H, W = img.shape[:2]
    pw = min(max(int(W * 0.40), 460), W - 32)
    ph = int(pw * 0.68)
    x0, y0 = 16, 16
    roi = img[y0:y0 + ph, x0:x0 + pw]
    roi[:] = (roi * 0.15 + np.array(COR_FUNDO) * 0.85).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), COR_CAL, 2)
    cv2.putText(img, f"EXEMPLO: {asc(ex['palavra']).upper()}", (x0 + 14, y0 + 32), FONTE, 0.75,
                COR_TEXTO, 2)
    sub = ("como a referencia faz (voce ainda nao gravou este)" if ex.get("referencia")
           else f"como foi feito na calibracao ({ex['indice'] + 1} de {ex['n']})")
    cv2.putText(img, sub, (x0 + 14, y0 + 56), FONTE, 0.45, COR_SUAVE, 1)
    ax0, ay0, aw, ah = x0 + 12, y0 + 66, pw - 24, ph - 90
    bx0, by0, bx1, by1 = ex["caixa"]
    esc = min(aw / max(bx1 - bx0, 1e-6), ah / max(by1 - by0, 1e-6))
    ox = ax0 + (aw - esc * (bx1 - bx0)) / 2
    oy = ay0 + (ah - esc * (by1 - by0)) / 2
    asp = float(ex["aspecto"])

    def pt(x, y):
        return int(ox + (x * asp - bx0) * esc), int(oy + (y - by0) * esc)

    pose = ex["pose"][k] if ex["pose"].ndim == 3 else None
    if pose is not None:
        visivel = {i for i in PONTOS_CORPO
                   if np.isfinite(pose[i, 0]) and np.nan_to_num(pose[i, 3]) > 0.5 and pose[i, 1] < 1.05}
        for a, b in LIGACOES_CORPO:
            if a in visivel and b in visivel:
                cv2.line(img, pt(*pose[a, :2]), pt(*pose[b, :2]), (150, 150, 150), 2)
        if 0 in visivel:
            raio = 12
            if 11 in visivel and 12 in visivel:
                raio = int(0.28 * esc * np.hypot((pose[11, 0] - pose[12, 0]) * asp, pose[11, 1] - pose[12, 1]))
            cv2.circle(img, pt(*pose[0, :2]), max(raio, 6), (150, 150, 150), 2)
    for slot in range(2):
        mao = ex["maos"][k, slot]
        if np.isnan(mao[0, 0]):
            continue
        cor = (0, 255, 0) if slot == ex["dom"] else (255, 200, 0)
        for a, b in LIGACOES_MAO:
            cv2.line(img, pt(*mao[a, :2]), pt(*mao[b, :2]), cor, 2)
        for q in mao:
            cv2.circle(img, pt(*q[:2]), 3, cor, -1)
    frac = min(1.0, no_ciclo / dur) if dur > 0 else 1.0
    cv2.rectangle(img, (x0 + 12, y0 + ph - 16), (x0 + pw - 12, y0 + ph - 10), COR_LINHA, -1)
    cv2.rectangle(img, (x0 + 12, y0 + ph - 16), (x0 + 12 + int((pw - 24) * frac), y0 + ph - 10), COR_CAL, -1)
    return True


def aviso_camera(img: np.ndarray, estado: str) -> None:
    """Faixa no pe da imagem quando a camera esta mandando imagem parada ou preta."""
    h, w = img.shape[:2]
    if estado == "parada":
        titulo = "IMAGEM PARADA"
        txt = "Parece uma camera virtual (OBS, NVIDIA Broadcast...)."
    else:
        titulo = "IMAGEM PRETA"
        txt = "Camera tampada, desligada ou virtual."
    linhas = quebrar(txt + " Aperte C para trocar de camera.", w - 48, 0.75, 2)
    topo = h - (62 + 36 * len(linhas))
    img[topo:] = (img[topo:] * 0.25).astype(np.uint8)
    cv2.putText(img, titulo, (24, topo + 44), FONTE, 1.1, (0, 220, 255), 3)
    for k, ln in enumerate(linhas):
        cv2.putText(img, ln, (24, topo + 44 + 36 * (k + 1)), FONTE, 0.75, COR_TEXTO, 2)


JANELA = "Libre-se"
cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
janela_ajustada = False         # tamanho inicial da janela, feito no 1o quadro
print("\nM = modo | ESPACO = limpar | BACKSPACE = apagar | S = salvar trecho | [ ] = luz da camera (E = auto) | "
      "C = trocar camera | Q = sair")
print("Modo CALIBRADO: K = calibrar | V = ver exemplo | N/P = proxima/anterior | R = refaz a palavra "
      "| D = desfaz | H = como se faz (dicionario)")
print(f"Tudo o que entra no texto tambem vai para: {SAIDA}\n")

while True:
    ret, frame = leitor.read()
    if not ret:
        break
    t = time.time()
    frame = cv2.flip(frame, 1)
    estado_img = monitor.update(frame, t)      # "", "parada" ou "preta"
    if estourada(frame) > 0.25:                # imagem muito clara: a mao some (pele clara, parede branca)
        clara_desde = t if clara_desde is None else clara_desde
        if t - clara_desde > 2 and t - aviso_luz_em > 60:
            aviso_luz_em = t
            aviso, aviso_ate = "imagem muito clara: [ escurece a camera", t + 8
            print("  imagem muito clara (a mao pode sumir): aperte [ para escurecer a camera, ] clareia, E automatico")
    else:
        clara_desde = None
    h, w = frame.shape[:2]
    aspecto_cam = w / h
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(mp.ImageFormat.SRGB, data=rgb)

    lm = None
    v3 = None                    # (maos, lado, pose) nos modos PALAVRAS v3 e CALIBRADO
    if mode == "CALIBRADO" or (mode == "PALAVRAS" and words_v3 is not None):
        maos, lado, pose = obter_det_v3().detectar(img, t)
        presentes = [k for k in range(2) if not np.isnan(maos[k, 0, 0])]
        if presentes:
            v3 = (maos, lado, pose)
            lm = maos[presentes[0]].reshape(-1).astype(np.float64)
            for k in presentes:
                cor = (0, 255, 0) if k == presentes[0] else (255, 200, 0)
                for x, y_, _ in maos[k]:
                    cv2.circle(frame, (int(x * w), int(y_ * h)), 4, cor, -1)
    else:
        res = landmarker.detect(img)
        if res and res.hand_landmarks:
            hand = res.hand_landmarks[0]
            lm = np.array([c for p in hand for c in (p.x, p.y, p.z)], dtype=np.float64)
            for p in hand:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
    if lm is not None and len(lm) != FEAT_DIM:
        lm = None
    trecho.append((t, lm, v3))
    while trecho and t - trecho[0][0] > TRECHO_S:
        trecho.popleft()

    info_alfa = None
    # ------------------------------------------------------- ALFABETO
    if mode == "ALFABETO":
        soletrador.aspecto = aspecto_cam
        info_alfa = soletrador.update(lm, t)
        acao, nova = info_alfa["acao"], info_alfa["letra"]
        if acao == "troca" and letra_atual:
            antiga, letra_atual[-1] = letra_atual[-1], nova
            registrar(f"{antiga} -> {nova}", "letra", t)
        elif acao in ("add", "troca"):
            letra_atual.append(nova)
            registrar(nova, "letra", t)

    # ------------------------------------------------------- CALIBRADO
    elif mode == "CALIBRADO":
        quadro = None if v3 is None else {"maos": v3[0], "lado": v3[1], "pose": v3[2]}
        sinal = seg_cal.update(quadro, t)
        if sinal is not None and calibrando:
            tratar_exemplo(sinal)
        elif sinal is not None and rec_cal is not None:
            res = rec_cal.predict(sinal, aspecto_cam)
            ultimo_top, ultimo_ok, ultimo_melhor = res["top"], res["aceito"], res["melhor"]
            ver_palavra = ultimo_ok or ultimo_melhor
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in ultimo_top)
            print(f"  sinal de {len(sinal['maos'])} frames -> {resumo}  (distancia/limite "
                  f"{res['razao']:.2f}){'' if ultimo_ok else '  NAO RECONHECIDO'}")
            if ultimo_ok:
                fechar_soletracao()
                texto.append(ultimo_ok.upper())
                registrar(ultimo_ok.upper(), "calibrado", t)
        if seg_cal.recording:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (0, 0, 255), -1)

    # ------------------------------------------------------- PALAVRAS
    else:
        if words_v3 is not None:
            quadro = None if v3 is None else {"maos": v3[0], "lado": v3[1], "pose": v3[2]}
            sinal = segmenter.update(quadro, t)
            if sinal is not None:
                ultimo_top = words_v3.predict(sinal, aspecto_cam)
                n_quadros = len(sinal["maos"])
        else:
            sinal = segmenter.update(lm, t)
            if sinal is not None:
                ultimo_top = words.predict(sinal)
                n_quadros = len(sinal)
        if sinal is not None:
            ultimo_ok = accept_word(ultimo_top)
            resumo = ", ".join(f"{p} {c*100:.0f}%" for p, c in ultimo_top)
            print(f"  sinal de {n_quadros} frames -> {resumo}"
                  f"{'' if ultimo_ok else '  (incerto)'}")
            if ultimo_ok:
                fechar_soletracao()
                texto.append(ultimo_ok.replace("_", " ").upper())       # conta_de_luz -> CONTA DE LUZ
                registrar(ultimo_ok.replace("_", " ").upper(), "palavra", t)
        if segmenter.recording:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 14, (0, 0, 255), -1)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now

    vis = frame
    if vis.shape[0] != ALTURA_TELA:
        esc = ALTURA_TELA / vis.shape[0]
        vis = cv2.resize(vis, (round(vis.shape[1] * esc), ALTURA_TELA))
    if exemplo_vis is not None and not desenhar_exemplo(vis, exemplo_vis, time.time()):
        exemplo_vis = None
    if estado_img:
        aviso_camera(vis, estado_img)
    painel = desenhar_painel(ALTURA_TELA, info_alfa,
                             seg_cal.recording if mode == "CALIBRADO" else segmenter.recording, t)
    tela = np.hstack([vis, painel])
    if not janela_ajustada:      # 75% do tamanho real, na proporcao certa (pode maximizar)
        cv2.resizeWindow(JANELA, int(tela.shape[1] * 0.75), int(tela.shape[0] * 0.75))
        janela_ajustada = True
    cv2.imshow(JANELA, tela)
    key = cv2.waitKey(1) & 0xFF
    tecla = chr(key).lower() if 32 <= key < 127 else ""   # aceita maiuscula (Caps Lock)
    if tecla in ("q", "m", "c") and mode == "ALFABETO" and soletrador is not None:
        # um gesto (J, Z...) que ainda nao terminou entra antes de trocar de modo/sair
        fim = soletrador.finalizar(t)
        if fim["acao"] == "troca" and letra_atual:
            antiga, letra_atual[-1] = letra_atual[-1], fim["letra"]
            registrar(f"{antiga} -> {fim['letra']}", "letra", t)
        elif fim["acao"] in ("add", "troca"):
            letra_atual.append(fim["letra"])
            registrar(fim["letra"], "letra", t)
    if tecla == "q":
        break
    elif tecla == "m":
        i = MODOS.index(mode)
        alvo = next((MODOS[(i + k) % len(MODOS)] for k in range(1, len(MODOS))
                     if modo_disponivel(MODOS[(i + k) % len(MODOS)])), mode)
        if alvo == mode:
            aviso, aviso_ate = "nenhum outro modo disponivel", t + 4
        else:
            mode = alvo
            fechar_soletracao()
            segmenter.reset()
            seg_cal.reset()
            calibrando = False
            exemplo_vis = None
            ultimo_top, ultimo_ok, ultimo_melhor = [], None, None
            if soletrador is not None:
                soletrador.reset()
            print(f"  modo: {mode}")
            if mode == "CALIBRADO":
                entrar_calibrado()
    elif tecla == "k" and mode == "CALIBRADO":
        if calibrando:
            terminar_calibracao()
            msg_cal("calibracao encerrada")
        else:
            falta = cal.proxima_incompleta(0)
            iniciar_calibracao(0 if falta is None else falta)
    elif tecla == "v" and mode == "CALIBRADO":
        prontas = [p for p in cal.palavras if cal.n(p) > 0]
        passando = exemplo_vis is not None             # V de novo enquanto o exemplo ainda passa
        if calibrando:
            alvo = cal.palavras[cal_idx]
        elif passando and exemplo_vis["palavra"] in prontas:
            alvo = prontas[(prontas.index(exemplo_vis["palavra"]) + 1) % len(prontas)]   # proxima
        elif ver_palavra in prontas:
            alvo = ver_palavra
        else:
            alvo = prontas[0] if prontas else None
        ex = None
        if alvo is not None:
            # sem exemplo seu: o da referencia; calibrando, V de novo alterna entre os dois
            quer_ref = (calibrando and passando and exemplo_vis["palavra"] == alvo
                        and not exemplo_vis.get("referencia"))
            ex = None if quer_ref else cal.exemplo_tipico(alvo)
            if ex is None:
                ex = exemplo_referencia(alvo, cal.vocab_arq, cal.pasta.parent, args.usuario, cal.repeticoes,
                                        args.referencia)
            if ex is None and quer_ref:
                ex = cal.exemplo_tipico(alvo)
        if ex is None:
            txt = (f"'{alvo}' ainda nao tem exemplo gravado: H abre o video no dicionario" if alvo
                   else "nenhum sinal calibrado ainda")
            if calibrando:
                msg_cal(txt, ruim=True)
            else:
                aviso, aviso_ate = txt, time.time() + 4
        else:
            exemplo_vis = preparar_exemplo(ex)
            exemplo_vis["t0"] = time.time()
            de = "da referencia" if ex.get("referencia") else f"({ex['indice'] + 1} de {ex['n']})"
            print(f"  mostrando o exemplo de '{alvo}' {de}")
    elif tecla == "h":                                  # como se faz: o sinal no dicionario de Libras
        if mode == "ALFABETO":
            url, sobre = DICIONARIO_ALFABETO, "alfabeto manual"
        elif mode == "CALIBRADO" and calibrando:
            sobre = cal.palavras[cal_idx]
            url = link_dicionario(sobre)
        elif ultimo_ok or ultimo_melhor:
            sobre = ultimo_ok or ultimo_melhor
            url = link_dicionario(sobre)
        else:
            url, sobre = DICIONARIO_INES, "dicionario do INES"
        try:
            abriu = webbrowser.open(url)
        except Exception:  # noqa: BLE001  (sem navegador: so mostra o endereco)
            abriu = False
        print(f"  como se faz ({sobre}): {url}")
        aviso = (f"dicionario aberto no navegador: {sobre}" if abriu
                 else "nao abriu o navegador: o endereco esta no terminal")
        aviso_ate = time.time() + 6
    elif tecla in ("n", "p") and mode == "CALIBRADO" and calibrando:
        cal_idx = (cal_idx + (1 if tecla == "n" else -1)) % len(cal.palavras)
        seg_cal.reset()
        print(f"  palavra {cal_idx + 1}: {cal.palavras[cal_idx]}")
    elif tecla == "r" and mode == "CALIBRADO" and calibrando:
        palavra = cal.palavras[cal_idx]
        n = cal.apagar_todos(palavra)
        cal_sessao.pop(palavra, None)
        rec_cal = cal.reconhecedor()
        seg_cal.reset()
        print(f"  '{palavra}': {n} exemplo(s) apagado(s); faca de novo")
        msg_cal(f"exemplos de '{palavra}' apagados: faca de novo")
    elif tecla == "d" and mode == "CALIBRADO" and calibrando:
        if ultimo_salvo is not None and ultimo_salvo[1].exists():
            palavra, arq = ultimo_salvo
            arq.unlink()
            if cal_sessao.get(palavra):
                cal_sessao[palavra].pop()
            cal_idx = cal.palavras.index(palavra) if palavra in cal.palavras else cal_idx
            rec_cal = cal.reconhecedor()
            ultimo_salvo = None
            print(f"  ultimo exemplo de '{palavra}' desfeito ({cal.n(palavra)}/{cal.repeticoes})")
            msg_cal(f"ultimo exemplo de '{palavra}' desfeito")
        else:
            msg_cal("nada para desfazer", ruim=True)
    elif tecla == "c":
        idx_antes = getattr(cap, "indice", 0)
        espera = tela.copy()                   # a janela trava enquanto a outra camera abre
        cv2.putText(espera, "trocando de camera...", (40, 90), FONTE, 1.4, (0, 220, 255), 3)
        cv2.imshow(JANELA, espera)
        cv2.waitKey(1)
        novo = trocar_camera(cap)
        if novo is None:
            print("  nenhuma camera abriu (nem a anterior)")
            break
        cap, leitor = novo, LeitorCamera(novo)
        exposicao = ExposicaoCamera(cap)
        monitor.reset()
        segmenter.reset()
        seg_cal.reset()
        if soletrador is not None:
            soletrador.reset()
        t_prev = time.time()
        idx = getattr(cap, "indice", "?")
        lw, lh = getattr(cap, "resolucao", (0, 0))
        if idx == idx_antes:
            print(f"  so existe a camera {idx}")
            aviso, aviso_ate = f"nenhuma outra camera; continua a camera {idx}", time.time() + 4
        else:
            print(f"  camera {idx} ({getattr(cap, 'backend', '')}, {lw}x{lh}). Para abrir direto "
                  f"nela: python libras_camera.py --camera {idx}")
            aviso, aviso_ate = f"camera {idx} (para abrir direto: --camera {idx})", time.time() + 5
    elif tecla in ("[", "]") and exposicao is not None:
        aviso, aviso_ate = exposicao.ajustar(-1 if tecla == "[" else 1), time.time() + 5
        print("  " + aviso)
    elif tecla == "e" and exposicao is not None:
        aviso, aviso_ate = exposicao.automatico(), time.time() + 4
        print("  " + aviso)
    elif tecla == "s":
        nome = salvar_trecho()
        if nome:
            print(f"  trecho salvo: {nome}")
            aviso, aviso_ate = f"salvo: debug/{nome.name}", t + 4
    elif tecla == " ":
        texto.clear()
        letra_atual.clear()
        ultimo_top, ultimo_ok, ultimo_melhor = [], None, None
    elif key == 8:
        if letra_atual:
            letra_atual.pop()
        elif texto:
            texto.pop()

cap.release()
cv2.destroyAllWindows()
print(f"Texto final: {texto_completo() or '(vazio)'}")
