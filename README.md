# 🤟 Libre-se

Reconhecimento de sinais em **Libras** (Língua Brasileira de Sinais) usando visão computacional e machine learning.

Extrai landmarks das mãos com **MediaPipe Hands**, calcula features temporais (estatísticas, ângulos dos dedos, distâncias relativas) e classifica com **XGBoost**.

## ⚡ v2 — palavras funcionando (outubro/2026)

O reconhecimento de **palavras** não funcionava porque a normalização zerava
o punho em todo frame: o modelo via só o formato da mão e não o **movimento**,
que é o que diferencia muitas palavras em Libras. Mudanças:

- `libres_features.py`: sequências de palavra agora têm 65 valores por frame
  (63 da forma da mão + trajetória x/y do punho em "tamanhos de mão") e
  15 features novas de movimento. O alfabeto continua exatamente igual.
- O sinal **inteiro** é reamostrado para 30 frames (antes: só os 30 primeiros).
- `integrate_minds.py`: 1 arquivo por vídeo do MINDS (antes: blocos de 30
  frames, muitos só com a mão subindo/descendo).
- "barulho" e "ruim" voltaram a ser treinados (são sinais reais do MINDS).
- Augmentation com espelhamento (webcam espelhada × MINDS, canhotos) e
  variação da amplitude do movimento.
- Na câmera, o sinal é segmentado: começa quando a mão aparece e termina
  quando ela sai do quadro ou fica parada ~1 s (`libres_realtime.py`).
- `libras_camera.py`: um app só para **alfabeto e palavras** (tecla `M`).

### Sem o CSV do MINDS? Extraia direto dos vídeos

Baixe os ZIPs dos sinalizadores em https://zenodo.org/records/2667329 (não precisa
de todos; 4 já bastam para treinar e avaliar), coloque em `minds_videos/` e rode:

```bash
python extrair_minds_videos.py
python train_rotulado.py
```

Não precisa descompactar: o script lê os vídeos de dentro dos ZIPs. Pode
interromper e rodar de novo; o que já foi processado é pulado.

### Coletar palavras pela webcam (dataset próprio, sem MINDS)

```bash
python coletar_palavras.py --sinalizador 1
python coletar_palavras.py --sinalizador 2 --palavras oi,obrigado,tchau --takes 15
```

Cada pessoa usa sempre o mesmo número de sinalizador. Faça o sinal e abaixe
a mão: o take é salvo sozinho em `coletados/<palavra>/SinalizadorNN_<palavra>_NNN.npy`.
Pastas `coletados/` de computadores diferentes podem ser juntadas (os nomes não
colidem). Depois: `python train_rotulado.py` e `python eval_loso.py`.

**Ordem para rodar depois de atualizar** (o modelo antigo não é compatível):

```bash
python integrate_minds.py      # regera coletados/ a partir do CSV do MINDS
python train_rotulado.py       # treina o modelo de palavras v2
python eval_loso.py            # acurácia por sinalizante (métrica do TCC)
python libras_camera.py        # câmera: alfabeto + palavras
```

Na câmera, modo PALAVRAS: comece com a mão fora do quadro, faça o sinal e
abaixe a mão.

## Funcionalidades

- Reconhecimento de letras do alfabeto (A-Z), números (1-10) e palavras específicas em Libras
- Pipeline completo: coleta → extração → treino → inferência em tempo real
- Aumentação de dados (ruído, escala, deslocamento, warp, dropout) para classes com poucas amostras
- Classificação com threshold de confiança (80%) para evitar falsos positivos
- Normalização dos landmarks centralizada no punho (torna independente da posição da mão)
- 639 features por signo: estatísticas dos landmarks, centróide, distâncias relativas e ângulos dos dedos

## Estrutura

```
Libre-se/
├── train_rotulado.py          # Pipeline de treino (extração → aumento → XGBoost)
├── test_camera_temporal.py    # Inferência em tempo real com webcam
├── collect_signs.py           # Coleta de sinais via webcam (A-Z, 1-10)
├── auto_split_signs.py        # Divisão automática de vídeos YouTube por movimento
├── slice_signs.py             # Divisão manual de vídeos com timestamps
├── integrate_minds.py         # Extração do dataset MINDS para formato .npy
├── modelo_rotulado_xgb.pkl    # Modelo treinado (61 classes, 639 features)
├── hand_landmarker.task       # Modelo MediaPipe de detecção de mãos
├── requirements.txt           # Dependências
└── README.md
```

## Instalação

```bash
git clone https://github.com/ElielFreitas/Libre-se.git
cd Libre-se
pip install -r requirements.txt
```

Baixe o modelo `hand_landmarker.task` do [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) e coloque na raiz do projeto (já incluso).

## Como usar

### Coletar sinais (A-Z e 1-10)

```bash
python collect_signs.py
```

Grava 5 takes por letra/número via webcam. Os arquivos `.npy` são salvos em `coletados/{classe}/`.

### Extrair dataset MINDS

```bash
python integrate_minds.py
```

Extrai os segmentos do dataset MINDS (CSV) e salva em `coletados/`.

### Treinar modelo

```bash
python train_rotulado.py
```

Processa vídeos em `palavras2/` e dados em `coletados/`, aplica aumentaçãono treino (3x), treina XGBoost com 639 features e salva `modelo_rotulado_xgb.pkl`.

### Testar em tempo real

```bash
python test_camera_temporal.py
```

Abre a webcam e exibe o sinal reconhecido com a confiança em tempo real.

### Processar vídeos YouTube

```bash
python auto_split_signs.py
```

Detecta segmentos de movimento em vídeos de compilações, verifica presença de mão com MediaPipe e corta com FFmpeg. Os clips salvos em `palavras2/` precisam ser renomeados antes do treino.

```bash
python slice_signs.py
```

Versão manual: digite `inicio fim label` por vídeo.

## Dataset MINDS

O modelo usa o dataset [MINDS](https://universe.robots.cloud.lncc.br) (Libras), que contém 4.518 amostras de 20 classes (~226 por classe), com landmarks extraídos por MediaPipe. Execute `integrate_minds.py` para extrair os segmentos do CSV.

## Features

Para cada frame de 21 landmarks (x, y, z), o pipeline calcula:

| Grupo | Features | Descrição |
|-------|----------|-----------|
| Landmarks normalizados | 504 | Média, mediana, std, min, max, range, diff para cada landmark (x,y,z) |
| Centróide | 15 | Média, mediana, std, min, max da posição do centróide |
| Distâncias relativas | 70 | Média, mediana, std, min, max, range, diff entre cada landmark e os demais |
| Ângulos dos dedos | 50 | 10 ângulos (MCP/PIP) × 5 estatísticas |

**Total: 639 features** por signo.

## Modelo atual

- **61 classes**: A-Y, 20 classes MINDS, 16 palavras avulsas
- **Algoritmo**: XGBoost (80 estimators, max_depth=4, learning_rate=0.15)
- **Acurácia no teste**: 45% (20 classes MINDS)
- **Acurácia no treino**: 96%

## Licença

MIT
