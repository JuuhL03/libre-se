# 🤟 Libre-se (SignBridge)

Protótipo do TCC **"SignBridge: Desenvolvimento de um Sistema Multiplataforma para
Reconhecimento de LIBRAS em Tempo Real e Tradução para Texto e Voz"** (Grupo 6,
Ciência da Computação, Faculdades Integradas de Bauru, 2026).

O sistema reconhece LIBRAS pela câmera comum do computador ou do celular e devolve o
resultado em texto e voz. A detecção das mãos e do corpo (MediaPipe) roda no aparelho, e
só as coordenadas dos pontos vão para o servidor, nunca a imagem. São três modos:

- **ALFABETO:** as 26 letras do alfabeto manual (Random Forest com regras de geometria; H,
  J, K, X e Z pelo movimento), com calibração opcional das letras de cada pessoa.
- **PALAVRAS:** palavras isoladas com as duas mãos e o corpo (XGBoost), treinadas com o
  MINDS-Libras e a LIBRAS-EQT-UECE (65 palavras).
- **CALIBRADO:** o vocabulário do dia a dia (cena da padaria) com exemplos da própria
  pessoa (DTW); o que não se parece com nenhum exemplo vira "não reconhecido".

O mesmo reconhecimento atende a **página web** (`web/` + `servidor/`, publicada no
Railway), ao **app móvel** Android e iOS (`app/librese/`) e ao **app de computador**
(`libras_camera.py`).

## 📁 Organização

| Pasta ou arquivo | O que é |
|---|---|
| `libres_features.py`, `libres_v3.py`, `libres_realtime.py`, `calibracao.py`, `alfabeto_calibrado.py`, `libres_modelos.py` | Módulos compartilhados: forma da mão, preparo e características das palavras, segmentador e reconhecedores, modo calibrado (DTW), calibração do alfabeto, salvar/abrir modelos. Treino, avaliação e servidor usam as mesmas funções |
| `servidor/` | Servidor FastAPI + WebSocket (`app.py`, `motor.py`), preparo da página e teste do servidor |
| `web/` | A página (HTML, CSS e JavaScript, sem frameworks) |
| `app/librese/` | App móvel em React Native com TypeScript; MediaPipe nativo em Kotlin e Swift |
| `libras_camera.py` | App de computador (janela com câmera e painel) |
| `treino_v3.py`, `avaliar_v3.py`, `diagnostico_v3.py`, `extrair_minds_v3.py`, `extrair_eqt_v3.py` | Palavras, versão final: extração das bases, treino, avaliação com pessoa fora do treino (LOSO) e diagnóstico |
| `train_alfabeto.py`, `coletar_alfabeto.py`, `importar_imagens_alfabeto.py` | Alfabeto: treino e coleta |
| `coletar_palavras.py` | Ferramenta de coleta de novos sinais (mesmo segmentador do app) |
| `teste_v3.py`, `teste_soletrador.py`, `teste_calibracao.py`, `servidor/teste_servidor.py` | Testes sem câmera (casos de teste do TCC) |
| `train_rotulado.py`, `eval_loso.py`, `integrate_minds.py`, `extrair_minds_videos.py`, `analisar_palavras.py` | Primeira e segunda versões do reconhecimento de palavras (seções 3.6.3 e 3.6.4 do TCC); o servidor usa o modelo da segunda versão como reserva |
| `ver_sinais.py`, `reproduzir_trecho.py` | Ferramentas de análise (confusões entre sinais, trechos gravados com a tecla S) |
| `modelo_alfabeto.pkl`, `modelo_palavras_v3.pkl`, `modelo_rotulado_xgb.pkl`, `hand_landmarker.task` | Modelos treinados e o modelo de mãos do MediaPipe |
| `vocabulario.txt` | Palavras do modo CALIBRADO |
| `pesquisa/` | Script do formulário da avaliação com usuários |
| `Dockerfile`, `publicar_site.bat` | Publicação da versão web |
| `avaliar_tcc.bat`, `avaliar_sem_preparo.bat` | Reproduzem as Tabelas 2 e 3 do TCC (Windows) |

## 🚀 Como rodar

**Versão web no computador** (a câmera funciona em `localhost`):

```bash
pip install -r servidor/requirements.txt -r servidor/requirements-modelos.txt
python servidor/preparar_web.py          # 1a vez: baixa o MediaPipe do navegador (~22 MB)
uvicorn servidor.app:app --port 8000     # abra http://localhost:8000
```

**App de computador:**

```bash
pip install -r requirements.txt
python libras_camera.py                  # M troca o modo: ALFABETO → PALAVRAS → CALIBRADO
```

**App móvel:** veja [App mobile](#-app-mobile-android-e-ios).

## 📊 Reproduzir os números do TCC

As bases públicas **não vêm junto** (licenças): baixe o MINDS-Libras
(https://zenodo.org/records/2667329, ZIPs em `minds_videos/`) e a LIBRAS-EQT-UECE
(https://zenodo.org/records/21398895, ZIP em `eqt_videos/`) e extraia os pontos com
`python extrair_minds_v3.py` e `python extrair_eqt_v3.py`. Depois:

| No TCC | Comando |
|---|---|
| Tabelas 2 e 3, versão final e divisão aleatória | `avaliar_tcc.bat` (ou `python avaliar_v3.py --dados minds --recortar --sem-mao-parada --aleatoria --comparar` e o mesmo com `--dados eqt`) |
| Tabelas 2 e 3, sem o preparo | `avaliar_sem_preparo.bat` (`python avaliar_v3.py --dados minds` e `--dados eqt`) |
| Tabela 1 (dados sintéticos) | `python teste_v3.py` |
| CT06 (alfabeto com movimento) | `python teste_soletrador.py --sementes 0,11,22,33,44,55 --tremor 1,2,3 --falhas 0,0.08` |
| CT07 (modo calibrado) | `python teste_calibracao.py` |
| CT08 e servidor | `python servidor/teste_servidor.py` |
| Modelo usado no app | `python treino_v3.py --recortar --sem-mao-parada` |

O restante deste README é o caderno de desenvolvimento: como cada parte funciona e o
histórico das versões.

## 🌐 Versão web (navegador do computador e do celular)

A mesma coisa do `libras_camera.py` (ALFABETO, PALAVRAS e CALIBRADO, com
calibração, conferência e o exemplo da tecla V), aberta por um endereço, sem
instalar nada no aparelho:

- **No navegador:** a página liga a câmera e roda o MediaPipe ali mesmo (mãos e
  corpo, os mesmos modelos e limiares do app). Para o servidor vão **só os pontos**,
  nunca a imagem (o requisito de não enviar imagens continua valendo).
- **No servidor** (`servidor/`, Python/FastAPI): o mesmo código e os mesmos modelos
  do app (`libres_realtime.py`, `libres_v3.py`, `calibracao.py`) reconhecem o sinal e
  devolvem o texto. A calibração de cada pessoa fica no servidor, por perfil.
- **Botões no lugar das teclas** (no computador as teclas também valem: M, ESPAÇO,
  BACKSPACE, K, V, N, P, R, D, H, C, F), layout para celular em pé e deitado, e
  **Falar**: lê o texto em voz alta (LIBRAS → fala).
- **Como se faz?** na calibração, logo abaixo da palavra: abre o vídeo do sinal num
  dicionário de Libras (veja [Não sabe fazer o sinal?](#não-sabe-fazer-o-sinal)); nas
  listas, cada palavra também abre o dicionário.

Conferido sem câmera de verdade: o motor web dá o mesmo texto que o app de
computador com os mesmos quadros; os pontos do MediaPipe no navegador (CPU) são
iguais aos do Python nas mesmas imagens (diferença < 0,0005); calibração,
reconhecimento, queda e volta da internet e o layout de celular testados num
Chromium automático. No aparelho, a página mede a velocidade e escolhe CPU (pontos
idênticos aos do app) ou GPU (mais rápida, diferença de ~0,005, do tamanho do
tremor normal do MediaPipe); para forçar: `?processamento=cpu` ou `gpu` no endereço.

**Câmera:** sem escolha, a página usa a webcam de verdade: pula câmera ocupada,
câmera virtual (OBS, NVIDIA Broadcast...) e câmera que manda imagem parada, preta
ou nenhuma imagem (como o app de computador). Para escolher: **Mais → Câmera…**
(ou a tecla `C`, que passa para a próxima); a escolha fica guardada no navegador.
Se nenhuma abrir, a tela mostra o motivo (ocupada por outro programa, sem
permissão...) e o botão **Escolher câmera**. Não deixe o `libras_camera.py` aberto
ao mesmo tempo: no Windows, uma câmera só abre em um programa por vez.

**Testar no computador** (a câmera funciona em `localhost`):

```bash
pip install -r servidor/requirements.txt
python servidor/preparar_web.py          # 1a vez: baixa o MediaPipe do navegador (~22 MB)
uvicorn servidor.app:app --port 8000     # abra http://localhost:8000
```

**Publicar no Railway** (o navegador do celular só libera a câmera em `https://`):

```bash
python treino_v3.py --recortar --sem-mao-parada   # o modelo que vai para o servidor
python servidor/fixar_versoes.py                  # mesmas versões do scikit-learn/xgboost do seu .venv
npm i -g @railway/cli                             # (ou: scoop install railway)
railway login
railway init                                      # cria o projeto
railway up                                        # sobe a pasta e constrói pelo Dockerfile
railway volume add --mount-path /data             # a calibração não se perde a cada publicação
railway domain                                    # gera o endereço https://....up.railway.app
```

No Windows, `publicar_site.bat` (dois cliques) faz o teste do servidor com os seus
modelos e, se passar, o `railway up`; os registros ficam em `publicar*.log`.

`railway up` não envia o que está no `.gitignore`/`.railwayignore` (vídeos, dados
extraídos, `.venv`); o Dockerfile baixa o MediaPipe do navegador e o modelo do corpo.
Depois de mudar o código ou o modelo, `railway up` de novo. Outra opção é ligar o
serviço do Railway ao repositório do GitHub (branch `main`): cada push publica
sozinho, com os modelos do repositório e as versões de `servidor/requirements-modelos.txt`
(rode `fixar_versoes.py` e faça commit do arquivo). Se a página avisar
"modelos salvos com outras versões", rode `fixar_versoes.py` e publique de novo.
Opcional: a variável `LIBRES_CODIGO` (no painel do Railway) faz a página pedir um
código para entrar. O plano Hobby (US$ 5/mês com US$ 5 de uso) basta.

**Modo pesquisa (teste com usuários do TCC):** o link com `?pesquisa=1`
(ex.: `https://libre-se-web-production.up.railway.app/?pesquisa=1`) abre a página sem pedir
nome, mostra o roteiro do teste (alfabeto: L-I-B-R-A-S; calibrar e reconhecer oi, obrigado,
café, pão e tchau, 3 vezes cada) e **não salva nada** no servidor: os sinais gravados ficam
numa pasta temporária, apagada quando a sessão expira (20 min sem uso).

**Na apresentação:** calibre no mesmo aparelho, câmera e luz da apresentação; no
celular, use um suporte (as duas mãos livres); abra a página antes (a 1a carga baixa
~25 MB); deixe o app de computador como plano B, porque a versão web depende da
internet da sala.

## 📱 App mobile (Android e iOS)

Um app de celular (`app/librese/`, React Native 0.87 com TypeScript) que usa **o mesmo
servidor e a mesma IA da versão web**: os modos ALFABETO, PALAVRAS e CALIBRADO, com
calibração, conferência e o exemplo do sinal. Nada no servidor nem na página muda para
o app funcionar.

```
câmera do celular → MediaPipe no celular → pontos (mãos + corpo) → WebSocket /ws → servidor (motor.py) → texto
```

- **No celular:** o módulo nativo `app/librese/modules/react-native-libres-landmarker`
  (Swift no iOS, Kotlin no Android) roda o MediaPipe Tasks 0.10.33 em cada quadro da
  câmera (VisionCamera v5), com os mesmos ajustes da página: modo IMAGE, confiança 0,5,
  imagem espelhada, no máximo 640 px de largura e o corpo a cada 0,2 s. Os modelos
  `hand_landmarker.task` e `pose_landmarker_lite.task` vão dentro do app. Como na web,
  **só os pontos** vão para o servidor, nunca a imagem.
- **Conversa com o servidor:** `src/servidor/ClienteLibres.ts` fala o mesmo protocolo da
  página (`web/app.js`): manda os pontos em lotes a cada 80 ms, envia os comandos (modo,
  limpar, apagar, calibração) e reconecta sozinho se a internet cair. Cada celular é uma
  sessão no servidor, todos com o perfil `app` (`USUARIO_PADRAO` em `src/config.ts`): a
  calibração feita num celular vale para os outros.
- **Tela ao vivo** (`src/screens/AoVivoScreen.tsx`, botão TRADUZIR da tela inicial): a
  câmera em tela cheia, os modos no topo, a letra do momento com a confiança, o texto
  traduzido, as 3 opções mais prováveis e os botões Apagar e Limpar. No modo CALIBRADO,
  mostra a palavra da vez, quantos exemplos já foram feitos e os botões da calibração
  (anterior, próxima, ver como faz, desfazer, refazer e terminar). O topo mostra
  **"conectado · N ms"**, o tempo que o MediaPipe leva em cada quadro.
- **Servidor:** `src/config.ts` aponta para o Railway
  (`https://libre-se-web-production.up.railway.app`). Para testar com o servidor no
  computador, troque por `http://<IP do computador>:8000` (veja abaixo).

**Estado:** compila no Android (debug e release) e no iOS. Na etapa final, a equipe instalou o
app em celulares Android e iPhone e confirmou que a câmera, a detecção das mãos e o
reconhecimento funcionam com o mesmo servidor; no Android, a detecção usa a GPU. Ainda não
houve uma avaliação sistemática com usuários (a pesquisa do TCC usou a versão web).

**Gerar o APK (Android)** — node 22 ou mais novo, Android SDK e JDK 17:

```bash
cd app/librese
npm install
cd android && ./gradlew assembleRelease
# o APK sai em android/app/build/outputs/apk/release/app-release.apk
```

O APK é assinado com a chave de debug (dá para instalar direto no celular, não serve para a
Play Store) e não precisa do computador ligado. Para rodar com o celular no cabo e
recarregar o código na hora: `npx react-native run-android` (com o `npx react-native start`
aberto).

**iPhone** — Xcode 27 e CocoaPods:

```bash
cd app/librese
npm install
cd ios && pod install
```

Abra `ios/librese.xcworkspace` no Xcode, escolha o seu Apple ID em *Signing & Capabilities*
(a conta gratuita serve) e rode com o iPhone no cabo. O `Podfile` já sobe as bibliotecas para
o iOS 15 (o mínimo do Xcode 27) e corrige o link do MediaPipe, que vem com o `Info.plist`
apontando para `.a` em vez de `.framework`.

**Com o servidor no computador** (celular e computador na mesma rede):

```bash
uvicorn servidor.app:app --host 0.0.0.0 --port 8000   # --host 0.0.0.0: o celular precisa achar o computador
ipconfig getifaddr en0                                 # o IP do computador (Mac)
```

Troque `SERVIDOR_URL` em `app/librese/src/config.ts` por `http://<IP>:8000` e gere o app de
novo. Se o servidor pedir código (`LIBRES_CODIGO`), coloque-o em `CODIGO_ACESSO` no mesmo
arquivo.

**Se der problema:** app fechando no Android, com o celular no cabo:
`adb logcat -b crash`. Sem conexão, o topo da tela diz o motivo (sem conexão, servidor
cheio, código de acesso errado).

As telas antigas de foto e vídeo (`CameraScreen.tsx` e `TranslationScreen.tsx`) continuam
no código, mas a tela inicial não leva mais a elas, e a tradução delas ainda é simulada.

## 🖥️ Usando o app (`libras_camera.py`)

A janela tem a câmera (só com os pontos da mão) e, ao lado, um **painel escuro**
com o texto reconhecido, a letra/palavra do momento, as 3 opções mais
prováveis, os sinais que o modelo conhece e o histórico. O painel não depende
da luz da sala, e tudo o que entra no texto também vai para `saida_libras.txt`.

| Tecla | Ação |
|---|---|
| `M` | troca o modo: ALFABETO → PALAVRAS → CALIBRADO |
| `ESPAÇO` | limpa o texto |
| `BACKSPACE` | apaga o último item |
| `S` | salva os últimos 6 s de pontos da mão em `debug/` (sem imagem) |
| `H` | **como se faz?** abre no navegador o vídeo do sinal num dicionário de Libras: na calibração, a palavra da vez; fora dela, o último sinal; no ALFABETO, o alfabeto manual |
| `C` | troca de câmera (a próxima: 0, 1, 2... e volta) |
| `[` / `]` | **luz da câmera**: escurece / clareia (exposição do sensor) |
| `E` | volta a exposição para o automático |
| `Q` | sai |

**Qual câmera:** sem opção, o app testa as câmeras e **pula as de imagem preta
ou parada** — a câmera virtual do OBS desligada (logo do OBS), NVIDIA Broadcast,
DroidCam sem o celular. Câmera de verdade sempre tem ruído do sensor, então dois
quadros nunca saem idênticos; imagem idêntica por ~1 s é câmera virtual. Para
escolher uma câmera: `python libras_camera.py --camera 1` (vale também para
`coletar_palavras.py` e `coletar_alfabeto.py`). Se a imagem ficar parada ou preta
durante o uso, aparece um aviso na tela: aperte `C`.

**Sala muito clara / pele muito clara:** com a imagem estourada (branca), a mão
perde o contorno e o MediaPipe deixa de achá-la. Filtros na imagem (gama, CLAHE,
contraste) **não resolvem**: testamos em 283 fotos de mãos com luz forte simulada e
a detecção caiu (ex.: 68,9% sem filtro → 43,5% com gama), porque a parte branca já
não tem informação para recuperar. O que funciona é **a câmera captar menos luz**:
no app, `[` escurece (e `E`, ou voltar com `]` até o início, volta ao automático); na página,
**Mais → Luz da câmera** (tecla `L`): o meio da barra é o normal, à esquerda escurece. As duas avisam quando a imagem fica clara demais. Só funciona em câmeras
que deixam mudar a exposição (a maioria das webcams USB e o Chrome no Android);
nas outras, ajuda: fundo e roupa escuros, luz de frente (não atrás da pessoa) e
afastar da janela.

## 🎯 Modo CALIBRADO (palavras do dia a dia)

O modelo geral (MINDS) conhece só 20 palavras e, com uma pessoa que ele nunca
viu, acerta ~88%. Um sinal fora da lista (ex.: CIRCO) sempre vira a palavra mais
parecida (MAÇÃ). Palavras do dia a dia — oi, bom dia, obrigado, números — não
estão em nenhuma base pública com várias pessoas. O modo **CALIBRADO** resolve
os dois problemas com exemplos da **própria pessoa**:

1. `python libras_camera.py --usuario julia` e `M` até **CALIBRADO**.
2. Na primeira vez o app pede cada palavra de `vocabulario.txt` 5 vezes
   ("Faça o sinal: BOM DIA — 2 de 5"). Faça o sinal e **abaixe a mão** (tire do
   quadro); o sinal só termina quando a mão sai. Números: mostre a mão parada
   ~1 s e abaixe. São salvos só os pontos (`calibracao/julia/`), nenhuma imagem.
3. No fim, o app confere a calibração (cada exemplo contra os outros) e avisa
   no terminal e no painel: palavras **parecidas** entre si ("boa tarde ~ boa
   noite") e exemplos **estranhos** — refaça essas com `R`.
4. Depois, cada sinal é comparado com os exemplos (DTW: alinha as duas
   sequências no tempo, então fazer mais rápido ou mais devagar não atrapalha).
   Se não parecer com nenhum, aparece **NÃO RECONHECIDO** em vez de um chute.

| Tecla (modo CALIBRADO) | Ação |
|---|---|
| `K` | começa / termina a calibração (retoma das palavras que faltam) |
| `V` | mostra, num quadro no canto da imagem, **como o sinal foi feito na calibração** (o exemplo mais típico da palavra: mãos e corpo em movimento, como no espelho). Calibrando: o da palavra da vez — se você ainda não gravou, o do perfil **referencia** (abaixo); `V` de novo alterna entre o seu e o da referência. Reconhecendo: o do último sinal (ou do mais parecido, se não foi reconhecido); `V` de novo passa para a próxima palavra |
| `H` | **como se faz?** abre o vídeo da palavra da vez num dicionário de Libras |
| `N` / `P` | próxima / anterior palavra (durante a calibração) |
| `R` | refaz a palavra atual (apaga os exemplos dela) |
| `D` | desfaz o último exemplo (se saiu errado) |

Para ensaiar: antes de cada sinal, `V` mostra como você fez na calibração —
repetir do mesmo jeito é o que faz o app acertar. Quem calibra primeiro também
pode usar o `V` para mostrar os sinais para quem vai apresentar.

### Não sabe fazer o sinal?

A calibração **não ensina** o sinal: ela grava como **você** faz, e o app passa a
reconhecer aquele jeito. Então, antes de gravar uma palavra que você não conhece:

1. **Veja o sinal num dicionário de Libras.** `H` (na web, o botão **Como se faz?**
   abaixo da palavra, ou toque na palavra da lista) abre a busca da palavra no
   [Spread the Sign](https://spreadthesign.com/pt.br/search/) em português do Brasil,
   com vídeos dos sinais em Libras; números são procurados por extenso (2 → "dois").
   Se não achar, o [Dicionário da Libras do INES](https://dicionario.ines.gov.br/)
   (link ao lado na web) tem vídeos, exemplos e acepções. A Libras tem **variação
   regional**: o vídeo pode mostrar outra variante — para o app, o que importa é fazer
   sempre do mesmo jeito que na calibração.
2. **Ou use o perfil de referência.** Alguém que conhece os sinais (uma pessoa surda,
   intérprete, professor ou o colega que já aprendeu) calibra com o perfil
   `referencia` (`python libras_camera.py --usuario referencia`, ou o perfil
   "referencia" na web). Daí em diante, quem ainda não gravou uma palavra vê com `V`
   **como a referência faz** — o painel avisa ("V = como a referência faz"). Na
   calibração, `V` de novo alterna entre o seu exemplo e o da referência. Outro nome:
   `--referencia NOME` no app ou a variável `LIBRES_REFERENCIA` no servidor. O app de
   computador guarda em `calibracao/referencia/` e a web em `dados_web/calibracao/`
   (no Railway, no volume): copie a pasta de um para o outro ou grave pela própria
   página. Na web, qualquer um que escolher o perfil "referencia" pode mexer nos
   exemplos dele — com a página aberta ao público, use o `LIBRES_CODIGO`.
3. **Pule a palavra** que ainda não sabe (`N`, ou **Próxima**) e volte nela depois
   (`K` retoma as que faltam), ou tire-a do vocabulário.

Dicas: calibre **no lugar e com a câmera da apresentação** (luz, distância,
mesmo fps — o painel avisa se a câmera ficar bem mais lenta que na calibração);
faça os sinais do jeito que vai fazer na hora; cada pessoa usa o seu `--usuario`.
Para mudar o vocabulário, edite `vocabulario.txt` (uma palavra por linha) e
aperte `K`: o app pede só as palavras novas.

Teste sem câmera (`python teste_calibracao.py`): pessoas sintéticas com sinais
parados (números), com troca de forma (oi), compostos que começam igual (bom
dia / boa tarde / boa noite) e de duas mãos. Com 5 exemplos por palavra: 98,9%
de acerto, nenhum sinal trocado (os erros viram "não reconhecido"); 100% dos
sinais fora da calibração recusados; e, com a calibração de **outra** pessoa,
o app recusa quase tudo — por isso cada pessoa calibra a sua. São sinais
sintéticos: o número real sai do teste com a webcam.

**Letras com movimento (H, J, K, X, Z):** faça o movimento e, no fim, pare a
mão ou abaixe; a letra entra quando o movimento termina. Enquanto a mão se
move, o painel mostra "movimento...".

Numa letra com movimento a **mão inteira** se move e a **forma não muda** (o I
do J, o gancho do X); numa troca de letra (O → L, K → I) é a forma que muda.
Por isso o app procura, dentro do gesto, o maior trecho com a forma constante
(distâncias 3D entre pontas dos dedos, punho e polegar, que não mudam quando a
mão gira ou se afasta) e só mede o movimento ali. Assim a troca para a letra
(A → gancho → X) ou para a próxima (J → U) não atrapalha, e transições não
viram letras falsas.

| Letra | Forma da mão | Movimento |
|---|---|---|
| J | só o mínimo (I) | o punho anda (desenha o J); o I que entrou antes vira J |
| Z | indicador esticado | zigue-zague: muda de direção 2 vezes e desce |
| X | indicador esticado ou em gancho | para o lado, sacudido ou para trás (a mão diminui na imagem) |
| K | indicador e médio levantados | sobe e desce |
| H | indicador e médio (U, V ou K) | gira a mão (a palma vira 40° ou mais) ou vai para o lado |
| Y | regra de geometria: polegar e mínimo esticados, demais dobrados | — |
| U × R × V | regra de geometria: distância entre as pontas do indicador e do médio | — |

Movimento lento também vale (um J pequeno, ~1 mão/s, se durar 0,25 s), e a
letra parada que entrou antes (o I do J) é trocada mesmo se você segurou a
forma por alguns segundos antes de mexer.

Para repetir a mesma letra (OSSO), tire a mão do quadro entre as duas.

Com a mão parada, o app usa a média dos últimos 0,2 s (probabilidades e
pontos da mão): o tremor do MediaPipe deixa de trocar a letra quadro a quadro.

Teste sem câmera (`python teste_soletrador.py --sementes 0,11,22,33,44,55 --tremor 1,2,3 --falhas 0,0.08`):
27 sequências montadas com poses reais dos datasets (UVA, RUA, BOLA, SOL, LIVRE,
OSSO, JUIZ, HOJE, KIWI, AXE, J normal/pequeno e lento/depois de 2 s de I, Z, X para
o lado/sacudido/para trás/em gancho, H de lado/girando, K, Y...), transições suaves
entre letras, movimento simulado, tremor de 1 a 3 px, a 30 e a 15 fps.
Resultado: **972/972 (100%)** sem falhas de detecção e **894/972 (92,0%)** com 8% dos
quadros sem mão, mais rajadas de 2 a 4 quadros perdidos no meio dos gestos (98,4% a
30 fps e 85,6% a 15 fps). O que sobra de erro com falhas é quase sempre um gesto que
ficou com 1 ou 2 quadros visíveis (ex.: o último traço do Z inteiro sem mão).
A 10 fps, sem falhas, foram 158/162 (97,5%; só o O depois do H, em HOJE, se perde).
Com a profundidade (z) dos pontos reduzida à metade (`--z 0.5`: o MediaPipe numa
câmera real erra mais a profundidade, o que afeta a mão girando), 322/324 (99,4%).
Para as letras com movimento, prefira a câmera a 30 fps e boa luz (menos borrão,
menos falhas de detecção); o painel mostra o fps.

**Alguma letra ou palavra falhou?** Logo depois de tentar, aperte `S`. O
trecho salvo pode ser analisado sem câmera:

```bash
python reproduzir_trecho.py debug/trecho_AAAAMMDD_HHMMSS.npz            # resumo
python reproduzir_trecho.py debug/trecho_AAAAMMDD_HHMMSS.npz --quadros  # quadro a quadro
```

## 📚 Mais palavras: LIBRAS-EQT-UECE (45 sinais isolados, 5 pessoas)

Base aberta da UECE (2026), licença **CC BY-NC 4.0** (pesquisa e ensino; sem uso
comercial). Não suba os vídeos nem os pontos extraídos para o GitHub (`eqt_videos/`
e `eqt_v3/` já estão no `.gitignore`).

```bash
# 1. baixe "Libras-EQT-UECE (RGB).zip" (9 GB) em https://zenodo.org/records/21398895
#    e coloque o .zip (sem extrair) na pasta eqt_videos/
python extrair_eqt_v3.py --listar          # confere sinais, informantes e o que fica de fora
python extrair_eqt_v3.py --teste 3         # 3 vídeos, para conferir os números
python extrair_eqt_v3.py                   # todos (pode parar e continuar depois)
python avaliar_v3.py --dados eqt           # uma pessoa fora do treino, 5 rodadas
python avaliar_v3.py --dados minds,eqt     # as duas bases juntas (acerto de cada uma)
python treino_v3.py --recortar --sem-mao-parada   # modelo do app (as opções vão junto no modelo)
```

**Primeiro resultado (out/2026):** 58,6% com uma pessoa fora do treino (45 sinais,
5 informantes; o acaso daria 2,2%), de 41,8% (informante 5) a 67,8% (informante 2).
No MINDS são 88% com 20 sinais e 11 pessoas. Com a primeira versão do
`--sem-mao-parada`, a EQT foi a 59,9% (informante 5: 45,1%): a regra só pegou 8%
dos sinais da informante 5 (ver abaixo), e parte da diferença era sorteio (ver
"Comparação justa"). Para entender a diferença e medir as correções:

```bash
python diagnostico_v3.py --dados minds,eqt          # ~1 min: o que muda de uma pessoa (e base) para outra
python avaliar_v3.py --dados eqt --recortar --sem-mao-parada --antes-depois    # i05 (~1 h)
python avaliar_v3.py --dados minds --recortar --sem-mao-parada --antes-depois  # s04, s11, s12 (~1 h)
```

**Com as duas opções (v13, nov/2026)**, pessoa fora do treino, mesmo sorteio nas
duas colunas:

| | sem opções | `--recortar --sem-mao-parada` |
|---|---|---|
| **LIBRAS-EQT-UECE** (45 sinais, 5 pessoas) | 58,9% | **72,3%** (+13,4) |
| pior e melhor pessoa | 41,5% (i05) e 68,5% (i02) | 67,1% (i04) e 76,4% (i05) |
| **MINDS-Libras** (20 sinais, 11 pessoas) | 88,6% | **91,5%** (+2,9) |
| pior e melhor pessoa | 65,3% (s04) e 99,0% (s05) | 67,4% (s04) e 99,0% (s01, s07) |

Na EQT, todas as informantes melhoraram: i05 +34,9 (de pior para melhor), i03
+13,0, i04 +8,6, i02 +6,9, i01 +3,6; as que quase não mudaram também ganharam,
porque treinam com os sinais mais limpos das outras. Palavras: já 46,7 → 96,7%,
trinta 46,7 → 93,3%, vencimento e comum 60 → 100%. No MINDS: s12 +9, s10 +8,
s11 +3; s02 −1 e s05 −2 (1 e 2 sinais). Palavras: esquina 82 → 94%, filho
66,7 → 78,4%. As regras foram desenhadas olhando o diagnóstico destas mesmas
pessoas (os limites não foram ajustados pelo acerto); vale confirmar com
gravações de pessoas novas.

**O que o diagnóstico mostrou (out/2026):** as gravações das 5 informantes são
parecidas (câmera 16:9, corpo detectado em 100% dos quadros, mesma distância). A
diferença está nas mãos: a informante 5 deixa as duas mãos à vista o tempo todo
(duas mãos em 94% dos quadros; as outras, 38–70%), descansando na altura da
barriga (0,9 largura de ombro abaixo dos ombros), em vez de abaixar a que não
participa do sinal. No MINDS, s11 e s12 deixam os braços pendurados à vista
(1,6 abaixo dos ombros). Para o modelo, um sinal de uma mão com a outra mão
parada à vista parece um sinal de duas mãos.

- `--sem-mao-parada`: medido no **meio** do sinal (sem os 20% do começo e do
  fim, quando quem descansa as mãos juntas ainda as encosta), a outra mão conta
  como ausente quando (a) **anda menos de 0,35** largura de ombro, **não toca** a
  dominante e fica **abaixo do peito** (punho mais de 0,6 largura de ombro abaixo
  dos ombros: barriga, colo, braço pendurado), ou (b) o vídeo começa com ela
  descansando embaixo e, no meio do sinal, ela continua no mesmo lugar sem ser
  tocada (mãos juntas na frente da barriga). A mão de apoio dos sinais de duas
  mãos (no espaço do sinal, ou tocada pela dominante) continua. A primeira
  versão exigia que as mãos **nunca** chegassem perto no vídeo inteiro e por
  isso quase não pegava a informante 5.
- `--recortar`: além da mão baixa (mais de 1 largura de ombro abaixo dos
  ombros) e parada, corta nas pontas o tempo em que as mãos ainda (ou de novo)
  estão paradas onde começaram, se começaram abaixo do peito (as mãos da
  informante 5 descansam a 0,9, e a regra antiga não cortava).
- `--antes-depois`: roda sem e com as opções e mostra a diferença pessoa por
  pessoa e as palavras que mais mudaram.

**Comparação justa:** o treino usa 8 variações sorteadas de cada sinal (ruído,
tamanho, velocidade, espelhamento). Até a v12, um sorteio único servia a todos
os sinais em sequência: mudar alguns sinais de uma pessoa deslocava o sorteio de
todos os seguintes, e a comparação com e sem uma opção misturava o efeito da
opção com o acaso (no 59,9%, as informantes que a regra quase não mudou também
variaram de −0,7 a +1,8 ponto). Agora cada sinal tem o seu próprio sorteio
(semente + nome do arquivo): o que não muda tem as mesmas variações nas duas
rodadas, e a diferença vem só dos sinais que as opções mudaram. Por isso o
"sem opções" do `--antes-depois` pode ficar alguns décimos diferente dos 58,6% e
88,0% medidos antes; no TCC, compare os dois números da mesma rodada.

Testes sintéticos (com o sorteio por sinal): braço pendurado à vista, 53,3% →
96,7% (as outras pessoas iguais); mão parada nas pontas, 96,7% → 100%; pessoas
sem nada disso, 100% com ou sem as opções. Com mãos juntas na barriga, as
opções pegam os sinais certos (o diagnóstico mostra) mas o acerto sintético não
mudou, nem para melhor nem para pior: lá, a subida e a descida da mão também
diferem das outras pessoas. Só os vídeos reais dizem.

O `diagnostico_v3.py` lê só os `.npz` e mostra, por pessoa: orientação da imagem,
detecção de mão e de corpo, mão já no quadro no começo/fim do vídeo, tempo
parado, tamanho da mão em relação aos ombros, quanto o `--recortar` cortaria e,
numa segunda tabela, como fica a **outra mão** (visível, parada, altura, longe
da dominante), em quantos sinais o `--sem-mao-parada` a ignoraria e por qual
regra, e duas medianas (`folga`, `afastou`) que mostram qual limite ajustar se
ela ainda escapar.
O `--recortar` tira do começo e do fim de cada sinal os quadros com a mão
**baixa** (mais de 1 largura de ombro abaixo dos ombros) **e parada**; a mão
entrando no quadro e um número segurado na altura do peito continuam. Num
teste sintético com repouso de 0,2 a 2 s por pessoa, o LOSO foi de 96,7% para
100%, e os sinais sem repouso (como os do MINDS) ficaram iguais. Treinando com
`python treino_v3.py --recortar` (ou `--sem-mao-parada`), a opção vai junto no
modelo e a câmera usa o mesmo preparo.

O pacote RGB traz também frases (parte contínua); o extrator usa só os sinais
isolados: pastas com **uma palavra** ("Energia.", "Luz.", "e-mail") ou com a glosa
em **maiúsculas** ("ANTERIOR", "SOMENTE ALGUNS CÔMODOS") — 45 sinais, 5
informantes, ~30 vídeos cada. O resto ("Eu quero pagar a conta.", "Casa toda.",
"Dia 15.") fica de fora. O `--listar` mostra a decisão; corrija com `--incluir` /
`--excluir`.
`treino_v3.py` e `avaliar_v3.py` leem todas as bases que existirem
(`minds_v3/`, `eqt_v3/`, `coleta_v3/`); `--dados` escolhe quais. Cada pessoa
vira um id: `s01`… (MINDS), `i01`… (EQT), `p01`… (coleta própria).

## 🔀 Palavras que se confundem (ex.: RAIO × CINCO)

O modo PALAVRAS escolhe, entre todas as palavras que o modelo aprendeu (as 20 do
MINDS e as 45 da EQT), a mais parecida com o sinal. Sinais com a mesma forma de mão
no mesmo lugar (mão aberta na frente do peito) só se separam pelo movimento: se ele
sai pequeno, rápido demais para a câmera ou diferente do das bases, ganha a vizinha.
E, ao vivo, você é uma pessoa (e uma câmera) que o modelo nunca viu.

1. **Veja o que o modelo aprendeu e compare com o seu sinal** (`ver_sinais.py`):
   ```bash
   python ver_sinais.py --listar                        # os nomes das palavras em cada base
   python ver_sinais.py raio cinco digital              # os sinais das bases lado a lado + o que muda
   # no app, logo depois do erro, aperte S; depois:
   python ver_sinais.py raio cinco --trecho debug/trecho_AAAAMMDD_HHMMSS.npz
   python ver_sinais.py raio cinco digital --figura figura_raio_cinco.png   # figura para o TCC
   ```
   A janela mostra o exemplo mais típico de cada palavra, só com os pontos e espelhado
   como no app (`N`/`P`: outras pessoas). Com `--trecho`, mostra também o **seu** sinal
   e o que o modelo achou dele. No terminal, uma tabela mostra o que muda entre as
   palavras: quanto o punho anda, a altura da mão, os dedos esticados, uma ou duas
   mãos. Os vídeos originais estão no .zip da EQT, na pasta com o nome da palavra.
2. **Meça quais palavras o modelo separa** (pessoa fora do treino):
   `python avaliar_v3.py --dados minds,eqt --recortar --sem-mao-parada`. No fim sai o
   acerto de cada palavra, com quem ela mais se confunde, e a lista das que passam de
   85% (`--meta`).
3. **Treine o modelo do app só com as palavras que vai mostrar** (menos vizinhas para
   confundir; use os nomes do `--listar`, separados por vírgula):
   ```bash
   python avaliar_v3.py --recortar --sem-mao-parada --palavras raio,digital,...   # o número do TCC
   python treino_v3.py --recortar --sem-mao-parada --palavras raio,digital,...    # o modelo do app
   ```
   Se precisar das duas palavras que se confundem, mantenha as duas e capriche no que
   as diferencia (o `ver_sinais.py` mostra): o movimento inteiro, do tamanho do das
   bases, com boa luz (mão nítida, câmera a 30 fps).
4. **Na apresentação**, o que tem de funcionar sempre vai para o modo CALIBRADO (os
   exemplos da própria pessoa; sinal que não parece com nenhum vira "não reconhecido").
5. **A longo prazo**, mais pessoas gravando as palavras confundidas
   (`coletar_palavras.py`) é o que mais melhora o modelo geral.

## ⚡ v3 — palavras com as duas mãos e o corpo (outubro/2026)

A v2 olha **uma mão** e mede o movimento em relação a ela mesma. Ficam de fora
a **outra mão** (América, esquina, acontecer, sapo e banheiro usam as duas) e o
**lugar** do sinal (o mesmo movimento perto do rosto ou do peito). A v3 guarda,
por quadro, as duas mãos (MediaPipe Hand Landmarker com `num_hands=2`) e o corpo
(MediaPipe Pose: nariz e ombros):

- a mão **dominante** é a que mais se move no sinal; cada mão fica sempre no
  mesmo "lugar" de um quadro para o outro (o MediaPipe devolve as mãos em
  qualquer ordem e às vezes troca o rótulo Left/Right);
- a posição das mãos é medida em **larguras de ombro** a partir do centro dos
  ombros, e a ponta do indicador em relação ao nariz: não depende da distância
  da câmera nem do formato da imagem (16:9 ou 4:3);
- 1341 features por sinal (as da v2 para cada mão + trajetórias em relação ao
  corpo e entre as mãos), XGBoost e avaliação deixando **um sinalizador inteiro
  fora do treino** (LOSO), com os 12 sinalizadores do MINDS.

Ordem para rodar (ZIPs dos 12 sinalizadores em `minds_videos/`):

```bash
python teste_v3.py                    # 5-15 min: confere todo o caminho com sinais sintéticos
python extrair_minds_v3.py --teste 3  # confere a extração em 3 vídeos
python extrair_minds_v3.py            # todos os vídeos: 30 min a algumas horas (pode parar e continuar)
python avaliar_v3.py --comparar       # número do TCC: pessoa fora do treino, v2 x v3 (15-40 min)
python treino_v3.py --recortar --sem-mao-parada   # modelo final -> modelo_palavras_v3.pkl (~5 min)
python libras_camera.py               # o modo PALAVRAS passa a usar a v3
```

O `teste_v3.py` usa sinais sintéticos montados para que a v2 não consiga
separar dois pares (mesmo movimento perto do rosto ou do peito; com ou sem a
outra mão): ele confere que a v3 separa (no teste: v2 66% x v3 100%) e que a
câmera recorta e reconhece os sinais quadro a quadro. Esses números conferem o
código; o número do TCC é o do `avaliar_v3.py` com os vídeos reais.

O `treino_v3.py` usa as mesmas 300 árvores do `avaliar_v3.py` (o acerto medido vale para o
modelo da câmera) e mostra o andamento a cada 25 árvores. No terminal, Ctrl+C interrompe o
programa: não use para copiar texto enquanto ele treina. `--sem-teste` pula a conferência
com vídeos fora do treino e vai direto ao modelo final.

Na primeira vez, `extrair_minds_v3.py` baixa o modelo de pose do MediaPipe
(`pose_landmarker_lite.task`, ~6 MB). O `avaliar_v3.py` lista as palavras que
passaram de 85% e mostra o comando para um vocabulário só com elas
(`--palavras ...`, que também vale para o `treino_v3.py`). Sem
`modelo_palavras_v3.pkl`, a câmera continua com a v2. Na câmera, a v3 desenha as
duas mãos (a segunda em azul) e a tecla `S` também guarda as duas mãos e a pose.

**Resultado com os vídeos reais** (`avaliar_v3.py --comparar`, 11 sinalizantes do
MINDS, 1.072 sinais, 20 palavras, pessoa fora do treino):

| | Acurácia (LOSO) | Pior e melhor sinalizante |
|---|---|---|
| v2 (uma mão) | 85,5% | 60,0% e 97,0% |
| **v3 (duas mãos + corpo)** | **88,0%** | 62,1% e 98,0% |
| v3, sorteio por sinal (nov/2026) | 88,6% | 65,3% e 99,0% |
| **v3 + `--recortar --sem-mao-parada`** | **91,5%** | 67,4% e 99,0% |

As duas últimas linhas são do `--antes-depois` (mesmo sorteio nas duas; ver
"Comparação justa" na seção da LIBRAS-EQT-UECE).

A v3 foi igual ou melhor que a v2 em todos os 11 sinalizantes. Com 3
sinalizantes, a v2 tinha 72,7% nas mesmas 20 palavras: mais pessoas no treino
foi o que mais ajudou, e as duas mãos + corpo somaram mais 2,5 pontos. 15 das 20
palavras passaram de 85%; as mais difíceis são filho (72,5%), ruim (80,0%) e
maçã (81,8%).

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

### Alfabeto com as mãos do grupo

O modelo de alfabeto original foi treinado com um único vídeo por letra e
reconhece mal outras mãos. Para treinar com o grupo (≈5 min por pessoa):

```bash
python coletar_alfabeto.py --pessoa 1     # cada pessoa com seu número
python train_alfabeto.py                  # usa coleta_alfabeto/ (+ palavras2/, se existir)
```

Com 2 ou mais pessoas, o treino mostra a acurácia testando em quem ficou fora.

Sem poder gravar? Importe imagens de datasets públicos (pasta por letra ou
export YOLO do Roboflow) e treine:

```bash
python importar_imagens_alfabeto.py --pasta CAMINHO --nome roboflow
python train_alfabeto.py
```

### Coletar palavras pela webcam (dataset próprio, sem MINDS)

```bash
python coletar_palavras.py --sinalizador 1
python coletar_palavras.py --sinalizador 2 --palavras oi,obrigado,tchau --takes 15
```

Cada pessoa usa sempre o mesmo número de sinalizador. Faça o sinal e abaixe
a mão: o take é salvo sozinho em dois formatos, `coleta_v3/<palavra>/PessoaNN_<palavra>_NNN.npz`
(duas mãos + corpo, lido pelo `treino_v3.py` e pelo `avaliar_v3.py`) e
`coletados/<palavra>/SinalizadorNN_<palavra>_NNN.npy` (só a mão principal, para a v2).
Na v3 o nome usa "Pessoa" para não se misturar com os sinalizadores do MINDS.
Pastas de computadores diferentes podem ser juntadas (os nomes não colidem).
Depois: `python avaliar_v3.py` e `python treino_v3.py` (ou, na v2,
`python train_rotulado.py` e `python eval_loso.py`).

**Ordem para rodar depois de atualizar** (o modelo antigo não é compatível):

```bash
python integrate_minds.py      # regera coletados/ a partir do CSV do MINDS
python train_rotulado.py       # treina o modelo de palavras v2
python eval_loso.py            # acurácia por sinalizante (métrica do TCC)
python libras_camera.py        # câmera: alfabeto + palavras
```

Na câmera, modo PALAVRAS: comece com a mão fora do quadro, faça o sinal e
abaixe a mão.

## Dados e licenças

- **MINDS-Libras** (Rezende, Almeida e Guimarães, 2021) e **LIBRAS-EQT-UECE** (Moraes et al.,
  2026, CC BY-NC 4.0): os vídeos e os pontos extraídos não fazem parte do projeto; vão só os
  modelos treinados.
- A calibração de cada pessoa (`calibracao/`, `dados_web/`) é dado pessoal e fica fora do
  projeto. No modo pesquisa, nada é guardado no servidor.
- Código disponível com os autores.
