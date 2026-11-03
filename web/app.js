// Libre-se web: camera + MediaPipe no navegador; o reconhecimento roda no
// servidor (servidor/app.py), com o mesmo codigo e os mesmos modelos do app de
// computador. Para o servidor vao SO os pontos das maos e do corpo, nunca a imagem.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "./vendor/tasks-vision/vision_bundle.mjs";

const PARAMS = new URLSearchParams(location.search);
const TESTE = PARAMS.has("teste");               // ganchos para os testes automaticos
const SEM_CAMERA = TESTE && PARAMS.has("semcamera");
// processamento do MediaPipe: "cpu" da pontos identicos aos do app de computador (Python);
// "gpu" e mais rapida em celular. "auto" (padrao): comeca na CPU e, se ficar lenta, testa a GPU.
const PROCESSAMENTO = (PARAMS.get("processamento") || "").toLowerCase();
const MS_LENTO = 45;                             // ms por quadro na CPU acima disso: testa a GPU
const LARGURA_DETECCAO = 640;                    // a mesma largura da extracao dos videos de treino
const POSE_CADA = 0.2;                           // s: o corpo muda devagar (como o DetectorV3)
const ENVIO_MS = 80;                             // manda os quadros em lotes, ~12 por segundo
const MAX_EM_VOO = 3;
const MAX_FILA = 240;                            // ~8 s de quadros guardados se a conexao cair
const LIGACOES_MAO = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const LIGACOES_CORPO = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
const VOLTAS_EXEMPLO = 2, PAUSA_EXEMPLO = 0.6;
const COR = { ok: "#5adc5a", mao1: "#00ff00", mao2: "#00c8ff", cal: "#be6ec8", grav: "#e63c3c",
  painel: "rgba(26,28,32,0.9)", texto: "#f0f0f0", suave: "#9aa0a8", linha: "#34383f", corpo: "#969696" };
const DICAS = {
  ALFABETO: "Letra parada entra depois de ~0,3 s estável. H, J, K, X e Z têm movimento: faça o movimento e, " +
    "no fim, pare a mão ou abaixe. Para repetir a mesma letra (OSSO), tire a mão do quadro entre as duas.",
  PALAVRAS: "Comece com a mão fora do quadro (ou abaixada), faça o sinal e abaixe a mão. Nos sinais de uma mão, " +
    "deixe a outra abaixada.",
  CALIBRADO: "Reconhece os sinais do vocabulário comparando com os exemplos gravados por você. Faça o sinal " +
    "e abaixe a mão. Não sabe um sinal? “Como se faz?” (H) ou a palavra na lista abrem o vídeo no dicionário de Libras.",
};
// "Como se faz?": a calibracao grava como VOCE faz o sinal; quem ainda nao sabe aprende antes num
// dicionario de Libras (o mesmo do app de computador: calibracao.py, link_dicionario)
const DICIONARIO_BUSCA = "https://spreadthesign.com/pt.br/search/?q=";   // Spread the Sign, Libras (pt.br)
const DICIONARIO_INES = "https://dicionario.ines.gov.br/";              // Dicionario da Libras do INES
const DICIONARIO_ALFABETO = "https://spreadthesign.com/pt.br/alphabet/";
const NUMEROS_EXTENSO = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"];
function termoDicionario(p) {                         // "BOM_DIA" -> "bom dia"; "2" -> "dois"
  const t = String(p).replace(/_/g, " ").trim().replace(/\s+/g, " ").toLowerCase();
  return /^\d+$/.test(t) && +t < NUMEROS_EXTENSO.length ? NUMEROS_EXTENSO[+t] : t;
}
const linkDicionario = (p) => DICIONARIO_BUSCA + encodeURIComponent(termoDicionario(p));

const $ = (s) => document.querySelector(s);
const ui = {
  tela: $("#tela"), carregando: $("#carregando"), carTitulo: $("#carregando-titulo"),
  carDetalhe: $("#carregando-detalhe"), tentar: $("#tentar"), faixa: $("#faixa"),
  fps: $("#fps"), conexao: $("#conexao"), perfil: $("#perfil"), modos: $("#modos"),
  texto: $("#texto"), entrou: $("#entrou"), agora: $("#agora"), lista: $("#lista"),
  historico: $("#historico ol"), aviso: $("#aviso"), acoes: $("#acoes"),
  dlgPerfil: $("#dlg-perfil"), campoUsuario: $("#campo-usuario"), blocoCodigo: $("#bloco-codigo"),
  campoCodigo: $("#campo-codigo"), dlgVocab: $("#dlg-vocab"), campoVocab: $("#campo-vocab"),
};

function ler(chave, padrao = "") { try { return localStorage.getItem(chave) ?? padrao; } catch { return padrao; } }
function gravar(chave, valor) { try { localStorage.setItem(chave, valor); } catch { /* sem armazenamento */ } }
function novoId() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }
const agoraS = () => (performance.timeOrigin + performance.now()) / 1000;   // relogio "de parede" (s)
const r4 = (x) => Math.round(x * 1e4) / 1e4;

const S = {
  info: null, est: null, modo: null, deteccao: "v3",
  usuario: ler("libres.usuario"), codigo: ler("libres.codigo"), sid: novoId(),
  ws: null, conectado: false, tentativas: 0, emVoo: 0, idLote: 0, tEnvio: 0, timerReconexao: null,
  fila: [], video: null, stream: null, cameras: [], camAtual: null,
  maos: null, pose: null, numHands: 2, trocando: false, ultimaPose: null, tPose: -1e9,
  fileset: null, proc: "", autoProc: false, testandoGPU: false, msDet: 0, nDet: 0,
  det: document.createElement("canvas"), visto: null, novoQuadro: false, tVideo: -1,
  fpsLocal: 0, tAnt: 0, exemplo: null, assinatura: {}, falarCada: ler("libres.falar") === "1",
  ultimoFalado: null,
};

// ------------------------------------------------------------------ carregando / erros
function carregando(titulo, detalhe = "", erro = false, escolher = false) {
  ui.carregando.hidden = false;
  ui.carTitulo.textContent = titulo;
  ui.carDetalhe.textContent = detalhe;
  ui.carregando.querySelector(".girando").style.display = erro ? "none" : "";
  ui.tentar.hidden = !erro;
  $("#escolher-camera").hidden = !escolher;
}
function pronto() { ui.carregando.hidden = true; }
function faixa(txt) { ui.faixa.hidden = !txt; ui.faixa.textContent = txt || ""; }
ui.tentar.addEventListener("click", () => location.reload());
$("#escolher-camera").addEventListener("click", () => escolherCamera());
$("#faixa-camera button").addEventListener("click", () => escolherCamera());

// ------------------------------------------------------------------ perfil
function pedirPerfil(motivo = "") {
  return new Promise((resolve) => {
    ui.campoUsuario.value = S.usuario || "";
    ui.blocoCodigo.hidden = !(S.info && S.info.precisa_codigo);
    ui.campoCodigo.value = S.codigo || "";
    if (motivo) ui.dlgPerfil.querySelector("p").textContent = motivo;
    ui.dlgPerfil.addEventListener("close", () => {
      const u = ui.campoUsuario.value.trim();
      if (u) { S.usuario = u; gravar("libres.usuario", u); }
      if (!ui.blocoCodigo.hidden) { S.codigo = ui.campoCodigo.value.trim(); gravar("libres.codigo", S.codigo); }
      resolve();
    }, { once: true });
    ui.dlgPerfil.showModal();
    ui.campoUsuario.focus();
  });
}
ui.perfil.addEventListener("click", async () => {
  await pedirPerfil("Cada pessoa tem a sua calibração. Trocar o perfil começa uma sessão nova.");
  S.sid = novoId();
  reconectar(true);
});

// ------------------------------------------------------------------ MediaPipe
async function criarPar(delegate, numHands = 2) {
  // os MESMOS limiares do app de computador e da extracao dos videos (modo IMAGEM)
  const criar = (Classe, opcoes) => Classe.createFromOptions(S.fileset, { ...opcoes,
    baseOptions: { ...opcoes.baseOptions, delegate } });
  const maos = await criar(HandLandmarker, {
    baseOptions: { modelAssetPath: "modelos/hand_landmarker.task" }, runningMode: "IMAGE", numHands,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
  const pose = await criar(PoseLandmarker, {
    baseOptions: { modelAssetPath: "modelos/pose_landmarker_lite.task" }, runningMode: "IMAGE", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, outputSegmentationMasks: false });
  return { maos, pose };
}

async function criarDetectores() {
  if (!S.fileset) S.fileset = await FilesetResolver.forVisionTasks(new URL("vendor/tasks-vision/wasm", location.href).href);
  const salvo = ler("libres.proc");
  let quer = PROCESSAMENTO === "gpu" ? "GPU" : PROCESSAMENTO === "cpu" ? "CPU"
    : PROCESSAMENTO === "auto" ? "auto" : (salvo === "GPU" || salvo === "CPU" ? salvo : "auto");
  S.autoProc = quer === "auto";
  if (quer === "auto") quer = "CPU";
  let par;
  try { par = await criarPar(quer, 2); }
  catch (e) {
    if (quer === "CPU") throw e;
    console.warn("GPU indisponivel; usando CPU", e);
    quer = "CPU"; par = await criarPar("CPU", 2);
  }
  S.maos = par.maos; S.pose = par.pose; S.numHands = 2; S.proc = quer;
}

async function testarGPU() {
  // auto: a CPU ficou lenta neste aparelho (celular): a GPU e mais rapida? (decide uma vez e guarda)
  S.testandoGPU = true;
  try {
    const gpu = await criarPar("GPU", S.numHands);
    gpu.maos.detect(S.det);                            // a 1a vez compila os shaders
    const t0 = performance.now();
    for (let i = 0; i < 5; i++) gpu.maos.detect(S.det);
    const msGpu = (performance.now() - t0) / 5;
    if (msGpu < 0.7 * S.msDet) {
      const velhos = [S.maos, S.pose];
      S.maos = gpu.maos; S.pose = gpu.pose; S.proc = "GPU"; S.msDet = msGpu;
      velhos.forEach((d) => { try { d.close(); } catch { /* ja fechado */ } });
      gravar("libres.proc", "GPU");
    } else {
      gpu.maos.close(); gpu.pose.close();
      gravar("libres.proc", "CPU");
    }
  } catch (e) { console.warn("GPU indisponivel", e); gravar("libres.proc", "CPU"); }
  S.autoProc = false; S.testandoGPU = false;
}

async function ajustarDeteccao(modo) {
  const tipo = (S.info && S.info.deteccao && S.info.deteccao[modo]) || "v3";
  S.deteccao = tipo;
  const n = tipo === "1mao" ? 1 : 2;           // alfabeto: uma mao, como o landmarker do app
  if (!S.maos || n === S.numHands) return;
  S.trocando = true;
  try { await S.maos.setOptions({ numHands: n }); S.numHands = n; } catch (e) { console.warn(e); }
  S.trocando = false;
  S.ultimaPose = null; S.tPose = -1e9;
}

// ------------------------------------------------------------------ camera
// cameras virtuais (OBS, NVIDIA Broadcast...) so funcionam com o programa delas aberto:
// sem escolha da pessoa, a pagina prefere a webcam de verdade (como o app de computador)
const VIRTUAL = /obs|virtual|broadcast|droidcam|manycam|snap ?camera|xsplit|vcam|epoccam|iriun|\bcamo\b/i;
const ERROS_CAMERA = {
  NotAllowedError: ["Câmera não autorizada", "Libere a câmera para este site: clique no ícone de câmera (ou no " +
    "cadeado) ao lado do endereço e escolha Permitir. No Windows, confira também Configurações > Privacidade e " +
    "segurança > Câmera > deixar os aplicativos da área de trabalho acessarem a câmera."],
  NotReadableError: ["A câmera está ocupada", "Outro programa está usando a câmera (o app de computador do " +
    "Libre-se, Teams, Meet, Zoom, OBS...). Feche-o, ou escolha outra câmera."],
  NotFoundError: ["Nenhuma câmera encontrada", "Conecte uma webcam e toque em Tentar de novo."],
  OverconstrainedError: ["A câmera escolhida não está disponível", "Ela pode ter sido desconectada. Escolha outra câmera."],
  SemImagem: ["A câmera não manda imagem", "Ela abriu mas não mandou nenhum quadro (câmera virtual desligada, " +
    "como a do OBS?). Escolha outra câmera."],
};
ERROS_CAMERA.SecurityError = ERROS_CAMERA.NotAllowedError;
const nomeCamera = (c, i) => (c && c.label) || `Câmera ${i + 1}`;

async function listarCameras() {
  try {
    const todos = await navigator.mediaDevices.enumerateDevices();
    S.cameras = todos.filter((d) => d.kind === "videoinput");
  } catch { S.cameras = []; }
  const b = $("#btn-camera");
  if (b) b.firstChild.textContent = S.camNome ? `Câmera: ${S.camNome.slice(0, 26)}… ` : "Câmera… ";
}

function comTempo(promessa, ms, msg) {
  return Promise.race([promessa, new Promise((_, falha) => setTimeout(() =>
    falha(Object.assign(new Error(msg), { name: "SemImagem" })), ms))]);
}

function esperarImagem(ms) {
  return new Promise((ok, falha) => {
    const t0 = performance.now();
    const ver = () => {
      if (S.video.videoWidth > 0 && S.video.readyState >= 2) return ok();
      if (performance.now() - t0 > ms) return falha(Object.assign(new Error("nenhum quadro"), { name: "SemImagem" }));
      setTimeout(ver, 100);
    };
    ver();
  });
}

async function pedirStream(deviceId) {
  const base = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" };
  try {
    return await navigator.mediaDevices.getUserMedia({ video: { ...base, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    if (e.name === "NotAllowedError" || e.name === "SecurityError") throw e;
    return await navigator.mediaDevices.getUserMedia({ video: deviceId ? base : true, audio: false });  // sem pedir tamanho
  }
}

function pararStream() {
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
}

async function usarStream(stream) {
  if (!S.video) {
    S.video = document.createElement("video");
    S.video.muted = true; S.video.playsInline = true; S.video.setAttribute("playsinline", "");
    S.video.style.cssText = "position:absolute;width:2px;height:2px;opacity:0;pointer-events:none;left:0;top:0";
    document.body.append(S.video);
  }
  S.video.srcObject = stream;
  try {
    await comTempo(S.video.play(), 6000, "a câmera não começou a mandar imagem");
    await esperarImagem(5000);
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    S.video.srcObject = null;
    throw e;
  }
  S.stream = stream;
  const trilha = stream.getVideoTracks()[0];
  S.camAtual = trilha && trilha.getSettings ? trilha.getSettings().deviceId : null;
  S.camNome = trilha ? trilha.label : "";
  S.tQuadro = performance.now();
  S.mon = novoMonitor();
  problemaImagem("");
  agendarQuadro();
  await listarCameras();
}

async function abrirCamera(escolhida = null) {
  // escolhida (pela pessoa) e guardada para a proxima vez; sem ela: a guardada, a padrao do sistema e,
  // se falharem, as outras (as de verdade antes das virtuais)
  const salva = ler("libres.camera");
  const tentadas = new Set();
  let erro = null;
  const tentar = async (id) => {
    tentadas.add(id || "padrao");
    pararStream();
    try { await usarStream(await pedirStream(id)); return true; }
    catch (e) {
      console.warn("camera", id || "padrao", e);
      erro = e;
      if (e.name === "NotAllowedError" || e.name === "SecurityError") throw e;
      return false;
    }
  };
  const ordem = escolhida ? [escolhida] : [salva, null].filter((x) => x !== "");
  let abriu = false;
  for (const id of ordem) if (!abriu) abriu = await tentar(id);
  if (!abriu && !escolhida) {
    await listarCameras();
    const outras = [...S.cameras].filter((c) => c.deviceId && !tentadas.has(c.deviceId))
      .sort((a, b) => VIRTUAL.test(a.label) - VIRTUAL.test(b.label));
    for (const c of outras) if (!abriu) abriu = await tentar(c.deviceId);
  }
  if (!abriu) throw erro || Object.assign(new Error("nenhuma câmera abriu"), { name: "NotFoundError" });
  if (escolhida) gravar("libres.camera", escolhida);
  // abriu sozinha numa camera virtual e existe uma de verdade: usa a de verdade
  if (!escolhida && !salva && VIRTUAL.test(S.camNome)) {
    const virtual = S.camAtual;
    const real = S.cameras.find((c) => c.deviceId && c.label && !VIRTUAL.test(c.label) && c.deviceId !== virtual);
    if (real && !(await tentar(real.deviceId))) await tentar(virtual);
  }
}

async function trocarPara(id, guardar = true) {
  const anterior = S.camAtual;
  try {
    await abrirCamera(id);
    if (!guardar) gravar("libres.camera", "");
    aviso(`câmera: ${S.camNome || "trocada"}`);
    return true;
  } catch (e) {
    const [titulo] = ERROS_CAMERA[e.name] || ["A câmera não abriu"];
    aviso(`${titulo} (${e.name}). Voltando para a anterior.`, true, 6000);
    if (anterior) { try { await abrirCamera(anterior); } catch { /* nenhuma abre */ } }
    return false;
  }
}

async function trocarCamera() {                      // tecla C: a proxima camera da lista
  await listarCameras();
  if (S.cameras.length < 2) { aviso("só existe uma câmera"); return; }
  const i = S.cameras.findIndex((c) => c.deviceId === S.camAtual);
  await trocarPara(S.cameras[(i + 1) % S.cameras.length].deviceId);
}

async function escolherCamera(motivo = "") {
  await listarCameras();
  const sel = $("#sel-camera");
  sel.replaceChildren(...S.cameras.map((c, i) => {
    const o = el("option", { value: c.deviceId, text: nomeCamera(c, i) + (c.deviceId === S.camAtual ? " (em uso)" : "")
      + (VIRTUAL.test(c.label) ? " — virtual" : "") });
    if (c.deviceId === S.camAtual) o.selected = true;
    return o;
  }));
  const semNomes = S.cameras.length && S.cameras.every((c) => !c.label);
  $("#camera-dica").textContent = motivo || (!S.cameras.length ? "Nenhuma câmera encontrada. Conecte uma webcam."
    : semNomes ? "O navegador ainda não liberou a câmera para este site, por isso as câmeras aparecem sem nome. " +
      "Libere a câmera (ícone ao lado do endereço) e escolha de novo."
      : "Escolha a webcam de verdade. As virtuais (OBS, NVIDIA Broadcast...) só funcionam com o programa delas aberto.");
  $("#dlg-camera").showModal();
}
$("#dlg-camera").addEventListener("close", async () => {
  const dlg = $("#dlg-camera");
  if (dlg.returnValue !== "usar") return;
  const id = $("#sel-camera").value;
  if (!id) return;
  if (!S.maos) {                                     // a camera nao tinha aberto: continua o inicio
    gravar("libres.camera", id);
    if (await etapaCamera(id)) await etapaDeteccao();
  } else {
    await trocarPara(id, true);
  }
});
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", listarCameras);
}

// imagem parada (camera virtual sem o programa), preta ou que parou de chegar: avisa (e, sem
// escolha da pessoa, tenta sozinho a proxima camera), como o MonitorImagem do app de computador
function novoMonitor() {
  const c = document.createElement("canvas"); c.width = 48; c.height = 27;
  return { c, ctx: c.getContext("2d", { willReadFrequently: true }), ant: null, t: 0, parada: 0, preta: 0 };
}
function monitorar() {
  const m = S.mon, agora = performance.now();
  if (!m || agora - m.t < 400) return;
  m.t = agora;
  m.ctx.drawImage(S.det, 0, 0, m.c.width, m.c.height);
  const d = m.ctx.getImageData(0, 0, m.c.width, m.c.height).data;
  const g = new Uint8Array(d.length / 4);
  let soma = 0, igual = !!m.ant;
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    g[j] = (d[i] + d[i + 1] + d[i + 2]) / 3; soma += g[j];
    if (igual && m.ant[j] !== g[j]) igual = false;
  }
  m.ant = g;
  m.parada = igual ? m.parada + 1 : 0;              // camera de verdade sempre tem ruido: nunca sai identica
  m.preta = soma / g.length < 12 ? m.preta + 1 : 0;
  problemaImagem(m.preta >= 4 ? "preta" : m.parada >= 4 ? "parada" : "");
}
function problemaImagem(tipo) {
  const fc = $("#faixa-camera");
  if (tipo === (S.problemaCam || "")) return;
  S.problemaCam = tipo;
  if (!tipo) { fc.hidden = true; return; }
  if (!ler("libres.camera") && S.cameras.length > 1) {        // ninguem escolheu: tenta a proxima sozinho
    S.trocadas = S.trocadas || new Set();
    S.trocadas.add(S.camAtual);
    const livres = S.cameras.filter((c) => c.deviceId && !S.trocadas.has(c.deviceId));
    const prox = livres.find((c) => !VIRTUAL.test(c.label)) || livres[0];
    if (prox) { trocarPara(prox.deviceId, false); return; }
  }
  const txt = { parada: "está mandando a imagem parada (parece câmera virtual: OBS, NVIDIA Broadcast...)",
    preta: "está mandando imagem preta (tampada, desligada ou virtual)", sem: "parou de mandar imagem" }[tipo];
  fc.querySelector("span").textContent = `A câmera "${S.camNome || "atual"}" ${txt}.`;
  fc.hidden = false;
}

function agendarQuadro() {
  // uma corrente so por elemento de video (trocar de camera reaproveita o mesmo)
  if (!S.video || !S.video.requestVideoFrameCallback || S.rvfc) return;
  S.rvfc = true;
  const proximo = () => { S.novoQuadro = true; S.tQuadro = performance.now(); S.video.requestVideoFrameCallback(proximo); };
  S.video.requestVideoFrameCallback(proximo);
}

// ------------------------------------------------------------------ deteccao de cada quadro
function detectar() {
  const v = S.video;
  if (!v || v.readyState < 2 || !v.videoWidth || S.trocando || !S.maos) return;
  if (v.requestVideoFrameCallback) {
    if (!S.novoQuadro) return;
    S.novoQuadro = false;
  } else {
    if (v.currentTime === S.tVideo) return;
    S.tVideo = v.currentTime;
    S.tQuadro = performance.now();
  }
  const t = agoraS();
  const dw = Math.min(LARGURA_DETECCAO, v.videoWidth);
  const dh = Math.round(dw * v.videoHeight / v.videoWidth);
  if (S.det.width !== dw || S.det.height !== dh) { S.det.width = dw; S.det.height = dh; }
  const dc = S.det.getContext("2d", { willReadFrequently: false });
  dc.setTransform(-1, 0, 0, 1, dw, 0);              // espelhada, como o cv2.flip do app
  dc.drawImage(v, 0, 0, dw, dh);
  dc.setTransform(1, 0, 0, 1, 0, 0);
  monitorar();
  const tDet = performance.now();
  let res;
  try { res = S.maos.detect(S.det); } catch (e) { console.warn(e); return; }
  const m = [null, null], l = [-1, -1];
  (res.landmarks || []).slice(0, 2).forEach((mao, k) => {
    m[k] = mao.flatMap((p) => [r4(p.x), r4(p.y), r4(p.z)]);
    const cat = res.handedness && res.handedness[k] && res.handedness[k][0];
    l[k] = cat ? (cat.categoryName.toLowerCase().startsWith("r") ? 1 : 0) : -1;
  });
  let p = null;
  if (S.deteccao === "v3" && S.pose) {
    if (t - S.tPose >= POSE_CADA) {
      try {
        const rp = S.pose.detect(S.det);
        const pts = rp.landmarks && rp.landmarks[0];
        S.ultimaPose = pts ? pts.flatMap((q) => [r4(q.x), r4(q.y), r4(q.z),
          r4(typeof q.visibility === "number" ? q.visibility : 1.0)]) : null;
      } catch (e) { console.warn(e); S.ultimaPose = null; }
      S.tPose = t;
    }
    p = S.ultimaPose;
  }
  const ms = performance.now() - tDet;
  S.msDet = S.msDet ? 0.9 * S.msDet + 0.1 * ms : ms;
  S.nDet += 1;
  if (S.autoProc && !S.testandoGPU && S.nDet >= 40) {
    if (S.msDet > MS_LENTO) testarGPU();
    else { S.autoProc = false; gravar("libres.proc", "CPU"); }
  }
  if (S.tAnt) {
    const dt = t - S.tAnt;
    if (dt > 0 && dt < 1) S.fpsLocal = S.fpsLocal ? 0.9 * S.fpsLocal + 0.1 / dt : 1 / dt;
  }
  S.tAnt = t;
  S.visto = { m };
  enfileirar({ t: Math.round(t * 1e6) / 1e6, a: r4(dw / dh), m, l, p });
}

function enfileirar(q) {
  S.fila.push(q);
  if (S.fila.length > MAX_FILA) S.fila.splice(0, S.fila.length - MAX_FILA);
}

// ------------------------------------------------------------------ conexao
function urlWs() {
  const q = new URLSearchParams({ usuario: S.usuario || "padrao", sid: S.sid });
  if (S.codigo) q.set("codigo", S.codigo);
  if (S.modo) q.set("modo", S.modo);
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?${q}`;
}

function conectar() {
  clearTimeout(S.timerReconexao);
  ui.conexao.textContent = "conectando…";
  const ws = new WebSocket(urlWs());
  S.ws = ws;
  ws.onopen = () => { S.conectado = true; S.tentativas = 0; S.emVoo = 0; ui.conexao.textContent = "conectado"; faixa(""); };
  ws.onmessage = (ev) => { let msg; try { msg = JSON.parse(ev.data); } catch { return; } tratar(msg); };
  ws.onclose = (ev) => {
    if (S.ws !== ws) return;
    S.conectado = false; S.emVoo = 0;
    if (ev.code === 4001) { pedirPerfil("Código de acesso errado. Confira com o grupo.").then(() => reconectar(true)); return; }
    S.tentativas += 1;
    const espera = Math.min(8000, 500 * 2 ** Math.min(S.tentativas, 4));
    ui.conexao.textContent = "sem conexão";
    faixa(`Sem conexão com o servidor. Tentando de novo em ${Math.round(espera / 1000)} s…`);
    S.timerReconexao = setTimeout(conectar, espera);
  };
}
function reconectar(sessaoNova = false) {
  if (sessaoNova) { S.fila = []; S.est = null; }
  const antigo = S.ws; S.ws = null;
  if (antigo) try { antigo.close(); } catch { /* ja fechado */ }
  conectar();
}

function enviar(obj) { if (S.ws && S.ws.readyState === 1) { S.ws.send(JSON.stringify(obj)); return true; } return false; }

setInterval(() => {
  if (!S.conectado || !S.fila.length) return;
  const agora = performance.now();
  if (S.emVoo >= MAX_EM_VOO && agora - S.tEnvio < 2000) return;     // servidor atrasado: espera
  if (S.emVoo >= MAX_EM_VOO) S.emVoo = 0;
  const lote = S.fila.splice(0, 90);
  if (enviar({ tipo: "quadros", id: ++S.idLote, q: lote })) { S.emVoo += 1; S.tEnvio = agora; }
  else S.fila.unshift(...lote);
}, ENVIO_MS);

function comando(cmd, extra = {}) {
  if (!enviar({ tipo: "cmd", cmd, id: ++S.idLote, ...extra })) aviso("sem conexão com o servidor");
}

function tratar(msg) {
  if (msg.tipo === "config") {
    S.info = { ...S.info, ...msg };
    ui.perfil.textContent = msg.usuario;
    renderizar(msg.estado);
    if (msg.avisos && msg.avisos.length) {              // ex.: modelo salvo com outra versao do scikit-learn
      console.warn("avisos do servidor:", msg.avisos);
      aviso("servidor: " + msg.avisos.join(" | "), true, 10000);
    }
  } else if (msg.tipo === "estado") {
    if (msg.ack) S.emVoo = Math.max(0, S.emVoo - 1);
    renderizar(msg);
  } else if (msg.tipo === "exemplo") {
    S.exemplo = { ...msg, t0: performance.now() / 1000 };
  } else if (msg.tipo === "erro") {
    faixa(msg.msg || "erro no servidor");
  }
}

// ------------------------------------------------------------------ painel
function aviso(txt, atencao = false, ms = 4000) {
  ui.aviso.textContent = txt || "";
  ui.aviso.title = txt || "";                          // o texto inteiro (a faixa mostra no maximo 2 linhas)
  ui.aviso.classList.toggle("atencao", !!atencao);
  clearTimeout(aviso.timer);
  if (txt) aviso.timer = setTimeout(() => { ui.aviso.textContent = ""; }, ms);
}
const nome = (p) => String(p).replace(/_/g, " ").toUpperCase();
const pct = (c) => (c == null ? "" : `${Math.round(c * 100)}%`);
function el(tag, attrs = {}, ...filhos) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  for (const f of filhos) if (f != null) e.append(f);
  return e;
}
function trocar(no, ...filhos) { no.replaceChildren(...filhos.filter((f) => f != null)); }
const externo = (href, text, cls = "") => el("a", { href, text, class: cls, target: "_blank", rel: "noopener noreferrer",
  title: "abre o dicionário numa nova aba" });
function mudou(chave, valor) {
  const s = JSON.stringify(valor);
  if (S.assinatura[chave] === s) return false;
  S.assinatura[chave] = s;
  return true;
}

function renderizar(e) {
  if (!e) return;
  const anterior = S.est;
  S.est = e;
  if (e.modo !== S.modo) { S.modo = e.modo; ajustarDeteccao(e.modo); S.exemplo = null; }
  ui.fps.textContent = `${(e.fps || S.fpsLocal || 0).toFixed(0)} fps${S.proc ? " · " + S.proc : ""}`;
  for (const b of ui.modos.querySelectorAll("button")) {
    b.classList.toggle("ativo", b.dataset.modo === e.modo);
    b.disabled = !(e.modos || []).includes(b.dataset.modo);
  }
  ui.texto.textContent = e.texto || "–";
  ui.texto.classList.toggle("vazio", !e.texto);
  ui.entrou.textContent = e.entrou ? `+ ${e.entrou}` : "";
  if (e.entrou && S.falarCada && e.entrou !== S.ultimoFalado && !(anterior && anterior.entrou === e.entrou)) {
    falar(e.entrou.includes("->") ? e.entrou.split("->").pop() : e.entrou);
  }
  S.ultimoFalado = e.entrou;
  if (e.aviso) aviso(e.aviso);
  if (mudou("agora", [e.modo, e.alfa, e.gravando, e.ultimo, e.cal && [e.cal.calibrando, e.cal.palavra, e.cal.n,
    e.cal.idx, e.cal.msg, e.cal.msg_ruim, e.cal.tem_reconhecedor, e.cal.ref]])) desenharAgora(e);
  if (mudou("lista", [e.modo, e.cal && [e.cal.palavras, e.cal.resumo, e.cal.detalhes, e.cal.calibrando, e.cal.fps_cal],
    S.info && S.info.palavras])) desenharLista(e);
  if (mudou("hist", e.historico)) {
    trocar(ui.historico, ...[...(e.historico || [])].reverse().map(([t, item]) =>
      el("li", {}, el("time", { text: new Date(t * 1000).toLocaleTimeString("pt-BR") }), el("span", { text: item }))));
  }
  const cal = e.modo === "CALIBRADO";
  const calibrando = cal && e.cal && e.cal.calibrando;
  for (const b of ui.acoes.querySelectorAll("button[data-cmd]")) {
    const c = b.dataset.cmd;
    if (c === "k") {
      b.hidden = !cal;
      b.firstChild.textContent = calibrando ? "Terminar calibração " : "Calibrar ";
    } else if (c === "v") b.hidden = !cal;
    else if ("pnrd".includes(c)) b.hidden = !calibrando;
  }
  $("#btn-vocab").hidden = !cal || calibrando;
  document.body.classList.toggle("calibrando", !!calibrando);   // celular: esconde o TEXTO para caber o painel
  $("#acoes-ctx").hidden = !cal;
}

function desenharAgora(e) {
  const f = [];
  if (e.modo === "ALFABETO") {
    f.push(el("div", { class: "rotulo", text: "Letra agora" }));
    const a = e.alfa;
    if (a && a.letra) {
      const cor = a.aceita ? "ok" : "suave";
      f.push(el("div", {}, el("span", { class: `grande ${cor}`, text: a.letra + (a.aceita ? "" : "?") }),
        el("span", { class: `medio ${cor}`, text: `  ${pct(a.conf)}` }),
        a.movendo ? el("span", { class: "medio atencao", text: "  movimento…" }) : null));
      f.push(el("div", { class: "suave linha", text: (a.top || []).map(([l, c]) => `${l} ${pct(c)}`).join("   ") }));
    } else f.push(el("div", { class: "suave", text: "mostre a mão" }));
  } else if (e.modo === "CALIBRADO" && e.cal && e.cal.calibrando) {
    const c = e.cal;
    f.push(el("div", { class: "rotulo" }, el("span", { text: `Calibração — palavra ${c.idx + 1} de ${c.total}` })));
    f.push(el("div", { class: "suave", text: "Faça o sinal:" }));
    f.push(el("div", { class: "grande", text: nome(c.palavra) }));
    f.push(el("div", { class: "comosefaz" },
      externo(linkDicionario(c.palavra), "Como se faz? ↗", "botao-link"),
      externo(DICIONARIO_INES, "Dicionário INES ↗", "botao-link discreto")));
    if (c.ref && !c.n) f.push(el("div", { class: "suave linha", text: "Ver exemplo (V): como a referência faz este sinal" }));
    const bol = el("div", { class: "bolinhas" });
    for (let k = 0; k < c.rep; k++) bol.append(el("i", { class: k < c.n ? "cheia" : "" }));
    bol.append(el("span", { text: `${Math.min(c.n, c.rep)} de ${c.rep}` }));
    f.push(bol);
    f.push(e.gravando ? el("div", { class: "gravando", text: "GRAVANDO… abaixe a mão no fim" })
      : el("div", { class: "suave", text: "faça o sinal e abaixe a mão" }));
    if (c.msg) f.push(el("div", { class: `msg ${c.msg_ruim ? "ruim" : "boa"}`, text: c.msg }));
  } else if (e.modo === "CALIBRADO" && e.cal && e.cal.tem_reconhecedor && e.ultimo.top.length && !e.gravando) {
    f.push(el("div", { class: "rotulo", text: "Último sinal" }));
    if (e.ultimo.ok) {
      e.ultimo.top.forEach(([p, c], k) => f.push(el("div", { class: k ? "suave" : "grande ok", text: `${nome(p)}  ${pct(c)}` })));
    } else {
      f.push(el("div", { class: "medio ruim", text: "NÃO RECONHECIDO" }));
      if (e.ultimo.melhor) f.push(el("div", { class: "suave", text: `mais parecido: ${nome(e.ultimo.melhor)}` }));
      f.push(el("div", { class: "suave", text: "(não entrou no texto)" }));
    }
    if (e.cal.msg) f.push(el("div", { class: `msg ${e.cal.msg_ruim ? "ruim" : "boa"}`, text: e.cal.msg }));
  } else {
    f.push(el("div", { class: "rotulo", text: "Último sinal" }));
    if (e.gravando) {
      f.push(el("div", { class: "gravando", text: "GRAVANDO SINAL…" }));
      f.push(el("div", { class: "suave linha", text: "abaixe a mão para terminar" }));
    } else if (e.modo === "CALIBRADO" && e.cal && !e.cal.tem_reconhecedor) {
      f.push(el("div", { class: "suave", text: "SEM CALIBRAÇÃO" }));
      f.push(el("div", { class: "atencao linha", text: "toque em Calibrar (K)" }));
    } else if (e.ultimo.top.length) {
      e.ultimo.top.forEach(([p, c], k) => {
        const aceito = k === 0 && e.ultimo.ok;
        f.push(el("div", { class: k ? "suave" : `grande ${aceito ? "ok" : "suave"}`, text: `${nome(p)}  ${pct(c)}` }));
      });
      if (!e.ultimo.ok) f.push(el("div", { class: "suave linha", text: "(incerto — não entrou no texto)" }));
    } else f.push(el("div", { class: "suave", text: "faça o sinal e abaixe a mão" }));
    if (e.modo === "CALIBRADO" && e.cal && e.cal.msg) {
      f.push(el("div", { class: `msg ${e.cal.msg_ruim ? "ruim" : "boa"}`, text: e.cal.msg }));
    }
  }
  trocar(ui.agora, ...f);
}

function desenharLista(e) {
  const f = [];
  if (e.modo === "PALAVRAS" && S.info && S.info.palavras) {
    const pal = S.info.palavras;
    f.push(el("div", { class: "rotulo", text: `Sinais que o modelo conhece (${pal.length})` }));
    const itens = el("div", { class: "itens" });
    pal.forEach((p, k) => {
      if (k) itens.append(", ");
      itens.append(externo(linkDicionario(p), String(p).replace(/_/g, " ").toLowerCase(), "palavra"));
    });
    f.push(itens);
    f.push(el("div", { class: "suave linha", text: "Toque numa palavra para ver o sinal no dicionário (pode haver " +
      "variação regional: o modelo aprendeu com os vídeos do conjunto de dados)." }));
  } else if (e.modo === "CALIBRADO" && e.cal) {
    const c = e.cal;
    const prontas = c.palavras.filter(([, n]) => n > 0);
    f.push(el("div", { class: "rotulo" }, el("span", { text: `Sinais calibrados (${prontas.length} de ${c.total})` })));
    const itens = el("div", { class: "itens" });
    c.palavras.forEach(([p, n], k) => {
      if (k) itens.append(", ");
      itens.append(externo(linkDicionario(p), p.toLowerCase() + (n >= c.rep ? "" : n ? "*" : " (falta)"),
        n >= c.rep ? "palavra" : "palavra falta"));
    });
    f.push(itens);
    if (c.resumo && c.resumo.length) {
      f.push(el("div", { class: "resumo" }, ...c.resumo.map((r) =>
        el("div", { class: r.startsWith("conferencia") ? "ok" : "atencao", text: r }))));
    }
    if (c.detalhes && c.detalhes.length) {
      f.push(el("details", {}, el("summary", { text: "detalhes da conferência" }),
        el("ul", {}, ...c.detalhes.map((d) => el("li", { text: d })))));
    }
    const fpsCal = c.fps_cal;
    if (fpsCal && e.fps > 3 && e.fps < 0.7 * fpsCal) {
      f.push(el("div", { class: "atencao linha", text: `câmera mais lenta que na calibração (${e.fps.toFixed(0)}/${fpsCal.toFixed(0)} fps)` }));
    }
  }
  if (e.modo && DICAS[e.modo]) f.push(el("div", { class: "suave linha", text: DICAS[e.modo] }));
  trocar(ui.lista, ...f);
}

// ------------------------------------------------------------------ desenho (camera, maos, exemplo)
function desenhar() {
  const c = ui.tela, dpr = window.devicePixelRatio || 1;
  const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr);
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  const v = S.video;
  let area = { x: 0, y: 0, w: W, h: H };
  if (v && v.videoWidth) {
    const esc = Math.min(W / v.videoWidth, H / v.videoHeight);
    const w = v.videoWidth * esc, h = v.videoHeight * esc;
    area = { x: (W - w) / 2, y: (H - h) / 2, w, h };
    ctx.save(); ctx.translate(area.x + w, area.y); ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, w, h); ctx.restore();
  }
  if (S.visto) {                                   // os pontos que o MediaPipe achou (como no app)
    S.visto.m.forEach((mao, k) => {
      if (!mao) return;
      ctx.fillStyle = k === 0 ? COR.mao1 : COR.mao2;
      for (let i = 0; i < 63; i += 3) {
        ctx.beginPath(); ctx.arc(area.x + mao[i] * area.w, area.y + mao[i + 1] * area.h, 4 * dpr, 0, 2 * Math.PI); ctx.fill();
      }
    });
  }
  if (S.est && S.est.gravando) {
    ctx.fillStyle = COR.grav; ctx.beginPath();
    ctx.arc(area.x + area.w - 40 * dpr, area.y + 40 * dpr, 14 * dpr, 0, 2 * Math.PI); ctx.fill();
  }
  if (S.exemplo && !desenharExemplo(ctx, S.exemplo, performance.now() / 1000, area, dpr)) S.exemplo = null;
}

function desenharExemplo(ctx, ex, agora, area, dpr) {
  const T = ex.maos.length;
  const dur = T / ex.fps, volta = dur + PAUSA_EXEMPLO, dt = agora - ex.t0;
  if (dt >= VOLTAS_EXEMPLO * volta || T === 0) return false;
  const noCiclo = dt % volta;
  const k = Math.min(T - 1, Math.floor(noCiclo * ex.fps));
  const pw = Math.min(Math.max(area.w * 0.42, 250 * dpr), area.w - 24 * dpr), ph = pw * 0.7;
  const x0 = area.x + 12 * dpr, y0 = area.y + 12 * dpr, s = dpr;
  ctx.fillStyle = COR.painel; ctx.fillRect(x0, y0, pw, ph);
  ctx.strokeStyle = COR.cal; ctx.lineWidth = 2 * s; ctx.strokeRect(x0, y0, pw, ph);
  ctx.fillStyle = COR.texto; ctx.font = `700 ${16 * s}px system-ui, sans-serif`;
  ctx.fillText(`EXEMPLO: ${nome(ex.palavra)}`, x0 + 12 * s, y0 + 24 * s);
  ctx.fillStyle = COR.suave; ctx.font = `${12 * s}px system-ui, sans-serif`;
  ctx.fillText(ex.referencia ? "como a referência faz (você ainda não gravou este)"
    : `como foi feito na calibração (${ex.indice + 1} de ${ex.n})`, x0 + 12 * s, y0 + 42 * s);
  const ax0 = x0 + 12 * s, ay0 = y0 + 50 * s, aw = pw - 24 * s, ah = ph - 70 * s;
  const [bx0, by0, bx1, by1] = ex.caixa;
  const esc = Math.min(aw / Math.max(bx1 - bx0, 1e-6), ah / Math.max(by1 - by0, 1e-6));
  const ox = ax0 + (aw - esc * (bx1 - bx0)) / 2, oy = ay0 + (ah - esc * (by1 - by0)) / 2;
  const asp = ex.aspecto;
  const pt = (x, y) => [ox + (x * asp - bx0) * esc, oy + (y - by0) * esc];
  const linha = (a, b) => { ctx.beginPath(); ctx.moveTo(...pt(...a)); ctx.lineTo(...pt(...b)); ctx.stroke(); };
  if (ex.corpo) {
    const corpo = ex.corpo[k], idx = Object.fromEntries(ex.pontos_corpo.map((p, i) => [p, i]));
    const ok = (p) => corpo[idx[p]] && corpo[idx[p]][0] !== null;
    ctx.strokeStyle = COR.corpo; ctx.lineWidth = 2 * s;
    for (const [a, b] of LIGACOES_CORPO) if (ok(a) && ok(b)) linha(corpo[idx[a]], corpo[idx[b]]);
    if (ok(0)) {
      let raio = 12 * s;
      if (ok(11) && ok(12)) {
        const [x1, y1] = corpo[idx[11]], [x2, y2] = corpo[idx[12]];
        raio = 0.28 * esc * Math.hypot((x1 - x2) * asp, y1 - y2);
      }
      ctx.beginPath(); ctx.arc(...pt(...corpo[idx[0]]), Math.max(raio, 6 * s), 0, 2 * Math.PI); ctx.stroke();
    }
  }
  ex.maos[k].forEach((mao, slot) => {
    if (!mao || mao[0][0] === null) return;
    const cor = slot === ex.dom ? COR.mao1 : COR.mao2;
    ctx.strokeStyle = cor; ctx.fillStyle = cor; ctx.lineWidth = 2 * s;
    for (const [a, b] of LIGACOES_MAO) linha(mao[a], mao[b]);
    for (const q of mao) { ctx.beginPath(); ctx.arc(...pt(...q), 3 * s, 0, 2 * Math.PI); ctx.fill(); }
  });
  const frac = dur > 0 ? Math.min(1, noCiclo / dur) : 1;
  ctx.fillStyle = COR.linha; ctx.fillRect(x0 + 12 * s, y0 + ph - 14 * s, pw - 24 * s, 5 * s);
  ctx.fillStyle = COR.cal; ctx.fillRect(x0 + 12 * s, y0 + ph - 14 * s, (pw - 24 * s) * frac, 5 * s);
  return true;
}

function laco() {
  try { detectar(); } catch (e) { console.warn(e); }
  if (S.stream && S.tQuadro && document.visibilityState === "visible") {
    const parada = performance.now() - S.tQuadro > 3000;          // nenhum quadro novo em 3 s
    if (parada) problemaImagem("sem");
    else if (S.problemaCam === "sem") problemaImagem("");
  }
  desenhar();
  requestAnimationFrame(laco);
}
function iniciarLaco() { if (!S.lacoAtivo) { S.lacoAtivo = true; requestAnimationFrame(laco); } }

// ------------------------------------------------------------------ voz (LIBRAS -> fala)
function falar(txt) {
  if (!("speechSynthesis" in window) || !txt) return;
  const u = new SpeechSynthesisUtterance(String(txt).toLowerCase());
  u.lang = "pt-BR";
  const vozes = speechSynthesis.getVoices().filter((v) => v.lang && v.lang.toLowerCase().startsWith("pt"));
  if (vozes.length) u.voice = vozes.find((v) => v.lang.toLowerCase() === "pt-br") || vozes[0];
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ------------------------------------------------------------------ botoes e teclado
ui.modos.addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-modo]");
  if (b && !b.disabled) comando("modo", { alvo: b.dataset.modo });
});
ui.acoes.addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-cmd]");
  if (b) comando(b.dataset.cmd);
});
$("#btn-camera").addEventListener("click", () => escolherCamera());
// depois de clicar, o botao perde o foco: o ESPACO do teclado nao "clica" nele de novo
document.addEventListener("click", (ev) => {
  const b = ev.target.closest && ev.target.closest("button");
  if (b && !b.closest("dialog")) b.blur();
});
$("#btn-tela-cheia").addEventListener("click", telaCheia);
$("#btn-vocab").addEventListener("click", () => {
  const pal = (S.est && S.est.cal && S.est.cal.palavras) || [];
  ui.campoVocab.value = pal.map(([p]) => p).join("\n");
  ui.dlgVocab.showModal();
});
ui.dlgVocab.addEventListener("close", () => {
  if (ui.dlgVocab.returnValue !== "salvar") return;
  const palavras = ui.campoVocab.value.split("\n").map((s) => s.trim()).filter(Boolean);
  comando("vocabulario", { palavras });
});
function telaCheia() {
  const d = document;
  if (d.fullscreenElement) d.exitFullscreen && d.exitFullscreen();
  else d.documentElement.requestFullscreen && d.documentElement.requestFullscreen().catch(() => {});
}
// voz (LIBRAS -> fala) e o menu "Mais"
$("#btn-falar").addEventListener("click", () => falar(S.est && S.est.texto));
const btnAuto = $("#btn-falar-auto");
const rotuloAuto = () => { btnAuto.textContent = S.falarCada ? "Voz automática: ligada" : "Voz automática: desligada"; };
rotuloAuto();
btnAuto.addEventListener("click", () => { S.falarCada = !S.falarCada; gravar("libres.falar", S.falarCada ? "1" : "0"); rotuloAuto(); });
const menuMais = $("#menu-mais"), btnMais = $("#btn-mais");
function menu(abrir) { menuMais.hidden = !abrir; btnMais.setAttribute("aria-expanded", String(abrir)); }
btnMais.addEventListener("click", (ev) => { ev.stopPropagation(); menu(menuMais.hidden); });
menuMais.addEventListener("click", () => menu(false));
document.addEventListener("click", (ev) => { if (!menuMais.hidden && !ev.target.closest("#menu-mais")) menu(false); });

function comoSeFaz() {                               // tecla H: o sinal da vez no dicionario
  const e = S.est || {};
  let url = DICIONARIO_INES;
  if (e.modo === "ALFABETO") url = DICIONARIO_ALFABETO;
  else if (e.cal && e.cal.calibrando && e.cal.palavra) url = linkDicionario(e.cal.palavra);
  else if (e.ultimo && (e.ultimo.ok || e.ultimo.melhor)) url = linkDicionario(e.ultimo.ok || e.ultimo.melhor);
  window.open(url, "_blank", "noopener");
}

document.addEventListener("keydown", (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (document.querySelector("dialog[open]")) return;
  if (ev.target && ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;
  const k = ev.key.toLowerCase();
  const mapa = { m: () => comando("modo"), " ": () => comando("limpar"), backspace: () => comando("apagar"),
    c: trocarCamera, f: telaCheia, h: comoSeFaz };
  if (mapa[k]) { ev.preventDefault(); mapa[k](); return; }
  if (S.modo === "CALIBRADO" && "kvnprd".includes(k) && k.length === 1) { ev.preventDefault(); comando(k); }
});

// tela do celular sempre acesa durante a demonstracao
let travaTela = null;
async function manterTelaAcesa() {
  try { if ("wakeLock" in navigator && document.visibilityState === "visible") travaTela = await navigator.wakeLock.request("screen"); }
  catch { travaTela = null; }
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") manterTelaAcesa(); });

// ------------------------------------------------------------------ inicio
async function etapaCamera(escolhida = null) {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    carregando("Câmera bloqueada", "O navegador só libera a câmera em páginas seguras (https://) ou em localhost. " +
      "Abra o endereço https:// do servidor (ou http://localhost:8000 no próprio computador).", true);
    return false;
  }
  try {
    carregando("Abrindo a câmera…", "Permita o uso da câmera quando o navegador perguntar.");
    await abrirCamera(escolhida);
    return true;
  } catch (e) {
    console.error(e);
    const [titulo, dica] = ERROS_CAMERA[e.name] || ["A câmera não abriu", "Tente de novo ou escolha outra câmera."];
    carregando(titulo, `${dica}  [${e.name}${e.message ? ": " + e.message : ""}]`, true,
      !(e.name === "NotAllowedError" || e.name === "SecurityError"));
    return false;
  }
}

async function etapaDeteccao() {
  try {
    carregando("Carregando o reconhecimento das mãos…", "Na primeira vez baixa ~25 MB; depois fica guardado no navegador.");
    await criarDetectores();
    await ajustarDeteccao(S.modo || (S.info.modos && S.info.modos[0]));
  } catch (e) {
    console.error(e);
    carregando("O reconhecimento das mãos não carregou", String(e && e.message || e), true);
    return;
  }
  pronto();
  manterTelaAcesa();
  iniciarLaco();
}

async function iniciar() {
  carregando("Carregando…", "Buscando a configuração do servidor.");
  try {
    S.info = await (await fetch("api/info", { cache: "no-store" })).json();
  } catch {
    carregando("Servidor fora do ar", "Não foi possível falar com o servidor. Confira a internet e tente de novo.", true);
    return;
  }
  if (!S.usuario || (S.info.precisa_codigo && !S.codigo)) await pedirPerfil();
  ui.perfil.textContent = S.usuario || "padrao";
  conectar();
  if (SEM_CAMERA) { pronto(); iniciarLaco(); return; }
  if (await etapaCamera()) await etapaDeteccao();
}

// ------------------------------------------------------------------ ganchos de teste (?teste=1)
if (TESTE) {
  window.__libres = {
    estado: () => S.est,
    info: () => S.info,
    processamento: () => ({ proc: S.proc, msDet: S.msDet, nDet: S.nDet, fpsLocal: S.fpsLocal }),
    fila: () => S.fila.length,
    conectado: () => S.conectado,
    injetar: (quadros) => {                          // quadros prontos (sem camera): vao para a fila e para a tela
      for (const q of quadros) enfileirar(q);
      if (quadros.length) S.visto = { m: quadros[quadros.length - 1].m };
    },
    derrubar: () => { if (S.ws) S.ws.close(); },    // simula a internet caindo (a pagina reconecta sozinha)
    camera: () => ({ id: S.camAtual, nome: S.camNome, problema: S.problemaCam || "", lista: S.cameras.map((c) => c.label) }),
    comando,
    async detectarImagem(url, numHands = 2) {        // pontos do MediaPipe JS numa imagem (sem espelhar)
      if (!S.maos) await criarDetectores();
      S.autoProc = false;
      const img = new Image(); img.src = url; await img.decode();
      const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d").drawImage(img, 0, 0);
      await S.maos.setOptions({ numHands }); S.numHands = numHands;
      const r = S.maos.detect(c);
      const rp = S.pose ? S.pose.detect(c) : null;
      return { maos: r.landmarks.map((m) => m.map((p) => [p.x, p.y, p.z])),
        lado: r.handedness.map((h) => h[0].categoryName),
        pose: rp && rp.landmarks[0] ? rp.landmarks[0].map((p) => [p.x, p.y, p.z, p.visibility]) : null };
    },
  };
}

iniciar();
