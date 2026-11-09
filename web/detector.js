// Libre-se web: o MediaPipe (maos + corpo) roda AQUI, num Web Worker, fora da thread da pagina.
// Antes ele rodava dentro do requestAnimationFrame: com as duas maos e o corpo (modo PALAVRAS) cada
// quadro passava de 33 ms, a imagem travava e chegavam 15-25 quadros/s ao servidor (o treino e a 30).
// E o mesmo arranjo do app movel: a camera fica fluida e a deteccao pega sempre o quadro mais novo.
//
// Worker CLASSICO de proposito: o MediaPipe carrega o proprio WASM com importScripts(), que nao
// existe em worker de modulo. A biblioteca (ES module) entra por import() dinamico.
//
// Mensagens (pagina -> worker), atendidas uma de cada vez, na ordem:
//   {tipo:"iniciar", proc:"gpu"|"cpu", numHands}           -> {tipo:"pronto", proc}
//   {tipo:"maos", n}                                        -> {tipo:"maos", n}
//   {tipo:"quadro", img:ImageBitmap, t, a, pose:bool}       -> {tipo:"pontos", t, a, m, l, p, ms, proc}
//   {tipo:"imagem", img:ImageBitmap, numHands, id}          -> {tipo:"imagem", id, maos, lado, pose}
// Erros: {tipo:"erro", id?, msg}

let FilesetResolver, HandLandmarker, PoseLandmarker;
let fileset = null, maos = null, pose = null, numHands = 2, proc = "GPU", forcarCpu = false;
const r4 = (x) => Math.round(x * 1e4) / 1e4;
const url = (rel) => new URL(rel, self.location.href).href;

async function carregarBiblioteca() {
  if (FilesetResolver) return;
  ({ FilesetResolver, HandLandmarker, PoseLandmarker } = await import(url("vendor/tasks-vision/vision_bundle.mjs")));
  fileset = await FilesetResolver.forVisionTasks(url("vendor/tasks-vision/wasm"));
}

async function criarPar(delegate, n) {
  // os MESMOS limiares do app de computador, do app movel e da extracao dos videos (modo IMAGEM)
  const criar = (Classe, opcoes) => Classe.createFromOptions(fileset, { ...opcoes,
    baseOptions: { ...opcoes.baseOptions, delegate } });
  const m = await criar(HandLandmarker, {
    baseOptions: { modelAssetPath: url("modelos/hand_landmarker.task") }, runningMode: "IMAGE", numHands: n,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
  let p;
  try {
    p = await criar(PoseLandmarker, {
      baseOptions: { modelAssetPath: url("modelos/pose_landmarker_lite.task") }, runningMode: "IMAGE", numPoses: 1,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, outputSegmentationMasks: false });
  } catch (e) { try { m.close(); } catch { /* ja fechado */ } throw e; }
  return { m, p };
}

function fechar() {
  for (const d of [maos, pose]) if (d) try { d.close(); } catch { /* ja fechado */ }
  maos = null; pose = null;
}

async function criarDetectores() {
  // como o app movel: tenta a GPU e, se ela falhar (criar ou detectar), fica na CPU de vez
  if (!forcarCpu) {
    try { const par = await criarPar("GPU", numHands); maos = par.m; pose = par.p; proc = "GPU"; return; }
    catch (e) { console.warn("[detector] GPU indisponivel; usando a CPU", e); }
  }
  const par = await criarPar("CPU", numHands);
  maos = par.m; pose = par.p; proc = "CPU";
}

// roda bloco(); se a GPU falhar no meio, recria tudo na CPU e tenta mais uma vez
async function comFallback(bloco) {
  try { return bloco(); }
  catch (e) {
    if (proc !== "GPU") throw e;
    console.warn("[detector] GPU falhou; passando para a CPU", e);
    fechar(); forcarCpu = true;
    await criarDetectores();
    return bloco();
  }
}

async function quadro(msg) {
  const { img, t, id } = msg;
  const t0 = performance.now();
  try {
    const res = await comFallback(() => maos.detect(img));
    const m = [null, null], l = [-1, -1];
    (res.landmarks || []).slice(0, 2).forEach((mao, k) => {
      m[k] = mao.flatMap((q) => [r4(q.x), r4(q.y), r4(q.z)]);
      const cat = res.handedness && res.handedness[k] && res.handedness[k][0];
      l[k] = cat ? (cat.categoryName.toLowerCase().startsWith("r") ? 1 : 0) : -1;
    });
    let p;                                     // undefined = nao pedida neste quadro (vale a ultima)
    if (msg.pose && pose) {
      try {
        const rp = await comFallback(() => pose.detect(img));
        const pts = rp.landmarks && rp.landmarks[0];
        p = pts ? pts.flatMap((q) => [r4(q.x), r4(q.y), r4(q.z),
          r4(typeof q.visibility === "number" ? q.visibility : 1.0)]) : null;
      } catch (e) { console.warn(e); p = null; }
    }
    self.postMessage({ tipo: "pontos", id, t, a: msg.a, m, l, p, ms: performance.now() - t0, proc });
  } finally {
    try { img.close(); } catch { /* ja fechado */ }
  }
}

async function imagem(msg) {                    // gancho dos testes (?teste=1): pontos de uma imagem parada
  const { img, id } = msg;
  try {
    if (msg.numHands !== numHands) { await maos.setOptions({ numHands: msg.numHands }); numHands = msg.numHands; }
    const r = await comFallback(() => maos.detect(img));
    const rp = pose ? await comFallback(() => pose.detect(img)) : null;
    self.postMessage({ tipo: "imagem", id,
      maos: r.landmarks.map((mm) => mm.map((q) => [q.x, q.y, q.z])),
      lado: r.handedness.map((h) => h[0].categoryName),
      pose: rp && rp.landmarks[0] ? rp.landmarks[0].map((q) => [q.x, q.y, q.z, q.visibility]) : null });
  } finally { try { img.close(); } catch { /* ja fechado */ } }
}

async function atender(msg) {
  switch (msg.tipo) {
    case "iniciar":
      forcarCpu = msg.proc === "cpu";
      numHands = msg.numHands || 2;
      await carregarBiblioteca();
      fechar();
      await criarDetectores();
      self.postMessage({ tipo: "pronto", proc });
      return;
    case "maos":
      if (maos && msg.n !== numHands) { await maos.setOptions({ numHands: msg.n }); numHands = msg.n; }
      self.postMessage({ tipo: "maos", n: numHands });
      return;
    case "quadro":
      if (!maos) { try { msg.img.close(); } catch { /* */ } self.postMessage({ tipo: "pontos", id: msg.id, t: msg.t, pulado: true }); return; }
      await quadro(msg);
      return;
    case "imagem":
      await imagem(msg);
      return;
  }
}

// uma mensagem de cada vez (setOptions e a criacao sao assincronos: nada de quadro no meio)
let fila = Promise.resolve();
self.onmessage = (ev) => {
  const msg = ev.data;
  fila = fila.then(() => atender(msg)).catch((e) => {
    console.error("[detector]", e);
    if (msg.tipo === "quadro" && msg.img) try { msg.img.close(); } catch { /* */ }
    self.postMessage({ tipo: "erro", id: msg.id, origem: msg.tipo, msg: String((e && e.message) || e) });
  });
};
