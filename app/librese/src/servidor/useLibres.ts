// Liga a camera do app a IA do Libre-se:
//   camera (VisionCamera v5) -> MediaPipe no celular (react-native-libres-landmarker)
//   -> pontos de cada quadro -> WebSocket /ws do servidor -> estado (texto traduzido)
//
// Faz no celular o mesmo que a funcao detectar() da versao web (libre-se/web/app.js).

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { useFrameOutput, type Frame } from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';
import {
  criarLandmarker,
  type PontosQuadro,
} from 'react-native-libres-landmarker';
import {
  ClienteLibres,
  type Conexao,
  type EstadoLibres,
  type ExemploSinal,
  type InfoLibres,
  type Modo,
  type Quadro,
} from './ClienteLibres';

const POSE_CADA = 0.2; // s: o corpo muda devagar (como na web)

// FORA do componente: o useFrameOutput recria a saida da camera quando este objeto muda
// (useMemo por referencia). Um objeto novo a cada render reconfigurava a camera a cada
// resposta do servidor (~12x/s): travava, picotava os quadros e nenhuma letra ficava
// estavel o bastante (0,3 s) para entrar no texto.
const RESOLUCAO_DETECCAO = { width: 640, height: 480 }; // a mesma largura de deteccao da web
const SEM_AVISO_QUADRO_PERDIDO = () => {};
const ESPELHAR_SEMPRE = Platform.OS === 'android';

export type Deteccao = '1mao' | 'v3';

/** Medias do ultimo segundo, para o topo da tela: de onde vem a lentidao. */
export type Diagnostico = {
  qps: number; // quadros por segundo que chegaram ao servidor
  ms: number; // deteccao nativa inteira (imagem + maos + corpo)
  msImagem: number;
  msMaos: number;
  msCorpo: number;
  msQuadro: number; // o quadro inteiro na thread da camera (ler o buffer + deteccao)
  processamento: string; // 'GPU' ou 'CPU'
};

type Soma = Omit<Diagnostico, 'qps' | 'processamento'> & { n: number };
const somaZerada = (): Soma => ({
  n: 0,
  ms: 0,
  msImagem: 0,
  msMaos: 0,
  msCorpo: 0,
  msQuadro: 0,
});

type Opcoes = {
  usuario?: string;
  cameraFrontal: boolean;
  ativo: boolean;
};

export function useLibres({ usuario, cameraFrontal, ativo }: Opcoes) {
  const [conexao, setConexao] = useState<Conexao>('conectando');
  const [detalheConexao, setDetalheConexao] = useState<string | undefined>();
  const [info, setInfo] = useState<InfoLibres | null>(null);
  const [estado, setEstado] = useState<EstadoLibres | null>(null);
  const [erro, setErro] = useState<string | null>(null);
  const [maoVisivel, setMaoVisivel] = useState(false);
  const [diag, setDiag] = useState<Diagnostico | null>(null);
  // false ate o 1o quadro detectado: na 1a vez o MediaPipe carrega os modelos e prepara a GPU
  // (alguns segundos); a tela mostra "preparando" em vez de parecer travada
  const [pronto, setPronto] = useState(false);
  const prontoRef = useRef(false);
  // "Ver como faz": o exemplo que o servidor mandou e quando chegou (a animacao conta dali)
  const [exemplo, setExemplo] = useState<
    (ExemploSinal & { recebidoEm: number }) | null
  >(null);
  // aviso do proprio app (ex.: a palavra nao tem exemplo da referencia), some sozinho
  const [avisoApp, setAvisoApp] = useState<string | null>(null);
  const timerAviso = useRef<ReturnType<typeof setTimeout> | null>(null);
  // "Ver como faz" mostra SEMPRE o exemplo da referencia, nunca o da pessoa. O servidor
  // mostra o da pessoa quando ela ja gravou; pedir de novo enquanto ele passa troca para o
  // da referencia (servidor/motor.py, Sessao._exemplo). fase 1: 1o pedido; 2: o 2o.
  const pedidoRef = useRef({ fase: 0, ate: 0 });

  const landmarker = useMemo(() => criarLandmarker(), []);
  const clienteRef = useRef<ClienteLibres | null>(null);
  const t0 = useRef(Date.now());
  const ultimoAvisoMao = useRef(false);
  const ultimoMs = useRef(0);
  const soma = useRef<Soma>(somaZerada());

  // deteccao do modo atual: '1mao' (alfabeto) ou 'v3' (palavras: 2 maos + corpo)
  const modo = estado?.modo ?? null;
  const deteccao: Deteccao = (modo && info?.deteccao?.[modo]) || 'v3';

  const avisar = useCallback((txt: string) => {
    setAvisoApp(txt);
    if (timerAviso.current) clearTimeout(timerAviso.current);
    timerAviso.current = setTimeout(() => setAvisoApp(null), 6000);
  }, []);

  // 1. conexao com o servidor
  useEffect(() => {
    if (!ativo) return;
    const cliente = new ClienteLibres(
      {
        conexao: (c, d) => {
          setConexao(c);
          setDetalheConexao(d);
        },
        config: (i, e) => {
          setInfo(i);
          setEstado(e);
          setErro(null);
        },
        estado: e => setEstado(e),
        exemplo: ex => {
          if (ex.referencia) {
            pedidoRef.current.fase = 0;
            setExemplo({ ...ex, recebidoEm: Date.now() });
            return;
          }
          // veio o exemplo da propria pessoa: nao mostra
          const p = pedidoRef.current;
          if (p.fase === 1 && Date.now() < p.ate) {
            p.fase = 2;
            cliente.comando('v'); // pede de novo: o servidor troca para o da referencia
            return;
          }
          p.fase = 0;
          avisar(
            `"${ex.palavra.replace(/_/g, ' ')}" ainda não tem exemplo da referência. Veja o vídeo em "Como se faz?"`,
          );
        },
        erro: m => setErro(m),
      },
      usuario,
    );
    clienteRef.current = cliente;
    cliente.iniciar();
    return () => {
      cliente.parar();
      clienteRef.current = null;
    };
  }, [ativo, usuario, avisar]);

  // trocou de modo: o exemplo que estava passando sai (como na web)
  useEffect(() => {
    setExemplo(null);
  }, [modo]);

  // trocou o tipo de deteccao (alfabeto: 1 mao; palavras: 2 maos + corpo): o detector e
  // refeito no proximo quadro (e o do corpo criado na 1a vez) -> "preparando" de novo
  useEffect(() => {
    prontoRef.current = false;
    setPronto(false);
  }, [deteccao]);

  // 2. numero de maos do detector conforme o modo (1 no alfabeto, como o landmarker do app)
  useEffect(() => {
    try {
      landmarker.configurar(deteccao === '1mao' ? 1 : 2);
    } catch (e: any) {
      setErro(`detector: ${e?.message ?? String(e)}`);
    }
  }, [landmarker, deteccao]);

  // 3. cada quadro detectado (thread da camera) chega aqui (thread do React)
  const receberPontos = useCallback(
    (
      r: PontosQuadro,
      t: number,
      comPose: boolean,
      pose: number[] | null,
      msQuadro: number,
    ) => {
      const m: [number[] | null, number[] | null] = [
        r.maos.length >= 63 ? r.maos.slice(0, 63) : null,
        r.maos.length >= 126 ? r.maos.slice(63, 126) : null,
      ];
      const l: [number, number] = [r.lados[0] ?? -1, r.lados[1] ?? -1];
      const q: Quadro = {
        t: Math.round(t * 1e6) / 1e6,
        a: Math.round((r.largura / r.altura) * 10000) / 10000,
        m,
        l,
        p: comPose && pose && pose.length === 132 ? pose : null,
      };
      clienteRef.current?.quadro(q);

      const visivel = m[0] !== null;
      if (visivel !== ultimoAvisoMao.current) {
        ultimoAvisoMao.current = visivel;
        setMaoVisivel(visivel);
      }
      // medias de 1 em 1 s (um setState por segundo, nao por quadro)
      if (!prontoRef.current) {
        prontoRef.current = true;
        setPronto(true);
      }

      const s = soma.current;
      s.n += 1;
      s.ms += r.ms;
      s.msImagem += r.msImagem;
      s.msMaos += r.msMaos;
      s.msCorpo += r.msCorpo;
      s.msQuadro += msQuadro;
      const agora = Date.now();
      const passou = agora - ultimoMs.current;
      if (passou >= 1000) {
        const media = (v: number) => Math.round(v / s.n);
        setDiag({
          qps: Math.round((s.n * 1000) / passou),
          ms: media(s.ms),
          msImagem: media(s.msImagem),
          msMaos: media(s.msMaos),
          msCorpo: media(s.msCorpo),
          msQuadro: media(s.msQuadro),
          processamento: r.processamento,
        });
        ultimoMs.current = agora;
        soma.current = somaZerada();
      }
    },
    [],
  );

  // o erro nativo chega com o stack trace do Java junto: na tela vai so a 1a linha (a causa),
  // e o texto inteiro vai para o log (adb logcat / Metro)
  const avisarErro = useCallback((m: string) => {
    console.warn(`[libres] detector: ${m}`);
    const linha =
      m.split('\n').find(l => l.trim() && !/^\s*at\s/.test(l)) ?? m;
    setErro(`detector: ${linha.trim().slice(0, 200)}`);
  }, []);

  // 4. o frame processor: MediaPipe nativo em cada quadro da camera
  const comPose = deteccao === 'v3';
  const inicio = t0.current;
  const onFrame = useCallback(
    (frame: Frame) => {
      'worklet';
      try {
        const g = globalThis as any;
        const tQuadro = Date.now();
        const t = (tQuadro - inicio) / 1000;
        // a pose so a cada 0,2 s (a ultima vale para os quadros do meio), como na web
        const pedirPose =
          comPose &&
          (g.__libresTPose === undefined || t - g.__libresTPose >= POSE_CADA);
        // de pe (enablePhysicalBufferRotation) e espelhada como selfie, como a web e o modelo
        // (treinado com a imagem espelhada).
        // Android: o ImageAnalysis do CameraX NUNCA espelha os pixels e, com o mirrorMode
        // 'auto', o VisionCamera 5 marca isMirrored = false ate na frontal: espelha sempre.
        // iOS: na frontal o quadro ja vem espelhado; na traseira, espelha aqui.
        const espelhar = ESPELHAR_SEMPRE
          ? true
          : cameraFrontal
          ? frame.isMirrored
          : !frame.isMirrored;
        const r = landmarker.detectar(
          frame.getPixelBuffer(),
          frame.width,
          frame.height,
          frame.bytesPerRow,
          frame.pixelFormat,
          frame.orientation,
          espelhar,
          pedirPose,
        );
        if (pedirPose) {
          g.__libresTPose = t;
          g.__libresPose = r.pose.length === 132 ? r.pose : null;
        }
        if (!comPose) g.__libresPose = null;
        scheduleOnRN(
          receberPontos,
          r,
          t,
          comPose,
          comPose ? g.__libresPose ?? null : null,
          Date.now() - tQuadro,
        );
      } catch (e: any) {
        scheduleOnRN(avisarErro, String(e?.message ?? e));
      } finally {
        frame.dispose();
      }
    },
    [landmarker, comPose, cameraFrontal, inicio, receberPontos, avisarErro],
  );

  const frameOutput = useFrameOutput({
    pixelFormat: 'rgb',
    targetResolution: RESOLUCAO_DETECCAO,
    enablePhysicalBufferRotation: true,
    dropFramesWhileBusy: true,
    onFrame,
    onFrameDropped: SEM_AVISO_QUADRO_PERDIDO,
  });

  const comando = useCallback(
    (cmd: string, dados?: Record<string, unknown>) => {
      const ok = clienteRef.current?.comando(cmd, dados) ?? false;
      if (!ok) setErro('sem conexão com o servidor');
      return ok;
    },
    [],
  );

  const trocarModo = useCallback(
    (alvo: Modo) => comando('modo', { alvo }),
    [comando],
  );

  // botao "Ver como faz": o exemplo da referencia (ver o ouvinte 'exemplo' acima)
  const verComoFaz = useCallback(() => {
    pedidoRef.current = { fase: 1, ate: Date.now() + 4000 };
    return comando('v');
  }, [comando]);

  return {
    frameOutput,
    conexao,
    detalheConexao,
    info,
    estado,
    erro,
    limparErro: () => setErro(null),
    maoVisivel,
    diag,
    pronto,
    deteccao,
    exemplo,
    fecharExemplo: () => setExemplo(null),
    avisoApp,
    verComoFaz,
    comando,
    trocarModo,
  };
}
