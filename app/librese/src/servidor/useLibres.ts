// Liga a camera do app a IA do Libre-se:
//   camera (VisionCamera v5) -> MediaPipe no celular (react-native-libres-landmarker)
//   -> pontos de cada quadro -> WebSocket /ws do servidor -> estado (texto traduzido)
//
// Faz no celular o mesmo que a funcao detectar() da versao web (libre-se/web/app.js).

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

export type Deteccao = '1mao' | 'v3';

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
  const [msDeteccao, setMsDeteccao] = useState(0);
  // "Ver como faz": o exemplo que o servidor mandou e quando chegou (a animacao conta dali)
  const [exemplo, setExemplo] = useState<
    (ExemploSinal & { recebidoEm: number }) | null
  >(null);

  const landmarker = useMemo(() => criarLandmarker(), []);
  const clienteRef = useRef<ClienteLibres | null>(null);
  const t0 = useRef(Date.now());
  const ultimoAvisoMao = useRef(false);
  const ultimoMs = useRef(0);

  // deteccao do modo atual: '1mao' (alfabeto) ou 'v3' (palavras: 2 maos + corpo)
  const modo = estado?.modo ?? null;
  const deteccao: Deteccao = (modo && info?.deteccao?.[modo]) || 'v3';

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
        exemplo: ex => setExemplo({ ...ex, recebidoEm: Date.now() }),
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
  }, [ativo, usuario]);

  // trocou de modo: o exemplo que estava passando sai (como na web)
  useEffect(() => {
    setExemplo(null);
  }, [modo]);

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
    (r: PontosQuadro, t: number, comPose: boolean, pose: number[] | null) => {
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
      const agora = Date.now();
      if (agora - ultimoMs.current > 1000) {
        ultimoMs.current = agora;
        setMsDeteccao(Math.round(r.ms));
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
        const t = (Date.now() - inicio) / 1000;
        // a pose so a cada 0,2 s (a ultima vale para os quadros do meio), como na web
        const pedirPose =
          comPose &&
          (g.__libresTPose === undefined || t - g.__libresTPose >= POSE_CADA);
        // de pe (enablePhysicalBufferRotation) e espelhada como selfie: na frontal o quadro ja vem
        // espelhado; na traseira, espelha aqui (o modelo foi treinado com a imagem espelhada)
        const espelhar = cameraFrontal ? frame.isMirrored : !frame.isMirrored;
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

  return {
    frameOutput,
    conexao,
    detalheConexao,
    info,
    estado,
    erro,
    limparErro: () => setErro(null),
    maoVisivel,
    msDeteccao,
    deteccao,
    exemplo,
    fecharExemplo: () => setExemplo(null),
    comando,
    trocarModo,
  };
}
