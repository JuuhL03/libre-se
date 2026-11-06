// Cliente do WebSocket /ws do servidor do Libre-se (libre-se/servidor/app.py).
//
// E o mesmo protocolo da versao web (libre-se/web/app.js, secao "conexao"):
//   app -> servidor  { tipo: 'quadros', id, q: [quadro, ...] }   pontos de cada quadro
//                    { tipo: 'cmd', cmd, id, ...dados }          modo, limpar, apagar, calibracao...
//                    { tipo: 'ping', t }
//   servidor -> app  { tipo: 'config', usuario, modos, deteccao, palavras, estado }
//                    { tipo: 'estado', ack, ...estado }          texto traduzido e painel
//                    { tipo: 'exemplo', ... }                    exemplo de um sinal (calibracao)
//                    { tipo: 'erro', erro, msg }
//
// Nenhuma imagem sai do celular: so os pontos das maos e do corpo (RNF06 do TCC).

import { CODIGO_ACESSO, SERVIDOR_WS, USUARIO_PADRAO } from '../config';

// Mesmos valores da web
const ENVIO_MS = 80; // manda os quadros em lotes, ~12 por segundo
const MAX_EM_VOO = 3; // lotes sem resposta antes de esperar o servidor
const MAX_POR_LOTE = 90; // MAX_QUADROS_POR_MSG do servidor
const MAX_FILA = 240; // ~8 s de quadros guardados se a conexao cair

/** Um quadro no formato que o servidor espera (Sessao._converter em servidor/motor.py). */
export type Quadro = {
  t: number; // segundos, crescente
  a: number; // largura / altura da imagem analisada
  m: [number[] | null, number[] | null]; // 63 numeros (21 pontos x, y, z) por mao
  l: [number, number]; // lado de cada mao: 1 direita, 0 esquerda, -1 sem rotulo
  p: number[] | null; // 132 numeros (33 pontos x, y, z, visibilidade) ou null
};

export type Modo = 'ALFABETO' | 'PALAVRAS' | 'CALIBRADO';

/** Estado do painel que o servidor manda (Sessao.estado em servidor/motor.py). */
export type EstadoLibres = {
  modo: Modo;
  modos: Modo[];
  texto: string;
  pesquisa: boolean;
  fps: number;
  gravando: boolean;
  entrou: string | null;
  historico: [number, string][];
  aviso: string | null;
  ultimo: {
    top: [string, number | null][];
    ok: string | null;
    melhor: string | null;
  };
  alfa?: {
    letra: string;
    conf: number | null;
    aceita: boolean;
    movendo: boolean;
    top: [string, number | null][];
  } | null;
  alfa_cal?: unknown;
  alfa_pessoal?: boolean;
  cal?: {
    calibrando: boolean;
    rep: number;
    total: number;
    tem_reconhecedor: boolean;
    palavras: [string, number][];
    msg: string | null;
    msg_ruim: boolean;
    idx?: number;
    palavra?: string;
    n?: number;
    ref?: boolean;
  };
};

/** O que /api/info e a mensagem 'config' trazem. */
export type InfoLibres = {
  usuario?: string;
  modos: Modo[];
  deteccao: Record<string, '1mao' | 'v3'>;
  palavras: string[];
  avisos?: string[];
};

/** Ponto (x, y) normalizado na imagem; null onde a mao/corpo nao foi detectado. */
type Ponto2 = [number | null, number | null];

/**
 * Exemplo de um sinal ("Ver como faz", comando 'v'): como ele foi feito na
 * calibracao, quadro a quadro (servidor/motor.py, preparar_exemplo).
 */
export type ExemploSinal = {
  tipo: 'exemplo';
  palavra: string;
  indice: number; // qual dos exemplos gravados (0..n-1)
  n: number;
  referencia: boolean; // true: e o exemplo do perfil de referencia (a pessoa ainda nao gravou)
  aspecto: number; // largura / altura da imagem em que foi gravado
  fps: number;
  dom: number; // qual das 2 maos e a dominante (0 ou 1)
  caixa: [number, number, number, number]; // x0, y0, x1, y1 (x ja multiplicado pelo aspecto)
  maos: [Ponto2[] | null, Ponto2[] | null][]; // (T, 2, 21, 2)
  corpo: (Ponto2 | null)[][] | null; // (T, 9, 2), na ordem de pontos_corpo
  pontos_corpo: number[]; // indices do MediaPipe Pose: nariz, ombros, cotovelos, punhos, quadris
};

export type Conexao =
  | 'conectando'
  | 'conectado'
  | 'sem-conexao'
  | 'codigo-errado'
  | 'servidor-cheio';

type Ouvintes = {
  conexao?: (c: Conexao, detalhe?: string) => void;
  config?: (info: InfoLibres, estado: EstadoLibres) => void;
  estado?: (estado: EstadoLibres) => void;
  exemplo?: (exemplo: ExemploSinal) => void;
  erro?: (msg: string) => void;
};

function novoId(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export class ClienteLibres {
  private ws: WebSocket | null = null;
  private fila: Quadro[] = [];
  private emVoo = 0;
  private tEnvio = 0;
  private idLote = 0;
  private tentativas = 0;
  private timerEnvio: ReturnType<typeof setInterval> | null = null;
  private timerReconexao: ReturnType<typeof setTimeout> | null = null;
  private fechado = false;
  private readonly sid = novoId(); // a mesma sessao se a conexao cair e voltar
  conectado = false;

  constructor(
    private ouvintes: Ouvintes,
    private usuario: string = USUARIO_PADRAO,
    private modo: Modo | null = null,
  ) {}

  private url(): string {
    const q: string[] = [
      `usuario=${encodeURIComponent(this.usuario || 'padrao')}`,
      `sid=${encodeURIComponent(this.sid)}`,
    ];
    if (CODIGO_ACESSO) q.push(`codigo=${encodeURIComponent(CODIGO_ACESSO)}`);
    if (this.modo) q.push(`modo=${encodeURIComponent(this.modo)}`);
    return `${SERVIDOR_WS}/ws?${q.join('&')}`;
  }

  iniciar(): void {
    this.fechado = false;
    this.conectar();
    if (!this.timerEnvio)
      this.timerEnvio = setInterval(() => this.enviarLote(), ENVIO_MS);
  }

  parar(): void {
    this.fechado = true;
    if (this.timerEnvio) clearInterval(this.timerEnvio);
    if (this.timerReconexao) clearTimeout(this.timerReconexao);
    this.timerEnvio = null;
    this.timerReconexao = null;
    const ws = this.ws;
    this.ws = null;
    this.conectado = false;
    this.fila = [];
    try {
      ws?.close();
    } catch {
      // ja fechado
    }
  }

  private conectar(): void {
    if (this.timerReconexao) clearTimeout(this.timerReconexao);
    this.ouvintes.conexao?.('conectando');
    const ws = new WebSocket(this.url());
    this.ws = ws;
    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.conectado = true;
      this.tentativas = 0;
      this.emVoo = 0;
      this.ouvintes.conexao?.('conectado');
    };
    ws.onmessage = ev => {
      let msg: any;
      try {
        msg = JSON.parse(String(ev.data));
      } catch {
        return;
      }
      this.tratar(msg);
    };
    ws.onerror = () => {
      // o onclose vem logo depois e cuida da reconexao
    };
    ws.onclose = ev => {
      if (this.ws !== ws) return;
      this.conectado = false;
      this.emVoo = 0;
      if (this.fechado) return;
      if (ev.code === 4001) {
        this.ouvintes.conexao?.('codigo-errado', 'codigo de acesso errado');
        return;
      }
      this.tentativas += 1;
      const espera = Math.min(8000, 500 * 2 ** Math.min(this.tentativas, 4));
      this.ouvintes.conexao?.(
        ev.code === 4002 ? 'servidor-cheio' : 'sem-conexao',
        `tentando de novo em ${Math.round(espera / 1000)} s`,
      );
      this.timerReconexao = setTimeout(() => this.conectar(), espera);
    };
  }

  private tratar(msg: any): void {
    if (!msg || typeof msg !== 'object') return;
    if (msg.tipo === 'config') {
      const { estado, ...info } = msg;
      this.ouvintes.config?.(info as InfoLibres, estado as EstadoLibres);
    } else if (msg.tipo === 'estado') {
      if (msg.ack) this.emVoo = Math.max(0, this.emVoo - 1);
      this.ouvintes.estado?.(msg as EstadoLibres);
    } else if (msg.tipo === 'exemplo') {
      if (Array.isArray(msg.maos) && msg.maos.length > 0)
        this.ouvintes.exemplo?.(msg as ExemploSinal);
    } else if (msg.tipo === 'erro') {
      this.ouvintes.erro?.(msg.msg || 'erro no servidor');
    }
  }

  private enviar(obj: unknown): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
      return true;
    }
    return false;
  }

  /** Guarda o quadro para o proximo lote (a cada 80 ms, como na web). */
  quadro(q: Quadro): void {
    this.fila.push(q);
    if (this.fila.length > MAX_FILA)
      this.fila.splice(0, this.fila.length - MAX_FILA);
  }

  private enviarLote(): void {
    if (!this.conectado || !this.fila.length) return;
    const agora = Date.now();
    if (this.emVoo >= MAX_EM_VOO && agora - this.tEnvio < 2000) return; // servidor atrasado: espera
    if (this.emVoo >= MAX_EM_VOO) this.emVoo = 0;
    const lote = this.fila.splice(0, MAX_POR_LOTE);
    if (this.enviar({ tipo: 'quadros', id: ++this.idLote, q: lote })) {
      this.emVoo += 1;
      this.tEnvio = agora;
    } else {
      this.fila.unshift(...lote);
    }
  }

  /**
   * Comandos da sessao (Sessao.comando em servidor/motor.py):
   * - 'modo' (com { alvo }), 'limpar', 'apagar'
   * - CALIBRADO: 'k' (comecar/terminar), 'n' / 'p' (proxima/anterior), 'r' (refazer palavra),
   *   'd' (desfazer ultimo exemplo), 'v' (ver o exemplo do sinal)
   * - ALFABETO: 'kalfa' (calibrar o alfabeto), 'nalfa' / 'palfa', 'ralfa' (apagar calibracao)
   * - 'vocabulario' (com { palavras })
   */
  comando(cmd: string, dados: Record<string, unknown> = {}): boolean {
    if (cmd === 'modo' && typeof dados.alvo === 'string')
      this.modo = dados.alvo as Modo;
    return this.enviar({ tipo: 'cmd', cmd, id: ++this.idLote, ...dados });
  }
}
