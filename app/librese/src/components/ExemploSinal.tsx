// "Ver como faz": desenha o exemplo de um sinal (maos e corpo em movimento) num quadro
// no canto da camera, como a versao web (libre-se/web/app.js, desenharExemplo).
// Passa 2 vezes e some; tocar no quadro fecha antes.

import React, { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, useWindowDimensions } from 'react-native';
import { Circle, G, Line, Rect, Svg } from 'react-native-svg';
import type { ExemploSinal as Exemplo } from '../servidor/ClienteLibres';

// mesmas ligacoes e cores da web
const LIGACOES_MAO: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const LIGACOES_CORPO: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
];
const VOLTAS = 2;
const PAUSA = 0.6; // s entre uma passada e outra
const COR_MAO1 = '#00ff00'; // mao dominante
const COR_MAO2 = '#00c8ff';
const COR_CORPO = '#969696';
const COR_CAL = '#be6ec8';
const COR_LINHA = '#34383f';

type XY = [number | null, number | null];
const valido = (p: XY | null | undefined): p is [number, number] =>
  !!p && p[0] != null && p[1] != null;

type Props = {
  exemplo: Exemplo & { recebidoEm: number };
  onFim: () => void;
};

export default function ExemploSinal({ exemplo: ex, onFim }: Props) {
  const { width: larguraTela } = useWindowDimensions();
  const [agora, setAgora] = useState(Date.now());
  // o pai recria o onFim a cada render (o estado do servidor chega ~12x/s): guarda o ultimo
  const fimRef = useRef(onFim);
  fimRef.current = onFim;

  const T = ex.maos.length;
  const fps = ex.fps > 0 ? ex.fps : 30;
  const dur = T / fps;
  const volta = dur + PAUSA;

  // relogio da animacao (~30 quadros por segundo enquanto o exemplo passa)
  useEffect(() => {
    let vivo = true;
    let id = 0;
    const passo = () => {
      if (!vivo) return;
      const t = Date.now();
      if ((t - ex.recebidoEm) / 1000 >= VOLTAS * volta) {
        fimRef.current();
        return;
      }
      setAgora(t);
      id = requestAnimationFrame(passo);
    };
    id = requestAnimationFrame(passo);
    return () => {
      vivo = false;
      cancelAnimationFrame(id);
    };
  }, [ex, volta]);

  if (T === 0) return null;

  const dt = Math.max(0, (agora - ex.recebidoEm) / 1000);
  const noCiclo = dt % volta;
  const k = Math.min(T - 1, Math.floor(noCiclo * fps));
  const frac = dur > 0 ? Math.min(1, noCiclo / dur) : 1;

  // quadro: ~62% da largura da tela (no maximo 300), proporcao 4:3
  const pw = Math.min(300, larguraTela * 0.62);
  const ph = pw * 0.75;
  const [bx0, by0, bx1, by1] = ex.caixa;
  const esc = Math.min(pw / Math.max(bx1 - bx0, 1e-6), ph / Math.max(by1 - by0, 1e-6));
  const ox = (pw - esc * (bx1 - bx0)) / 2;
  const oy = (ph - esc * (by1 - by0)) / 2;
  const asp = ex.aspecto;
  const px = (x: number) => ox + (x * asp - bx0) * esc;
  const py = (y: number) => oy + (y - by0) * esc;

  const elementos: React.ReactNode[] = [];

  // corpo: ombros, bracos, quadris e a cabeca (circulo no nariz)
  if (ex.corpo && ex.corpo[k]) {
    const corpo = ex.corpo[k];
    const idx: Record<number, number> = {};
    ex.pontos_corpo.forEach((p, i) => (idx[p] = i));
    const ponto = (p: number) => corpo[idx[p]] as XY | null;
    LIGACOES_CORPO.forEach(([a, b], i) => {
      const pa = ponto(a);
      const pb = ponto(b);
      if (valido(pa) && valido(pb))
        elementos.push(
          <Line key={`c${i}`} x1={px(pa[0])} y1={py(pa[1])} x2={px(pb[0])} y2={py(pb[1])}
            stroke={COR_CORPO} strokeWidth={2} />,
        );
    });
    const nariz = ponto(0);
    if (valido(nariz)) {
      let raio = 12;
      const o1 = ponto(11);
      const o2 = ponto(12);
      if (valido(o1) && valido(o2))
        raio = 0.28 * esc * Math.hypot((o1[0] - o2[0]) * asp, o1[1] - o2[1]);
      elementos.push(
        <Circle key="cabeca" cx={px(nariz[0])} cy={py(nariz[1])} r={Math.max(raio, 6)}
          stroke={COR_CORPO} strokeWidth={2} fill="none" />,
      );
    }
  }

  // maos: a dominante em verde, a outra em azul
  (ex.maos[k] || []).forEach((mao, slot) => {
    if (!mao || !valido(mao[0])) return;
    const cor = slot === ex.dom ? COR_MAO1 : COR_MAO2;
    LIGACOES_MAO.forEach(([a, b], i) => {
      const pa = mao[a];
      const pb = mao[b];
      if (valido(pa) && valido(pb))
        elementos.push(
          <Line key={`m${slot}-${i}`} x1={px(pa[0])} y1={py(pa[1])} x2={px(pb[0])} y2={py(pb[1])}
            stroke={cor} strokeWidth={2} />,
        );
    });
    mao.forEach((q, i) => {
      if (valido(q))
        elementos.push(<Circle key={`p${slot}-${i}`} cx={px(q[0])} cy={py(q[1])} r={3} fill={cor} />);
    });
  });

  const nome = ex.palavra.replace(/_/g, ' ').toUpperCase();
  const legenda = ex.referencia
    ? 'como a referência faz (você ainda não gravou este)'
    : `como foi feito na calibração (${ex.indice + 1} de ${ex.n})`;

  return (
    <Pressable style={[styles.quadro, { width: pw + 24 }]} onPress={onFim}>
      <Text style={styles.titulo} numberOfLines={1}>EXEMPLO: {nome}</Text>
      <Text style={styles.legenda} numberOfLines={2}>{legenda}</Text>
      <Svg width={pw} height={ph + 12}>
        <G>{elementos}</G>
        <Rect x={0} y={ph + 5} width={pw} height={5} rx={2.5} fill={COR_LINHA} />
        <Rect x={0} y={ph + 5} width={pw * frac} height={5} rx={2.5} fill={COR_CAL} />
      </Svg>
      <Text style={styles.fechar}>toque para fechar</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  quadro: {
    position: 'absolute',
    top: 160,
    left: 12,
    backgroundColor: 'rgba(26,28,32,0.92)',
    borderColor: COR_CAL,
    borderWidth: 2,
    borderRadius: 12,
    padding: 12,
  },
  titulo: { color: '#f0f0f0', fontSize: 15, fontWeight: '700' },
  legenda: { color: '#9aa0a8', fontSize: 11, marginTop: 2, marginBottom: 6 },
  fechar: { color: '#9aa0a8', fontSize: 10, textAlign: 'right', marginTop: 4 },
});
