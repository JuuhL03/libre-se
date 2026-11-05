// Traducao ao vivo: a camera manda os pontos das maos (e do corpo) para a IA do
// Libre-se pelo WebSocket do servidor e mostra o texto que ela devolve.
// Os modos e a calibracao sao os mesmos da versao web.

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { Svg, Path } from 'react-native-svg';
import { useLibres } from '../servidor/useLibres';
import type { Modo } from '../servidor/ClienteLibres';
import ExemploSinal from '../components/ExemploSinal';
import { linkComoSeFaz } from '../dicionario';

const INK = '#12172B';
const PARCHMENT = '#F5F1E8';
const GOLD = '#D4A24C';
const RED = '#E26D5A';

const NOME_MODO: Record<Modo, string> = {
  ALFABETO: 'Alfabeto',
  PALAVRAS: 'Palavras',
  CALIBRADO: 'Calibrado',
};

const nomeSinal = (p: string) => p.replace(/_/g, ' ');
const pct = (c: number | null | undefined) =>
  c == null ? '' : `${Math.round(c * 100)}%`;

function BackIcon() {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
      <Path
        d="M15 5L8 12L15 19"
        stroke={PARCHMENT}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

function FlipIcon() {
  return (
    <Svg width={22} height={22} viewBox="0 0 24 24" fill="none">
      <Path
        d="M4 7.5H8L9.5 5.5H14.5L16 7.5H20V18.5H4V7.5Z"
        stroke={PARCHMENT}
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Path
        d="M9 13H15M13 11L15 13L13 15"
        stroke={PARCHMENT}
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

function Botao({
  texto,
  onPress,
  destaque,
  desativado,
}: {
  texto: string;
  onPress: () => void;
  destaque?: boolean;
  desativado?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={desativado}
      style={({ pressed }) => [
        styles.botao,
        destaque && styles.botaoDestaque,
        { opacity: desativado ? 0.4 : pressed ? 0.8 : 1 },
      ]}
    >
      <Text style={[styles.botaoTexto, destaque && styles.botaoTextoDestaque]}>
        {texto}
      </Text>
    </Pressable>
  );
}

type Props = { onBack: () => void };

export default function AoVivoScreen({ onBack }: Props) {
  const insets = useSafeAreaInsets();
  const [frontal, setFrontal] = useState(true);
  const device = useCameraDevice(frontal ? 'front' : 'back');
  const { hasPermission, requestPermission } = useCameraPermission();

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  const libres = useLibres({ cameraFrontal: frontal, ativo: hasPermission });
  const { estado, conexao, info } = libres;
  const modos = (estado?.modos ?? info?.modos ?? []) as Modo[];

  if (!hasPermission) {
    return (
      <View style={[styles.centro, { paddingTop: insets.top }]}>
        <Text style={styles.mensagem}>
          Precisamos da câmera para ver os sinais.
        </Text>
        <Botao
          texto="Permitir acesso à câmera"
          onPress={requestPermission}
          destaque
        />
        <View style={{ height: 12 }} />
        <Botao texto="Voltar" onPress={onBack} />
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={[styles.centro, { paddingTop: insets.top }]}>
        <ActivityIndicator color={GOLD} />
        <Text style={styles.mensagem}>
          Nenhuma câmera encontrada (no simulador do iOS isso é esperado).
        </Text>
        <Botao texto="Voltar" onPress={onBack} />
      </View>
    );
  }

  const textoConexao =
    conexao === 'conectado'
      ? 'conectado'
      : conexao === 'conectando'
      ? 'conectando…'
      : conexao === 'codigo-errado'
      ? 'código de acesso errado'
      : conexao === 'servidor-cheio'
      ? 'servidor cheio'
      : 'sem conexão';

  const cal = estado?.modo === 'CALIBRADO' ? estado.cal : undefined;
  const alfa = estado?.modo === 'ALFABETO' ? estado.alfa : undefined;
  const top = estado?.ultimo?.top ?? [];
  const aviso = estado?.aviso || cal?.msg || null;

  // "Como se faz?": abre o video do sinal no dicionario de Libras (navegador do celular)
  const comoSeFaz = () => {
    Linking.openURL(linkComoSeFaz(estado ?? null)).catch(() => {});
  };

  return (
    <View style={styles.tela}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        outputs={[libres.frameOutput]}
      />

      {/* topo: voltar, conexao, trocar camera */}
      <View style={[styles.topo, { paddingTop: insets.top + 8 }]}>
        <TouchableOpacity style={styles.icone} onPress={onBack}>
          <BackIcon />
        </TouchableOpacity>
        <View style={styles.status}>
          <View
            style={[
              styles.ponto,
              { backgroundColor: conexao === 'conectado' ? '#5FBF77' : RED },
            ]}
          />
          <Text style={styles.statusTexto}>
            {textoConexao}
            {conexao === 'conectado' && libres.msDeteccao
              ? ` · ${libres.msDeteccao} ms`
              : ''}
          </Text>
        </View>
        <TouchableOpacity
          style={styles.icone}
          onPress={() => setFrontal(f => !f)}
        >
          <FlipIcon />
        </TouchableOpacity>
      </View>

      {/* modos */}
      <View style={styles.modos}>
        {modos.map(m => (
          <Pressable
            key={m}
            onPress={() => libres.trocarModo(m)}
            style={[styles.modo, estado?.modo === m && styles.modoAtivo]}
          >
            <Text
              style={[
                styles.modoTexto,
                estado?.modo === m && styles.modoTextoAtivo,
              ]}
            >
              {NOME_MODO[m] ?? m}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* indicador da mao e palpite atual */}
      <View style={styles.meio} pointerEvents="none">
        {!libres.maoVisivel && conexao === 'conectado' && (
          <Text style={styles.semMao}>mostre a mão inteira para a câmera</Text>
        )}
        {alfa?.letra ? (
          <Text style={[styles.letra, !alfa.aceita && { opacity: 0.55 }]}>
            {alfa.letra.toUpperCase()}{' '}
            <Text style={styles.conf}>{pct(alfa.conf)}</Text>
          </Text>
        ) : null}
        {estado?.gravando && (
          <Text style={styles.gravando}>● lendo o sinal…</Text>
        )}
      </View>

      {/* "Ver como faz": o exemplo do sinal (maos e corpo) passando no canto */}
      {libres.exemplo && (
        <ExemploSinal exemplo={libres.exemplo} onFim={libres.fecharExemplo} />
      )}

      {/* painel de baixo: texto traduzido, ultimo sinal e acoes */}
      <View style={[styles.painel, { paddingBottom: insets.bottom + 12 }]}>
        {(libres.erro || aviso) && (
          <Pressable onPress={libres.limparErro}>
            <Text
              style={[
                styles.aviso,
                (libres.erro || cal?.msg_ruim) && { color: RED },
              ]}
            >
              {libres.erro || aviso}
            </Text>
          </Pressable>
        )}

        {cal?.calibrando && cal.palavra ? (
          <View style={styles.calibracao}>
            <Text style={styles.calTitulo}>CALIBRANDO</Text>
            <Text style={styles.calPalavra}>{nomeSinal(cal.palavra)}</Text>
            <Pressable onPress={comoSeFaz} hitSlop={8}>
              <Text style={styles.link}>Como se faz? (vídeo no dicionário)</Text>
            </Pressable>
            <Text style={styles.calProgresso}>
              {cal.n ?? 0} de {cal.rep} exemplos · palavra {(cal.idx ?? 0) + 1}{' '}
              de {cal.total}
            </Text>
            <View style={styles.linhaBotoes}>
              <Botao texto="‹" onPress={() => libres.comando('p')} />
              <Botao texto="Ver como faz" onPress={() => libres.comando('v')} />
              <Botao texto="Desfazer" onPress={() => libres.comando('d')} />
              <Botao texto="›" onPress={() => libres.comando('n')} />
            </View>
            <View style={styles.linhaBotoes}>
              <Botao
                texto="Refazer palavra"
                onPress={() => libres.comando('r')}
              />
              <Botao
                texto="Terminar"
                onPress={() => libres.comando('k')}
                destaque
              />
            </View>
          </View>
        ) : (
          <>
            <Text style={styles.rotulo}>TRADUÇÃO</Text>
            <ScrollView
              style={styles.textoCaixa}
              contentContainerStyle={{ flexGrow: 1 }}
            >
              <Text style={styles.texto}>
                {estado?.texto || 'Faça um sinal…'}
              </Text>
            </ScrollView>
            {top.length > 0 && (
              <Text style={styles.top} numberOfLines={1}>
                {top
                  .slice(0, 3)
                  .map(([p, c]) => `${nomeSinal(p)} ${pct(c)}`)
                  .join('   ·   ')}
              </Text>
            )}
            <View style={styles.linhaBotoes}>
              <Botao texto="Apagar" onPress={() => libres.comando('apagar')} />
              <Botao texto="Limpar" onPress={() => libres.comando('limpar')} />
              <Botao texto="Como se faz?" onPress={comoSeFaz} />
              {estado?.modo === 'CALIBRADO' && (
                <Botao
                  texto={cal?.tem_reconhecedor ? 'Calibrar mais' : 'Calibrar'}
                  onPress={() => libres.comando('k')}
                  destaque
                />
              )}
            </View>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  tela: { flex: 1, backgroundColor: INK },
  centro: {
    flex: 1,
    backgroundColor: INK,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  mensagem: {
    color: PARCHMENT,
    fontSize: 16,
    textAlign: 'center',
    marginVertical: 16,
  },

  topo: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
  },
  icone: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: 'rgba(18,23,43,0.6)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  status: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(18,23,43,0.6)',
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  ponto: { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  statusTexto: { color: PARCHMENT, fontSize: 13 },

  modos: {
    position: 'absolute',
    top: 110,
    alignSelf: 'center',
    flexDirection: 'row',
    backgroundColor: 'rgba(18,23,43,0.6)',
    borderRadius: 20,
    padding: 4,
  },
  modo: { paddingHorizontal: 14, paddingVertical: 7, borderRadius: 16 },
  modoAtivo: { backgroundColor: GOLD },
  modoTexto: { color: PARCHMENT, fontSize: 13, fontWeight: '600' },
  modoTextoAtivo: { color: INK },

  meio: {
    position: 'absolute',
    top: 170,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  semMao: {
    color: PARCHMENT,
    backgroundColor: 'rgba(18,23,43,0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    fontSize: 13,
    overflow: 'hidden',
  },
  letra: { color: GOLD, fontSize: 72, fontWeight: '800', marginTop: 12 },
  conf: { fontSize: 20, color: PARCHMENT },
  gravando: { color: GOLD, marginTop: 8, fontSize: 14, fontWeight: '600' },

  painel: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(18,23,43,0.92)',
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    paddingHorizontal: 20,
    paddingTop: 16,
  },
  aviso: { color: GOLD, fontSize: 13, marginBottom: 8 },
  rotulo: { color: GOLD, fontSize: 11, letterSpacing: 2, fontWeight: '700' },
  textoCaixa: { maxHeight: 110, marginTop: 6 },
  texto: { color: PARCHMENT, fontSize: 26, fontWeight: '600' },
  top: { color: 'rgba(245,241,232,0.6)', fontSize: 12, marginTop: 8 },
  linhaBotoes: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginTop: 12,
    gap: 8,
  },
  botao: {
    borderWidth: 1,
    borderColor: 'rgba(245,241,232,0.35)',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  botaoDestaque: { backgroundColor: GOLD, borderColor: GOLD },
  botaoTexto: { color: PARCHMENT, fontSize: 14, fontWeight: '600' },
  botaoTextoDestaque: { color: INK },

  calibracao: { alignItems: 'center' },
  calTitulo: { color: GOLD, fontSize: 11, letterSpacing: 2, fontWeight: '700' },
  calPalavra: {
    color: PARCHMENT,
    fontSize: 32,
    fontWeight: '800',
    marginTop: 4,
  },
  calProgresso: { color: 'rgba(245,241,232,0.7)', fontSize: 13, marginTop: 4 },
  link: {
    color: GOLD,
    fontSize: 13,
    textDecorationLine: 'underline',
    marginTop: 4,
  },
});
