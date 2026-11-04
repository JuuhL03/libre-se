import React from 'react';
import { View, Text, StyleSheet, StatusBar, Pressable, Image } from 'react-native';

type Props = {
  onTranslatePress: () => void;
};

const INK = '#12172B';
const PARCHMENT = '#F5F1E8';
const GOLD = '#D4A24C';

// Cantos de mira, ecoam o viewfinder da câmera e emolduram a tela
function ViewfinderCorners() {
  return (
    <>
      <View style={[styles.corner, styles.cornerTL]} />
      <View style={[styles.corner, styles.cornerTR]} />
      <View style={[styles.corner, styles.cornerBL]} />
      <View style={[styles.corner, styles.cornerBR]} />
    </>
  );
}

// Logo real do app (arquivo de imagem)
function Logo() {
  return (
    <Image
      source={require('../assets/images/logo.png')}
      style={styles.logo}
      resizeMode="contain"
    />
  );
}

// Botão sólido com um leve brilho no topo (sem SVG, sem Animated —
// evita o bug de renderização visto ao combinar os três)
function TranslateButton({ onPress }: { onPress: () => void }) {
  return (
    <View style={styles.buttonShadowWrap}>
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [
          styles.button,
          {
            opacity: pressed ? 0.88 : 1,
            transform: [{ scale: pressed ? 0.96 : 1 }],
          },
        ]}
      >
        <View style={styles.buttonHighlight} />
        <Text style={styles.buttonText}>TRADUZIR</Text>
        <Text style={styles.buttonArrow}>→</Text>
      </Pressable>
    </View>
  );
}

export default function HomeScreen({ onTranslatePress }: Props) {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <ViewfinderCorners />

      <View style={styles.content}>
        <Logo />

        <Text style={styles.eyebrow}>TRADUÇÃO POR CÂMERA</Text>

        <Text style={styles.wordmark}>LIBRE-SE</Text>

        <View style={styles.divider} />

        <Text style={styles.tagline}>
          Aponte a câmera e faça o gesto{'\n'}e leia na hora, em português.
        </Text>
      </View>

      <TranslateButton onPress={onTranslatePress} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: INK,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  content: {
    alignItems: 'center',
    marginBottom: 56,
  },
  logo: {
    width: 88,
    height: 88,
    marginBottom: 22,
  },
  eyebrow: {
    color: GOLD,
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 3,
    marginBottom: 14,
  },
  wordmark: {
    color: PARCHMENT,
    fontSize: 42,
    fontWeight: '800',
    letterSpacing: 6,
  },
  divider: {
    width: 32,
    height: 1.5,
    backgroundColor: 'rgba(212, 162, 76, 0.5)',
    marginTop: 20,
    marginBottom: 20,
  },
  tagline: {
    color: 'rgba(245, 241, 232, 0.65)',
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
    maxWidth: 280,
  },
  buttonShadowWrap: {
    borderRadius: 4,
    shadowColor: GOLD,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 6,
    marginBottom: 12,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 17,
    paddingHorizontal: 44,
    borderRadius: 4,
    overflow: 'hidden',
    backgroundColor: GOLD,
  },
  buttonHighlight: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '50%',
    backgroundColor: 'rgba(255, 255, 255, 0.18)',
  },
  buttonText: {
    color: INK,
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 2,
  },
  buttonArrow: {
    color: INK,
    fontSize: 16,
    fontWeight: '700',
  },
  corner: {
    position: 'absolute',
    width: 28,
    height: 28,
    borderColor: 'rgba(212, 162, 76, 0.4)',
  },
  cornerTL: {
    top: 56,
    left: 24,
    borderTopWidth: 1.5,
    borderLeftWidth: 1.5,
  },
  cornerTR: {
    top: 56,
    right: 24,
    borderTopWidth: 1.5,
    borderRightWidth: 1.5,
  },
  cornerBL: {
    bottom: 56,
    left: 24,
    borderBottomWidth: 1.5,
    borderLeftWidth: 1.5,
  },
  cornerBR: {
    bottom: 56,
    right: 24,
    borderBottomWidth: 1.5,
    borderRightWidth: 1.5,
  },
});