import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Animated, Easing, Pressable } from 'react-native';

const INK = '#12172B';
const PARCHMENT = '#F5F1E8';
const GOLD = '#D4A24C';

export type CapturedMedia = {
  path: string;
  type: 'photo' | 'video';
};

type Props = {
  media: CapturedMedia | null;
  onDone: () => void;
};

// 3 pontinhos que sobem e descem alternadamente, tipo indicador de "digitando"
function BouncingDots() {
  const dot1 = useRef(new Animated.Value(0)).current;
  const dot2 = useRef(new Animated.Value(0)).current;
  const dot3 = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const bounce = (dot: Animated.Value) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(dot, {
            toValue: -8,
            duration: 300,
            easing: Easing.out(Easing.quad),
            useNativeDriver: true,
          }),
          Animated.timing(dot, {
            toValue: 0,
            duration: 300,
            easing: Easing.in(Easing.quad),
            useNativeDriver: true,
          }),
          Animated.delay(200),
        ]),
      );

    const anim1 = bounce(dot1);
    const anim2 = bounce(dot2);
    const anim3 = bounce(dot3);

    anim1.start();
    const t2 = setTimeout(() => anim2.start(), 150);
    const t3 = setTimeout(() => anim3.start(), 300);

    return () => {
      anim1.stop();
      anim2.stop();
      anim3.stop();
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [dot1, dot2, dot3]);

  return (
    <View style={styles.dotsRow}>
      <Animated.View style={[styles.dot, { transform: [{ translateY: dot1 }] }]} />
      <Animated.View style={[styles.dot, { transform: [{ translateY: dot2 }] }]} />
      <Animated.View style={[styles.dot, { transform: [{ translateY: dot3 }] }]} />
    </View>
  );
}

// Simulação temporária da resposta do backend — remover quando o
// FastAPI estiver pronto e substituir pela chamada real (ver useEffect abaixo)
function simulateBackendResponse(): Promise<string> {
  return new Promise((resolve) => {
    setTimeout(() => resolve('Obrigado'), 2500);
  });
}

export default function TranslationScreen({ media, onDone }: Props) {
  const [isLoading, setIsLoading] = useState(true);
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function translate() {
      setIsLoading(true);
      setErrorMessage(null);
      setTranslatedText(null);

      try {
        // ------------------------------------------------------------------
        // TODO: quando o backend Python estiver pronto, trocar o bloco abaixo
        // por algo assim:
        //
        // const formData = new FormData();
        // formData.append('file', {
        //   uri: `file://${media?.path}`,
        //   name: media?.type === 'photo' ? 'captura.jpg' : 'captura.mp4',
        //   type: media?.type === 'photo' ? 'image/jpeg' : 'video/mp4',
        // } as any);
        //
        // const response = await fetch('http://SEU_BACKEND_AQUI/traduzir', {
        //   method: 'POST',
        //   body: formData,
        // });
        // const json = await response.json();
        // const texto = json.texto_traduzido;
        // ------------------------------------------------------------------

        const texto = await simulateBackendResponse();

        if (!cancelled) {
          setTranslatedText(texto);
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage('Não foi possível traduzir. Tente novamente.');
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    if (media) {
      translate();
    }

    return () => {
      cancelled = true;
    };
  }, [media]);

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        {isLoading && (
          <>
            <Text style={styles.loadingText}>Traduzindo</Text>
            <BouncingDots />
          </>
        )}

        {!isLoading && translatedText && (
          <>
            <Text style={styles.eyebrow}>TRADUÇÃO</Text>
            <Text style={styles.resultText}>{translatedText}</Text>
          </>
        )}

        {!isLoading && errorMessage && (
          <Text style={styles.errorText}>{errorMessage}</Text>
        )}
      </View>

      {!isLoading && (
        <Pressable
          onPress={onDone}
          style={({ pressed }) => [
            styles.button,
            { opacity: pressed ? 0.88 : 1, transform: [{ scale: pressed ? 0.96 : 1 }] },
          ]}
        >
          <View style={styles.buttonHighlight} />
          <Text style={styles.buttonText}>TRADUZIR OUTRO</Text>
        </Pressable>
      )}
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
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: PARCHMENT,
    fontSize: 22,
    fontWeight: '600',
    marginBottom: 18,
  },
  dotsRow: {
    flexDirection: 'row',
    gap: 10,
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: GOLD,
  },
  eyebrow: {
    color: GOLD,
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 3,
    marginBottom: 16,
  },
  resultText: {
    color: PARCHMENT,
    fontSize: 40,
    fontWeight: '800',
    textAlign: 'center',
  },
  errorText: {
    color: '#ff8a80',
    fontSize: 16,
    textAlign: 'center',
    maxWidth: 280,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 17,
    paddingHorizontal: 32,
    borderRadius: 4,
    overflow: 'hidden',
    backgroundColor: GOLD,
    marginBottom: 12,
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
});