import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
  useVideoOutput,
} from 'react-native-vision-camera';

import { Svg, Path } from 'react-native-svg';

function HouseIcon() {
  return (
    <Svg width={26} height={26} viewBox="0 0 24 24" fill="none">
      <Path
        d="M3 10.5L12 3L21 10.5"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
      <Path
        d="M5.5 9.5V19.5C5.5 20.0523 5.94772 20.5 6.5 20.5H17.5C18.0523 20.5 18.5 20.0523 18.5 19.5V9.5"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
    </Svg>
  );
}

function FlipCameraIcon() {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none">
      <Path
        d="M4 7.5H8L9.5 5.5H14.5L16 7.5H20V18.5H4V7.5Z"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
      <Path
        d="M9.5 12.5C9.5 11.1193 10.6193 10 12 10C12.9319 10 13.7454 10.5093 14.1761 11.2649"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
      <Path
        d="M14.5 13.5C14.5 14.8807 13.3807 16 12 16C11.0681 16 10.2546 15.4907 9.82388 14.7351"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
      <Path
        d="M14 10L14.3 11.5L15.8 11.2"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
      <Path
        d="M10 16L9.7 14.5L8.2 14.8"
        stroke="#fff"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
    </Svg>
  );
}

type CapturedFile = {
  path: string;
};

type Props = {
  onBack?: () => void;
  onMediaReady?: (path: string, type: 'photo' | 'video') => void;
};

export default function CameraScreen({ onBack, onMediaReady }: Props) {
  const camera = useRef<Camera>(null);
  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>('back');
  const device = useCameraDevice(cameraPosition);
  const { hasPermission, requestPermission } = useCameraPermission();

  // API v5 da vision-camera: capturas ficam nos "outputs", não mais no ref da Camera
  const photoOutput = usePhotoOutput();
  const videoOutput = useVideoOutput({ enableAudio: false, fileType: 'mp4' });
  const recorderRef = useRef<any>(null);
  const isRecordingRef = useRef(false);

  const [photo, setPhoto] = useState<CapturedFile | null>(null);
  const [isTaking, setIsTaking] = useState(false);

  const [video, setVideo] = useState<CapturedFile | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(5);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const countdownIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const longPressTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isLongPressActiveRef = useRef(false);

  const MAX_RECORDING_SECONDS = 5;
  const LONG_PRESS_THRESHOLD_MS = 350;

  // Pede a permissão de câmera assim que a tela monta
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const clearTimers = () => {
    if (stopTimeoutRef.current) clearTimeout(stopTimeoutRef.current);
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    if (longPressTimeoutRef.current) clearTimeout(longPressTimeoutRef.current);
  };

  // Limpa os timers se a tela for desmontada no meio de uma gravação
  useEffect(() => {
    return () => clearTimers();
  }, []);

  const handleCapture = async () => {
    try {
      setIsTaking(true);
      const { filePath } = await photoOutput.capturePhotoToFile({}, {});
      setPhoto({ path: filePath });
    } catch (error: any) {
      console.error('Erro ao capturar foto:', error);
      Alert.alert(
        'Erro ao tirar foto',
        `Detalhes: ${error?.message ?? String(error)}`,
      );
    } finally {
      setIsTaking(false);
    }
  };

  const handleStartRecording = async () => {
    if (isRecording) return;

    setSecondsLeft(MAX_RECORDING_SECONDS);

    try {
      // Na v5, cada gravação precisa de um Recorder novo (não dá pra reaproveitar)
      const recorder = await videoOutput.createRecorder({});
      recorderRef.current = recorder;
      isRecordingRef.current = true;
      setIsRecording(true);

      await recorder.startRecording(
        (path: string) => {
          // onRecordingFinished
          clearTimers();
          isRecordingRef.current = false;
          setIsRecording(false);
          setVideo({ path });
        },
        (error: any) => {
          // onRecordingError
          console.error('Erro ao gravar vídeo:', error);
          clearTimers();
          isRecordingRef.current = false;
          setIsRecording(false);
          Alert.alert(
            'Erro ao gravar',
            `Detalhes: ${error?.message ?? String(error)}`,
          );
        },
      );

      countdownIntervalRef.current = setInterval(() => {
        setSecondsLeft((s) => Math.max(s - 1, 0));
      }, 1000);

      stopTimeoutRef.current = setTimeout(() => {
        handleStopRecording();
      }, MAX_RECORDING_SECONDS * 1000);
    } catch (error: any) {
      // Evita que uma falha aqui derrube o app inteiro
      console.error('Falha ao iniciar gravação:', error);
      setIsRecording(false);
      Alert.alert(
        'Erro ao gravar',
        `Detalhes: ${error?.message ?? String(error)}`,
      );
    }
  };

  const handleStopRecording = async () => {
    clearTimers();
    if (!recorderRef.current || !isRecordingRef.current) return;

    // Marca como "não gravando mais" já aqui, para evitar que o timer de 5s
    // e o soltar do dedo tentem parar a mesma gravação duas vezes
    isRecordingRef.current = false;

    try {
      await recorderRef.current.stopRecording();
    } catch (error) {
      console.error('Erro ao parar gravação:', error);
    }
  };

  const handleRetakeVideo = () => {
    setVideo(null);
  };

  const handleRetake = () => {
    setPhoto(null);
  };

  const handleFlipCamera = () => {
    // Evita trocar a câmera no meio de uma gravação
    if (isRecordingRef.current) return;
    setCameraPosition((prev) => (prev === 'back' ? 'front' : 'back'));
  };

  // Toque rápido = foto. Segurar (mais que o limite abaixo) = começa a gravar vídeo.
  const handlePressIn = () => {
    isLongPressActiveRef.current = false;
    longPressTimeoutRef.current = setTimeout(() => {
      isLongPressActiveRef.current = true;
      handleStartRecording();
    }, LONG_PRESS_THRESHOLD_MS);
  };

  const handlePressOut = () => {
    if (longPressTimeoutRef.current) {
      clearTimeout(longPressTimeoutRef.current);
      longPressTimeoutRef.current = null;
    }

    if (isLongPressActiveRef.current) {
      // Estava gravando: soltar o dedo para
      handleStopRecording();
      isLongPressActiveRef.current = false;
    } else {
      // Foi um toque rápido: tira foto
      handleCapture();
    }
  };

  // --- Estados de carregamento / erro ---

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>
          Precisamos da sua permissão para usar a câmera.
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Permitir acesso à câmera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (device == null) {
    // No simulador isso vai cair aqui, já que não existe device de câmera real
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" />
        <Text style={styles.message}>
          Nenhuma câmera encontrada. Se estiver no Simulador, isso é
          esperado — teste em um iPhone físico.
        </Text>
        {onBack && (
          <TouchableOpacity style={styles.homeButton} onPress={onBack}>
            <HouseIcon />
          </TouchableOpacity>
        )}

        {/* Botão aqui só para conferir o layout no simulador —
            sem device de câmera real, ele não captura nada */}
        <View style={styles.captureFooter}>
          <Text style={styles.hint}>Toque: foto · Segure: vídeo</Text>
          <View style={styles.captureButton}>
            <View style={styles.captureButtonInner} />
          </View>
        </View>
      </View>
    );
  }

  // --- Preview da foto capturada ---

  if (photo) {
    return (
      <View style={styles.flex}>
        <Image
          source={{ uri: `file://${photo.path}` }}
          style={styles.flex}
          resizeMode="contain"
        />
        <View style={styles.footer}>
          <TouchableOpacity style={styles.button} onPress={handleRetake}>
            <Text style={styles.buttonText}>Tirar outra</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.primaryButton]}
            onPress={() => onMediaReady?.(photo.path, 'photo')}
          >
            <Text style={styles.buttonText}>Usar essa foto</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Preview do vídeo gravado ---

  if (video) {
    return (
      <View style={[styles.flex, styles.centered]}>
        <Text style={styles.message}>
          Vídeo gravado ({MAX_RECORDING_SECONDS}s ou menos) ✓
        </Text>
        <View style={styles.footer}>
          <TouchableOpacity style={styles.button} onPress={handleRetakeVideo}>
            <Text style={styles.buttonText}>Gravar de novo</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, styles.primaryButton]}
            onPress={() => onMediaReady?.(video.path, 'video')}
          >
            <Text style={styles.buttonText}>Usar esse vídeo</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Preview da câmera ao vivo ---

  return (
    <View style={styles.flex}>
      <Camera
        ref={camera}
        style={styles.flex}
        device={device}
        isActive={true}
        outputs={[photoOutput, videoOutput]}
      />

      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>{secondsLeft}s</Text>
        </View>
      )}

      {onBack && (
        <TouchableOpacity style={styles.homeButton} onPress={onBack}>
          <HouseIcon />
        </TouchableOpacity>
      )}

      <TouchableOpacity
        style={styles.flipButton}
        onPress={handleFlipCamera}
        disabled={isRecording}
      >
        <FlipCameraIcon />
      </TouchableOpacity>

      <View style={styles.captureFooter}>
        <Text style={styles.hint}>Toque: foto · Segure: vídeo</Text>
        <TouchableOpacity
          style={[styles.captureButton, isRecording && styles.captureButtonRecording]}
          onPressIn={handlePressIn}
          onPressOut={handlePressOut}
          disabled={isTaking}
        >
          {isTaking ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <View
              style={[
                styles.captureButtonInner,
                isRecording && styles.captureButtonInnerRecording,
              ]}
            />
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
    backgroundColor: '#000',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#000',
  },
  message: {
    color: '#fff',
    textAlign: 'center',
    marginBottom: 16,
    fontSize: 16,
  },
  footer: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 16,
  },
  captureFooter: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 14,
  },
  hint: {
    color: 'rgba(255, 255, 255, 0.75)',
    fontSize: 13,
    fontWeight: '500',
  },
  button: {
    backgroundColor: '#333',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  primaryButton: {
    backgroundColor: '#0a84ff',
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
  },
  captureButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonRecording: {
    borderColor: '#ff3b30',
  },
  captureButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#fff',
  },
  captureButtonInnerRecording: {
    backgroundColor: '#ff3b30',
    width: 30,
    height: 30,
    borderRadius: 8,
  },
  homeButton: {
    position: 'absolute',
    top: 60,
    left: 20,
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  flipButton: {
    position: 'absolute',
    top: 60,
    right: 20,
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordingIndicator: {
    position: 'absolute',
    top: 60,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#ff3b30',
  },
  recordingText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
  },
});