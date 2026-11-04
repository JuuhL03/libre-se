import React, { useState } from 'react';
import { StatusBar, StyleSheet, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import HomeScreen from './src/screens/HomeScreen';
import CameraScreen from './src/screens/CameraScreen';
import TranslationScreen, {
  CapturedMedia,
} from './src/screens/TranslationScreen';
import AoVivoScreen from './src/screens/AoVivoScreen';

// 'aovivo': traducao ao vivo pela IA do Libre-se (WebSocket do servidor, mesma da versao web)
// 'camera' / 'translating': fluxo antigo de foto/video (resposta ainda simulada)
type Screen = 'home' | 'aovivo' | 'camera' | 'translating';

function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const [media, setMedia] = useState<CapturedMedia | null>(null);

  const handleMediaReady = (path: string, type: 'photo' | 'video') => {
    setMedia({ path, type });
    setScreen('translating');
  };

  const handleDone = () => {
    setMedia(null);
    setScreen('home');
  };

  return (
    <SafeAreaProvider>
      <View style={styles.container}>
        <StatusBar barStyle="light-content" />

        {screen === 'home' && (
          <HomeScreen onTranslatePress={() => setScreen('aovivo')} />
        )}

        {screen === 'aovivo' && (
          <AoVivoScreen onBack={() => setScreen('home')} />
        )}

        {screen === 'camera' && (
          <CameraScreen
            onBack={() => setScreen('home')}
            onMediaReady={handleMediaReady}
          />
        )}

        {screen === 'translating' && (
          <TranslationScreen media={media} onDone={handleDone} />
        )}
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default App;
