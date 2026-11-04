module.exports = {
  presets: ['module:@react-native/babel-preset'],
  // worklets: o useFrameOutput da VisionCamera roda o onFrame na thread da camera
  plugins: ['react-native-worklets/plugin'],
};
