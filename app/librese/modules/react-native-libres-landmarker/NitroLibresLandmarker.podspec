require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "NitroLibresLandmarker"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => min_ios_version_supported, :visionos => 1.0 }
  s.source       = { :git => "https://github.com/margelo/nitro.git", :tag => "#{s.version}" }

  s.source_files = [
    # Implementation (Swift)
    "ios/**/*.{swift}",
    # Autolinking/Registration (Objective-C++)
    "ios/**/*.{m,mm}",
    # Implementation (C++ objects)
    "cpp/**/*.{hpp,cpp}",
  ]

  load 'nitrogen/generated/ios/NitroLibresLandmarker+autolinking.rb'
  add_nitrogen_files(s)

  # modelos do MediaPipe (maos e corpo), os mesmos da versao web
  s.resource_bundles = { 'LibresLandmarkerModelos' => ['assets/*.task'] }
  s.frameworks = ['Accelerate', 'CoreVideo']

  # mesma familia da @mediapipe/tasks-vision 0.10.32 da versao web
  s.dependency 'MediaPipeTasksVision', '0.10.33'
  # o MediaPipe vem como xcframework: o CocoaPods so poe a pasta do .xcframework no caminho
  # de busca, e o Swift precisa da pasta onde o Xcode extrai o .framework
  # (junta com o que o add_nitrogen_files ja configurou, como o SWIFT_OBJC_INTEROP_MODE)
  s.pod_target_xcconfig = (s.attributes_hash['pod_target_xcconfig'] || {}).merge({
    'FRAMEWORK_SEARCH_PATHS' => '$(inherited) "${PODS_XCFRAMEWORKS_BUILD_DIR}/MediaPipeTasksCommon" "${PODS_XCFRAMEWORKS_BUILD_DIR}/MediaPipeTasksVision"'
  })
  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  install_modules_dependencies(s)
end
