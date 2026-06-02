# Integración y deployment — de laptop a app / vehículo

> ¿Qué tan integrable es el MVP actual (Python + OpenCV + MediaPipe FaceMesh en laptop con webcam) a una app móvil o a un sistema embebido vehicular, y cómo se haría?

---

## TL;DR

- **App Android/iOS con MediaPipe Tasks es el path más corto y viable**: 2-3 semanas de trabajo, latencia de 30-70 ms/frame en un Pixel 9 Pro usando la SDK oficial, wow-factor alto para feria.
- **Embebido automotriz real (Jetson Orin, Snapdragon Ride/Cockpit 8295, TDA4VM, S32V) está fuera de scope universitario**: cuesta miles de dólares, requiere toolchains cerrados, cámara IR calificada, integración CAN, y certificaciones (ISO 26262, ISO/SAE 21434). No es una tarde de hacking.
- **RGB es un tapón mortal para DMS real**: MediaPipe FaceMesh está entrenado en RGB y falla de noche. El estándar automotriz es cámara NIR 940 nm, y MediaPipe requeriría fine-tuning con datasets IR. MVP pragmático: declarar "solo diurno".
- **Prototipo barato de after-market**: Raspberry Pi 5 + Coral TPU (~180 USD) corre detección facial a ~20 fps; sirve como demo "caja en el coche" pero no es automotive-grade.
- **Lo fácil viaja, lo difícil no**: el pipeline (EAR, MAR, PERCLOS, head pose) es solo aritmética sobre landmarks, se porta trivialmente. Lo que *no* viaja es el modelo cuando cambias el sensor (RGB → IR).
- **Recomendación feria**: laptop + webcam como demo principal, y opcionalmente un APK Android con MediaPipe Tasks como "lo mismo en tu bolsillo". No intentar Jetson para la feria salvo que ya tengas la board y un mes libre.

---

## Stack actual vs. target

### Actual (MVP laptop)

```
+-----------+   USB   +----------+   Python   +-------------------+   cv2.imshow
| Webcam    |-------->| OpenCV   |----------->| MediaPipe         |----------------> GUI ventana
| RGB 720p  |         | VideoCap |            | FaceMesh (Py)     |     EAR/MAR/PERCLOS
+-----------+         +----------+            +-------------------+     prints en consola
```

### Target A — App móvil

```
+------------+  MIPI  +----------------+  +---------------------+  +-------------------+
| Cam frontal|------->| Camera2/AVFound|->| MediaPipe Tasks     |->| UI nativa +       |
| RGB selfie |        | ImageReader    |  | Face Landmarker     |  | alerta audio/haptic|
+------------+        +----------------+  +---------------------+  +-------------------+
```

### Target B — Embebido in-cabin

```
+-------------+ GMSL/ +------------------+   +------------------+   +-----------+
| Cam NIR 940 |------>| SoC auto (Jetson |-->| DMS pipeline     |-->| CAN bus   |
| 1-2 MP GS   | MIPI  | / 8295 / TDA4VM) |   | (TFLite/TensorRT)|   | MCU/ECU   |
+-------------+       +------------------+   +------------------+   +-----------+
       ^ LED IR                                                            |
       | pulsado                                                           v
                                                                 cluster / HMI / chime
```

---

## Opción A — App móvil (Android / iOS)

### Runtimes disponibles

#### 1. MediaPipe Tasks (Google AI Edge) — la opción por defecto

- API oficial Android/iOS/Web con `FaceLandmarker` que devuelve los **478 landmarks** (mismos del MVP) más **blendshapes** (útiles para parpadeo directamente, sin EAR hand-rolled) y matriz de transformación facial (head pose ya resuelto).
- Modos: `IMAGE`, `VIDEO`, `LIVE_STREAM`. En `VIDEO`/`LIVE_STREAM` reutiliza tracking entre frames, bajando latencia.
- Empaquetado: `.task` bundle, ~3-4 MB para face landmarker.
- **Latencia medida (reportada por usuarios, 2025)**: ~30-70 ms/frame en Pixel 9 Pro en la app Android, ~15-20 ms en WASM en desktop. Hay un issue abierto notando que la SDK es más lenta que correr el `.tflite` directamente con el benchmark tool (probablemente por overhead de copias y scheduling).

#### 2. TensorFlow Lite (LiteRT) + delegates

- Cargar directamente el `face_landmark.tflite` y alimentar manualmente ROIs del detector.
- Delegates Android: **GPU**, **NNAPI**, **Hexagon** (SoCs Qualcomm). El GPU delegate suele acelerar 2-5x en modelos con convs pesadas; en modelos pequeños como face landmark, el overhead de copia a GPU puede comerse la ganancia.
- Pros: control fino, puedes cuantizar a INT8 y mezclar con otros modelos.
- Cons: reimplementas el pre/post-processing que MediaPipe ya te da gratis.

#### 3. Core ML (iOS) + Vision framework

- `VNDetectFaceLandmarksRequest` da 76 puntos (suficiente para EAR/MAR, insuficiente para mallas densas de gaze).
- Core ML + Neural Engine (A14+) es lo más rápido en iPhone, pero no tienes los 478 puntos de MediaPipe a menos que conviertas el tflite → Core ML (`coremltools`).
- Startup latency del Core ML delegate de TFLite: 200-400 ms reportado en modelos pequeños (se paga una vez).

#### 4. ONNX Runtime Mobile

- Útil si el equipo ya usa ONNX. Para face landmark no hay ventaja clara vs. TFLite/MediaPipe; más bien lo contrario: menos optimizaciones específicas para estos modelos.

### Tabla comparativa (face landmark, gama media 2024-2026)

| Runtime                  | Dispositivo          | Latencia /frame | Notas                                              |
|--------------------------|----------------------|-----------------|----------------------------------------------------|
| MediaPipe Tasks          | Pixel 9 Pro          | ~30-70 ms       | Reportado en issue #5872 del repo, SDK 0.10.21     |
| MediaPipe Tasks (WASM)   | Laptop desktop       | ~15-20 ms       | Referencia de baseline                             |
| MediaPipe Tasks          | Pixel 6 / 6a         | sin benchmark publicado, estimación: 20-40 ms (Tensor G1/G2, similar a G4) |
| MediaPipe Tasks          | iPhone SE 3 (A15)    | sin benchmark publicado, estimación: 10-25 ms (Neural Engine) |
| MediaPipe Tasks          | Samsung A54 (Exynos 1380) | sin benchmark publicado, estimación: 40-80 ms |
| TFLite CPU (XNNPACK)     | Pixel 6              | sin benchmark público reciente para face_landmark  |
| TFLite GPU delegate      | Snapdragon 8 Gen 2   | variable; en modelos pequeños a veces peor que CPU |
| Vision framework (iOS)   | iPhone 12+           | <10 ms detección de cara, pero 76 landmarks, no 478|

Referencias: face mesh en Pixel 6 clásicamente se reportó en 5-15 ms con el pipeline nativo de MediaPipe (C++ + GL), pero con la SDK Tasks de alto nivel la latencia real observada sube a 30-70 ms por el overhead de Java/Kotlin wrappers.

### Consideraciones prácticas

- **Cámara**: usar la **frontal (selfie)** porque apunta al conductor. Fija orientación `portrait` o `landscape` según montaje del celular.
- **Soporte físico**: clip en ventilación o parabrisas, alineado a la cara del conductor (~40-70 cm). Requiere tutorial de instalación; mal posicionamiento = ruido en head pose.
- **Batería y térmica**: inferencia continua a 30 fps drena ~15-25% de batería/hora y calienta el SoC. Mitigaciones: bajar a 10-15 fps, requerir cargador conectado (razonable para uso automotriz), usar `LIVE_STREAM` mode con tracking.
- **Permisos**: `CAMERA` runtime permission en Android, `NSCameraUsageDescription` en iOS. El permiso es sensible para la store review: documentar que el video no sale del dispositivo.
- **Background**: iOS y Android restringen cámara en background. Necesitas `foreground service` (Android) o mantener la app en primer plano (iOS). Para DMS real esto choca con que el conductor también quiera navegación en pantalla.

---

## Opción B — Dispositivo embebido / in-cabin

### Hardware automotriz y semi-automotriz

| SoC / Plataforma               | TOPS (INT8) | Precio dev kit | Stack SW                            | Automotive grade |
|--------------------------------|-------------|----------------|-------------------------------------|------------------|
| NVIDIA Jetson Orin Nano Super  | 67          | ~249 USD       | JetPack, TensorRT, CUDA, DeepStream | No (industrial)  |
| NVIDIA Jetson Orin NX 16GB     | ~100        | ~700-900 USD   | JetPack, TensorRT                   | No (industrial)  |
| NVIDIA DRIVE Orin              | 254         | >10k USD       | DriveOS                             | Sí (ASIL-D island) |
| Qualcomm Snapdragon Cockpit 8295 | 30        | dev kit Lantronix SA8295P ~varios miles | QNX/Android Auto, SNPE | Sí (AEC-Q100)  |
| Qualcomm Snapdragon Ride Elite | >100 (claim)| contacto OEM   | Qualcomm AI Stack                   | Sí               |
| NXP S32V234                    | ~1 (APEX-2) | ~500-1k USD    | Vision SDK, APEX-CV                 | Sí (ASIL-B)      |
| NXP S32G                       | network/MCU oriented, no es para vision pesada | | | Sí (ASIL-D) |
| TI TDA4VM (Jacinto 7)          | 8 (C7xDSP + MMA) | SK-TDA4VM ~400 USD | TI Edge AI, TIDL, Vision Apps | Sí (hasta ASIL-D/SIL-3) |
| Renesas R-Car V4H              | ~34         | contacto       | R-Car SDK                           | Sí               |
| Raspberry Pi 5 + Coral USB     | 4 (Coral)   | ~80+100 USD    | libedgetpu, TFLite                  | No               |

Notas:

- **Jetson Orin Nano Super** es el sweet spot para un prototipo ambicioso: 67 TOPS a 249 USD, corre MediaPipe/TFLite/PyTorch sin dramas, tiene CSI-2 para cámaras. Pero **no es automotive grade** (no soporta rangos térmicos del coche ni ASIL).
- **Snapdragon Cockpit 8295** es el rey actual de cockpit digital (BMW Neue Klasse, muchos OEMs chinos). 30 TOPS suficientes para DMS + infotainment + voz simultáneamente. Acceso = tier 1/OEM, no se compra en retail.
- **TDA4VM** es específicamente diseñado para ADAS L2/L3, con aceleradores de visión (C7xDSP + matrix multiply) y soporte ASIL-D. Más barato que Orin automotive pero con toolchain más pesado (TIDL).
- **Raspberry Pi 5 + Coral**: prototipo after-market. Face detection a ~21 fps con Coral USB, ~2.6 ms de inferencia sobre PCIe en el Pi 5. No es automotive pero sirve para "caja que enchufas al encendedor".

### Integración con el vehículo

- **Input**: cámara NIR 940 nm por **GMSL2** o **FPD-Link III** (serializadores automotrices, no USB). Sincronizada con LED IR pulsado.
- **Output**: **CAN** o **CAN-FD** al cluster/ADAS ECU para chime, mensaje en HMI, o intervención (vibración de volante, endurecer cinturón). Requiere DBC de la OEM y mucha burocracia.
- **OS**: QNX, Linux automotive (Automotive Grade Linux), o Android Automotive.
- **Certificaciones**: ISO 26262 (functional safety, ASIL-B mínimo para DMS que intervenga), ISO/SAE 21434 (cybersecurity), UN R155/R156.

---

## Arquitectura de referencia propuesta

```
+-------------+     +----------------+     +----------------+     +-------------------+
|  Captura    | --> |  Pre-procesado | --> | Face detection | --> |  Face mesh        |
|  NIR/RGB    |     |  CLAHE / rect. |     |  (blazeface)   |     |  478 landmarks    |
|  ~30 fps    |     |  downscale     |     |                |     |  + blendshapes    |
+-------------+     +----------------+     +----------------+     +---------+---------+
                                                                            |
                                                                            v
+-----------------+     +------------------------+     +--------------------+
|   Alerta        | <-- | Estado (FSM / LSTM)    | <-- | Feature extraction |
| visual/auditiva |     | drowsy/distracted/ok   |     | EAR, MAR, PERCLOS, |
| haptica/CAN     |     | ventana temporal 1-3s  |     | head pose, gaze    |
+-----------------+     +------------------------+     +--------------------+
```

Notas:

- **Pre-procesado**: CLAHE (contrast limited adaptive histogram equalization) ayuda con iluminación mixta (sol lateral). Rectificación necesaria si la cámara está angulada.
- **Feature extraction** es pura aritmética sobre landmarks — portable 1:1 entre laptop, móvil y embebido.
- **Ensamble**: empezar con **FSM** (máquina de estados) por simplicidad e interpretabilidad. Un LSTM pequeño (hidden 32, ventana 90 frames = 3 s) mejora robustez pero requiere dataset etiquetado. Para feria, FSM basta.
- **Alerta**: doble canal — chime + mensaje visual. Escalado: warning → alert → takeover.

---

## Cámara: RGB vs IR

| Dimensión                   | RGB (webcam)         | NIR 940 nm (automotriz)      |
|-----------------------------|----------------------|------------------------------|
| Rendimiento nocturno        | Nulo sin iluminación | Excelente con LED IR activo  |
| Sensible a glare/sol directo| Sí                    | Parcialmente (filtro BP 940) |
| Lentes oscuros del conductor| Falla total           | Atraviesa la mayoría          |
| Modelos pre-entrenados (MediaPipe, BlazeFace) | Sí | No — requiere fine-tune      |
| Costo                       | 5-30 USD              | 50-300 USD módulo automotriz |
| Estándar Euro NCAP 2026 DMS | No compliant          | Requerido de facto           |

Puntos clave:

- **MediaPipe FaceMesh está entrenado con RGB**. Funciona "meh" en imágenes IR 940 nm (la piel se ve uniforme, las cejas y pelo casi negros, los labios sin contraste de color). Landmarks de ojos aún son razonables, pero MAR (boca) y gaze vector degradan.
- **Camino académico**: fine-tuning con datasets IR como **DMD (Driver Monitoring Dataset)**, **DriPE**, **YawDD**, o sintético (Anyverse genera NIR sintético para DMS).
- **Camino pragmático para el MVP/feria**: asumir RGB diurno, documentar explícitamente que "de noche el sistema reporta no disponible y no sustituye al conductor". Es una limitación aceptable para proyecto universitario; en producto real sería un killer.
- Euro NCAP 2026 requiere evaluación de distracción, fatiga y conductor incapacitado. Los OEMs líderes (BMW, Volvo) ya usan NIR dual-camera.

---

## Latencia objetivo y benchmarks publicados

### Objetivo temporal DMS

- **Detección de somnolencia**: típicamente < 2 s desde inicio de episodio (PERCLOS necesita ventana de ~1 minuto clásica, pero variantes modernas con eye closure sostenido >400-500 ms alertan antes).
- **Detección de distracción (mirada fuera de la carretera)**: < 3 s sostenido → warning.
- **Takeover request (L3)**: 6-10 s típico; el DMS debe validar estado del conductor en < 1 s antes del handover.

Esto implica que la pipeline de visión corra a **>=10 fps** sostenidos. No necesitas 60 fps; 15-30 fps con latencia end-to-end <100 ms es cómodo.

### Benchmarks reales

| Plataforma                     | Modelo                     | Latencia             | Fuente                              |
|--------------------------------|----------------------------|----------------------|-------------------------------------|
| Pixel 9 Pro (Android, Tasks)   | Face Landmarker            | 30-70 ms             | GitHub issue google-ai-edge #5872   |
| Desktop WASM                   | Face Landmarker            | 15-20 ms             | Mismo issue                         |
| Pixel 6 (pipeline MP C++/GL)   | Face Mesh (468 pts)        | 5-15 ms              | Docs legacy MediaPipe               |
| iPhone 12 (Vision framework)   | Face landmarks (76 pts)    | <10 ms detección     | docs Apple                          |
| Raspberry Pi 5 + Coral USB     | Face detection             | ~21 fps (~48 ms)     | pmcbride/coral-webcam-detection     |
| Raspberry Pi 5 + Coral PCIe    | Inferencia genérica         | 2.6-2.7 ms           | forums.raspberrypi.com              |
| Jetson Orin Nano Super         | Face Landmarker / MP       | sin benchmark oficial; estimación: 3-10 ms con TensorRT FP16 |

Notas importantes:

- Los 5-15 ms históricos de MediaPipe en Pixel 6 asumían el pipeline C++ con GL, no la SDK Tasks de alto nivel. Para la app real, contar con 30-70 ms.
- Latencia end-to-end incluye: captura (ImageReader) + conversión YUV→RGB + landmarker + features + render. Sumar ~15-30 ms adicionales al tiempo de inferencia puro.

---

## Privacidad y seguridad

### Datos biométricos

- El video facial del conductor es **dato biométrico** bajo GDPR (art. 9) y leyes análogas (LGPD Brasil, LFPDPPP México). Categoría especial → consentimiento explícito y granular.
- **On-device es la única arquitectura razonable**: no subir frames ni embeddings a la nube. Sólo métricas agregadas (drowsy events count, duración) con consentimiento.

### Cybersecurity automotriz

- **ISO/SAE 21434** (2021): standard obligatorio *de facto* para OEMs y Tier 1 en Europa; regula threat analysis and risk assessment (TARA), gestión de vulnerabilidades, security development lifecycle.
- **UNECE R155**: exige CSMS (Cybersecurity Management System) certificado para homologación de vehículos nuevos en UE (desde jul 2024 para todos los tipos nuevos).
- **ISO 26262** (functional safety): un DMS que intervenga (frenado de emergencia, takeover) cae en ASIL-B mínimo, A si solo avisa.

### Implicaciones para este proyecto

- MVP académico → no aplica formalmente, pero **documentar la postura**: inferencia 100% local, cero telemetría, opt-in explícito, borrado inmediato de frames tras procesar.
- Si alguna vez se llevara a producto: requiere threat model, firma de OTA, secure boot en el SoC, aislamiento del DMS respecto a infotainment.

### Consentimiento y ética

- El conductor debe poder saber que está siendo filmado y tener toggle off (con la consecuencia de perder la función).
- Datos **no** deben usarse para productividad/vigilancia laboral sin consentimiento específico y separable (caso Amazon/Uber con DMS ha tenido litigios).

---

## Esfuerzo estimado para cada path

| Path                                         | Tiempo (semanas) | Skills                                                    | Viabilidad feria |
|----------------------------------------------|------------------|-----------------------------------------------------------|------------------|
| Mantener laptop + webcam (status quo)        | 0                | Python, ya hecho                                          | Alta             |
| Pulir UI laptop (Tkinter/PyQt, alarmas)      | 0.5-1            | Python GUI                                                | Muy alta         |
| App Android con MediaPipe Tasks              | 2-3              | Kotlin/Java, Android Studio, CameraX                      | Alta             |
| App iOS con Vision + Core ML                 | 2-4              | Swift, Xcode, signing cert (Apple dev account 99 USD)     | Media (Mac req.) |
| App cross-platform (Flutter/React Native)    | 3-5              | RN/Flutter + bindings nativos al plugin MP                | Media            |
| Raspberry Pi 5 + Coral + cámara + carcasa    | 3-4              | Linux embed, TFLite, PCB/impresión 3D                     | Media-alta       |
| Jetson Orin Nano + cámara CSI + TensorRT     | 4-6              | JetPack, TensorRT, CUDA, CSI camera                       | Media (coste)    |
| SoC automotriz real (TDA4VM/S32V/8295) + CAN | 3-6 meses        | Toolchain vendor, QNX/AGL, ASIL, contactos OEM            | Nula             |

---

## Recomendación

Para la feria:

1. **Demo principal**: laptop + webcam. Es lo que ya funciona, se explica en 30 segundos, no falla por drivers exóticos en el stand. Invertir el tiempo en pulir UI, alertas audibles claras, y cómo visualizas EAR/MAR/PERCLOS en tiempo real.
2. **Demo secundaria (si hay dos personas con tiempo)**: APK Android con **MediaPipe Tasks Face Landmarker** corriendo en un celular montado en un soporte de coche. Esto demuestra el "claim" de que la solución es desplegable y no un experimento académico. 2-3 semanas para un developer con experiencia Android básica.
3. **Narrativa de roadmap**: mostrar la tabla de SoCs automotrices y decir "el camino a producto sería 8295/TDA4VM + cámara NIR, estimamos N meses-persona y requiere partnership con OEM/Tier 1". Esto da seriedad sin intentar implementarlo.

Lo que **no** recomiendo:

- Intentar Jetson sin tiempo: los primeros 3 días se van en flashear JetPack y pelearse con drivers de cámara.
- Intentar IR sin dataset y fine-tuning: MediaPipe dará resultados inconsistentes y quedará peor que el MVP RGB actual.
- Prometer "funciona en la noche" si solo tienes RGB.

---

## Para profundizar

### Docs oficiales

- [MediaPipe Face Landmarker — Android guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/android)
- [MediaPipe Face Landmarker — iOS guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/ios)
- [TensorFlow Lite (LiteRT) delegates overview](https://www.tensorflow.org/lite/performance/delegates)
- [Core ML delegate for TFLite on iOS](https://blog.tensorflow.org/2020/04/tensorflow-lite-core-ml-delegate-faster-inference-iphones-ipads.html)
- [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [Qualcomm Snapdragon Cockpit platform](https://www.qualcomm.com/automotive/solutions/cockpit)
- [TI TDA4VM product page](https://www.ti.com/product/TDA4VM)
- [Euro NCAP 2026 protocols](https://www.euroncap.com/en/for-engineers/protocols/2026-protocols/)

### Standards

- [ISO/SAE 21434:2021 — Road vehicles cybersecurity](https://www.iso.org/standard/70918.html)
- UNECE R155 / R156 (CSMS y SUMS)
- ISO 26262 (functional safety, ASIL)

### Referencias técnicas útiles

- Issue con latencia real en Pixel 9 Pro (MediaPipe Tasks SDK): https://github.com/google-ai-edge/mediapipe/issues/5872
- NIR 940 nm para DMS y Euro NCAP 2026: https://anyverse.ai/near-infrared-cameras-enable-in-vehicle-sensing-advances/
- Benchmarking edge AI (Jetson vs Pi5+Coral): [ResearchGate paper 391165194](https://www.researchgate.net/publication/391165194)

### Datasets para fine-tuning NIR / DMS

- DMD — Driver Monitoring Dataset (VCL-UC3M)
- DriPE
- YawDD (yawning detection)
- NTHU Drowsy Driver Detection Dataset
- Anyverse sintético (NIR simulado)

### Queries útiles para seguir investigando

- `"MediaPipe Tasks" "Face Landmarker" benchmark site:github.com`
- `"driver monitoring" "NIR 940" dataset`
- `"Jetson Orin" "TensorRT" "face landmark" fps`
- `TDA4VM TIDL face mesh benchmark`
- `"Snapdragon 8295" DMS in-cabin reference design`
- `"Euro NCAP" 2026 "driver state monitoring" protocol pdf`
- `GDPR biometric driver monitoring on-device consent`
