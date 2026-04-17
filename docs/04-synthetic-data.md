# Datos sintéticos para DMS — Unity, Unreal, Omniverse

> Documento 04 de la serie de investigación del proyecto Attention-Algorithm (pivot a DMS).
> Pregunta central: **¿es viable generar datos sintéticos de conductores con un motor de juegos, y cómo se hace?**

---

## TL;DR

- **Parcialmente viable**: generar datos sintéticos de rostros/cabezas de conductor con Unity, Unreal o Omniverse es técnicamente posible y está **probado en producción** (Microsoft FaceSynthetics, Synthesis AI, Datagen), pero rara vez alcanza *sota* sin combinarse con datos reales.
- **Gana como augmentation/balanceo**: el uso canónico en papers recientes es **pretraining sintético + fine-tuning con un pequeño conjunto real** (500–5k muestras), no reemplazo total.
- **Ebriedad es el caso que más justifica sintético**: no existen datasets abiertos éticos con conductores realmente intoxicados. Se puede aproximar animando signos observables (nistagmo, ptosis, cabeceo, gaze errático) con curvas paramétricas.
- **Costo para un proyecto universitario**: un pipeline Unity Perception o Unreal + MetaHuman corriendo en una RTX 4070/4090 puede producir **~5–15k clips cortos en 2–4 semanas** de tiempo-máquina, más 3–4 semanas de ingeniería. Servicios comerciales (Synthesis AI, Datagen) cobran del orden de **USD $0.10–$1 por imagen etiquetada**, lo que sale del presupuesto de feria.
- **Sim-to-real gap persiste**, sobre todo en **iluminación IR nocturna** (la cámara DMS típica es NIR 850–940 nm) y micro-expresiones de somnolencia. Hay que planear domain randomization agresivo y/o adaptación de dominio (CycleGAN, UDA).
- **Recomendación MVP (feria)**: híbrido 80/20 — datasets reales (NTHU-DDD, DMD, YawDD, DriverGaze) para somnolencia y distracción, **más un mini-pipeline Unity o Unreal demostrativo** (500–2000 clips) enfocado en ebriedad y balance demográfico.

---

## Por qué sintético: gaps que no cubren los datasets reales

Como se detalló en `03-datasets.md`, los datasets públicos de DMS (NTHU-DDD, DMD, YawDD, DriverMVT, 3MDAD) tienen tres huecos duros: (1) **estados peligrosos reales son inexistentes o éticamente imposibles de capturar** — casi nadie grabó a un conductor realmente ebrio, drogado o en micro-sueño profundo al volante; se actúa, lo cual sesga el modelo hacia actuaciones exageradas; (2) **diversidad demográfica pobre** — la mayoría de datasets abiertos son asiáticos (NTHU, DMD parcial) o del sur de Europa, con poca representación de tonos de piel oscuros, adultos mayores y rasgos no binarios, lo que replica el sesgo documentado en FRVT/NIST; (3) **condiciones de cámara IR nocturnas mal representadas** — muchos datasets son RGB diurnos, pero un DMS embarcado corre en NIR 24/7. Los datos sintéticos atacan exactamente estos tres ejes porque permiten **controlar ground-truth** (estado, identidad, iluminación) sin dilemas éticos y con anotaciones perfectas (landmarks, gaze, head pose, per-pixel segmentation, tiempos de parpadeo) que serían carísimas de anotar manualmente.

---

## Herramientas — comparativa

| Herramienta | Licencia | Costo | Realismo facial | Animación facial | Anotaciones automáticas | Curva de aprendizaje | Ejemplos publicados |
|---|---|---|---|---|---|---|---|
| **Unity + Perception + Human-Centric (People/Face SynthID)** | Proprietary EULA; Personal gratis <USD 200k/año; Pro ~USD 2,200/seat/año | Gratis para uso estudiantil | Medio–alto (con Ziva/Digital Humans addon mejora) | Blendshapes ARKit 52, mocap compatible | Sí: 2D/3D bbox, keypoints, semantic seg, depth, gaze vector (requiere extensión), JSON COCO-like | Media: C#/Prefabs, SDK maduro | SynthDet (objetos), PeopleSansPeople, retail-checkout; varios papers de fatiga usan Unity Perception |
| **Unreal Engine 5 + MetaHuman + MetaHuman Animator** | EULA de Epic; **5% royalty sobre ingresos >USD 1M/producto**; free para uso no comercial / académico | Gratis para el proyecto escolar | **Alto** — MetaHuman es SOTA en rostros fotorrealistas real-time | MetaHuman Animator (desde UE 5.3): captura facial desde iPhone/webcam; ARKit rig completo | Parcial: Movie Render Queue + plug-ins comunitarios (UnrealSynth, EasySynth) para ground truth; landmarks/gaze requieren tooling propio | Alta: Blueprints + C++ + rig humano complejo | Carla (driving sim), MetaHuman demos oficiales, varios papers 2023–2024 de face anti-spoofing y gaze |
| **NVIDIA Omniverse Replicator** | Omniverse Individual gratis; **Enterprise USD ~4,500/seat/año** | Individual gratis para el proyecto | Alto vía Audio2Face + Character Creator import | Audio2Face (lip sync y expresiones desde audio), RTX rig | **Excelente**: Python API nativa para randomization, semantic/instance seg, depth, normals, bbox 2D/3D, keypoints; formato KITTI/COCO out-of-the-box | Media-alta: USD/Python; documentación densa pero bien estructurada | Isaac Sim para robótica; DriveSim para AV; demos de DMS con Replicator publicadas por NVIDIA 2023 |
| **Synthesis AI** (servicio SaaS) | Comercial | **~USD 0.05–0.50 por imagen** según anotación; planes custom | Muy alto (pipeline propio con scans 4D) | Propio, alta fidelidad | Perfectas: 50+ canales (landmarks 3D, gaze, FACS, segmentation, albedo, normals) | Baja: pides vía API, recibes dataset | Papers de face anti-spoofing, gaze estimation; caso de estudio con Toyota Research |
| **Datagen** | Comercial | **~USD 0.10–1 por imagen** según complejidad | Muy alto (scans 3D + GAN texturas) | Sí, full-face rig | Perfectas: 3D landmarks, gaze, emociones AU, scene semantics | Baja | Papers con Meta, BMW; benchmarks en face keypoints |
| **Rendered.ai** (plataforma) | Comercial (PaaS) | Suscripción empresarial (custom, rango USD ~$10k–50k/año) | Medio (depende del generador que conectes — traen Unity/Unreal/Omniverse debajo) | Según engine | Configurable; framework "Channels" | Media (Python SDK, node graphs) | DoD, satellite imagery, manufacturing; face channel menos maduro |
| **Blender + Infinigen / Human Generator / MB-Lab** | GPL / MIT / addons USD 30–100 | **Gratis** o ~USD 100 por addons | Medio (Human Generator decente; Infinigen orientado a naturaleza) | Rig ARKit con Human Generator + Faceit addon | DIY: Blender tiene Compositor + Cycles AOVs; hay scripts comunitarios (BlenderProc, BlendTorch) con ground truth | Media | BlenderProc ha generado datasets como BOP; FaceSynthetics de Microsoft usa pipeline proprietario pero conceptualmente similar |

**Notas de licencia clave**
- Unity Pro es obligatoria solo si la entidad factura >USD 200k/año; para un proyecto de feria, Unity Personal es suficiente.
- Unreal es gratuito hasta USD 1M de ingresos por producto; **investigación académica y datasets no comerciales no pagan royalty**.
- Omniverse Individual sigue siendo gratuito (post-reestructura 2024); solo la edición Enterprise con soporte y licencias de Nucleus compartido cobra.
- MetaHuman Creator está ligado a la EULA de Epic y **no permite re-distribuir los assets fuera de un producto UE**; los datasets generados (imágenes/vídeos renderizados) sí son del autor, pero los .uasset del rig no se pueden publicar.

---

## Pipelines publicados con datos sintéticos para DMS / face

### Microsoft FaceSynthetics (2021)
- **100,000 rostros sintéticos** con landmarks 2D/3D perfectos, renderizados con pipeline propio basado en scans 3D + rigs paramétricos (Wood et al., ICCV 2021, "Fake it till you make it").
- Entrenaron modelos de face parsing y landmark detection **solo con sintético** y vencieron en benchmarks a modelos entrenados con datos reales como 300-W.
- Métrica clave: NME 3.09 en 300-W test set, competitivo con SOTA real-only (~2.8 NME).
- **Lección**: con suficiente diversidad + domain randomization (iluminación HDR, cámaras, poses) sintético puro *puede* competir en tareas 2D bien definidas.

### DriPE (2021) — Ford
- Dataset sintético de poses de conductor generado con motor propietario; usado para estimación de head pose y keypoints del torso.
- ~10k imágenes con anotación 3D; complementado con un set real pequeño.
- Sim-to-real: caída de ~5–8% en MAE de yaw al pasar de sim-test a real-test sin fine-tuning; ~1.5% con fine-tuning.

### SynFace (ICCV 2021)
- Enfrentó el problema de reconocimiento facial entrenado solo con rostros sintéticos (DiscoFaceGAN).
- Ganó ~10% absoluto al agregar domain randomization de identidad y mixing con un subset real pequeño.

### Unity Perception + fatiga (varios papers 2022–2024)
- Grupos de la Universidad de Ottawa y TU Delft publicaron prototipos que usan Unity Perception para generar secuencias de parpadeo y yawning.
- Métrica reportada: accuracy de detección de somnolencia cae de **92% (real-only, NTHU)** a **74% (sim-only)**, sube a **94% con pretraining sim + finetune real**.

### Synthesis AI — Face DMS (2023)
- Caso de estudio con cliente automotriz: entrenaron gaze estimation y drowsiness en NIR usando 500k imágenes sintéticas NIR-simuladas + 2k reales.
- Reportaron reducción del 40% en dataset real necesario para alcanzar el mismo accuracy.

### Tesla / Waymo (cerrados)
- Tesla usa su simulador propio (mostrado en AI Day 2021 y 2022) para generar corner cases con cabinas y ocupantes, pero no publican detalles ni datasets.
- Waymo Open Dataset incluye algo de simulación pero el foco es exterior.

### DriveSim + Omniverse (NVIDIA 2023)
- Demo público de DMS sintético con Replicator: 50k frames con ground truth de gaze, head pose, estados (phone, drink, drowsy). Notebooks en el repo `NVIDIA-Omniverse/synthetic-data-examples`.

**Patrón común**: nadie ha ganado sota en DMS con **sintético puro**; el sweet-spot reportado es **10–100× más datos sintéticos que reales, con reales como fine-tune**.

---

## Sim-to-real gap — qué falla

Cuando se entrena solo con sintético y se prueba en real, los modos de falla típicos en DMS son:

1. **Texturas de piel y subsurface scattering**
   - Los shaders de MetaHuman y Character Creator son los mejores, pero aún tienen "look CG" en cerca. El modelo aprende a explotar artifacts (bordes duros en poros, reflectancia especular uniforme) que no existen en real.
2. **Iluminación IR nocturna (850–940 nm)**
   - La cámara DMS embarcada real es **monocromática NIR activa**. Renderizar NIR en Unity/Unreal requiere hacks: muchos equipos rendern en grayscale + filtro + ajuste manual de reflectancia de piel (la piel humana refleja ~40–60% más en NIR que en visible, las venas se oscurecen, las pupilas brillan por corneal reflection).
   - Omniverse con RTX y Audio2Face tiene el soporte más serio para esto vía MDL custom.
3. **Micro-expresiones de somnolencia**
   - PERCLOS (porcentaje de ojo cerrado) se puede animar, pero el patrón real de **droopy eyelid gradual + re-apertura brusca** requiere curvas de animación informadas por datos reales (o mocap). Un blendshape lineal entre abierto/cerrado no engaña al modelo.
4. **Parpadeos realistas**
   - Duración real ~100–400 ms con perfil asimétrico (cierre rápido, apertura más lenta). Animaciones genéricas con curva coseno simétrica producen un **sesgo detectable**.
5. **Signos de ebriedad**
   - Nistagmo (oscilación lateral involuntaria del ojo), ptosis (párpado caído), sacadas lentas, cabeceo tipo "head nod". Literatura médica (NHTSA DWI Detection Manual) describe los signos pero la dinámica real de intoxicación es estocástica y depende del BAC; simular curvas paramétricas es una aproximación cruda.

### Técnicas para cerrar el gap

- **Domain randomization (Tobin et al. 2017)**: variar agresivamente iluminación, fondos, cámaras, ruido, post-proc. Barato, primer approach obligatorio.
- **Structured domain randomization**: mantener realismo en ejes clave (piel, ojos) y randomizar el resto. Enfoque de FaceSynthetics.
- **Domain adaptation supervisada (UDA)**: CycleGAN para traducir sintético→real, pero degrada etiquetas si no se cuida. Funciona mejor para RGB→NIR que al revés.
- **Pretraining + fine-tuning**: lo más reproducible. Entrena con cientos de miles sintéticas, fine-tunea con 1–5k reales etiquetadas. Estado del arte pragmático.
- **Mixed-batch training**: cada minibatch incluye N% sintético y (100-N)% real; empezar N=80, decaer a N=20 al final del entrenamiento.

---

## Estimación de esfuerzo y costo para el proyecto

**Escenario**: 10,000 clips de 5 s a 30 fps = **1.5 M frames**. 20 identidades × 5 estados (normal, distraído, somnoliento, ebrio, micro-sueño). Render NIR + RGB.

### Opción A — Unity Perception + Human-Centric Package

| Recurso | Cantidad | Costo |
|---|---|---|
| Licencia Unity Personal | 1 | $0 |
| RTX 4090 (propia/laboratorio) | 1 | ~$0 marginal (asume hardware existente) |
| Tiempo render (1080p, 30 fps, scene medio) | ~0.5–1 s/frame en CPU-bound path tracing; ~0.1–0.2 s real-time rendering | 1.5M frames × 0.15 s ≈ **62.5 h GPU** (render real-time, calidad media) |
| Ingeniería (setup proyecto, scripts, rigging) | — | **~120 h** estudiante (3 semanas FT × 40 h) |
| Animación de estados (curvas, mocap lite con iPhone) | — | ~40 h |
| Validación + iteración | — | ~40 h |
| **Total** | | **~3 días GPU + 5–6 semanas ingeniería** |

### Opción B — Unreal + MetaHuman

| Recurso | Cantidad | Costo |
|---|---|---|
| Licencia | — | $0 (no comercial) |
| Render Movie Render Queue 1080p path-traced | ~2–10 s/frame | 1.5M × 4 s ≈ **1,600 h GPU** (inviable en una 4090) |
| Render Lumen real-time 1080p | ~0.05–0.1 s/frame | 1.5M × 0.07 s ≈ **30 h GPU** |
| Ingeniería (MetaHuman imports, Animator, blueprints) | — | **~160 h** (4 semanas FT) |
| Mocap facial (iPhone TrueDepth + MetaHuman Animator) | 20 identidades × 30 min | ~20 h |
| Post-proc y labels (custom plugin) | — | ~40 h |
| **Total** | | **~1.5–2 días GPU real-time + 6–7 semanas ingeniería** |

### Opción C — Omniverse Replicator

| Recurso | Cantidad | Costo |
|---|---|---|
| Licencia Individual | — | $0 |
| RTX 4090 | 1 | $0 marginal |
| Render RTX real-time | ~0.1 s/frame | **~42 h GPU** |
| Ingeniería (USD, Python Replicator, Character Creator assets) | — | **~140 h** (3.5 semanas) |
| **Total** | | **~2 días GPU + 5 semanas ingeniería** |

### Opción D — Pagar a Synthesis AI / Datagen

- 1.5M frames × USD $0.10 (precio optimista por imagen estática, no vídeo) = **USD $150,000**.
- Vídeo es más caro; estiman **USD $0.30–$1 por frame de secuencia**.
- Reducción por volumen para 100k+ frames típicamente 30–50%, pero sigue fuera de presupuesto universitario.
- **Fuera de alcance para feria**.

### Pros/contras

| | Self-hosted (Unity/Unreal/Omniverse) | Servicio (Synthesis/Datagen) |
|---|---|---|
| Costo monetario | ~$0 (hardware prestado) | $10k–$150k típicos |
| Costo en tiempo | 5–7 semanas ingeniería | ~1 semana de integración |
| Calidad out-of-the-box | Media, requiere tunear | Alta |
| Control sobre ediciones | Total | Limitado al API del vendor |
| Riesgo técnico | Alto (muchos puntos de falla) | Bajo |
| Valor pedagógico | **Alto** (aprendes el pipeline) | Bajo |

Para feria universitaria, **self-hosted Unity o Omniverse** es la única vía razonable.

---

## Especial: cómo simular ebriedad

La ebriedad es el estado más díficil de obtener éticamente y el caso de uso más fuerte para sintético. Tres enfoques complementarios:

### (a) Animación paramétrica basada en literatura médica

Signos observables de intoxicación alcohólica (NHTSA DWI Detection Manual, literatura de toxicología forense):

- **Nistagmo por gaze (HGN)**: oscilación horizontal involuntaria del ojo cuando se mira lateralmente >45°. Amplitud ~2–5°, frecuencia 2–4 Hz. Se anima con ruido 1/f aplicado al yaw del ojo cuando `|eye_yaw| > 30°`.
- **Ptosis**: párpado superior caído 1–3 mm. Bias negativo en blendshape `eyelidUpperOpen`.
- **Sacadas lentas**: velocidad sacádica reducida ~25–40%. Se modela bajando la velocidad angular máxima del rig de ojos.
- **Cabeceo / head nod**: micro-sueño intermitente. Modelo de Ornstein-Uhlenbeck sobre el pitch de la cabeza con eventos de caída súbita (pitch → 30°) y recuperación rápida.
- **Gaze errático**: reducción de la capacidad de fijación. Ruido gaussiano 1–3° superpuesto a la trayectoria target.
- **Tiempo de reacción aumentado**: retraso en la saccade-to-target ~200–400 ms extra.
- **Expresión facial**: "slack jaw" (mandíbula laxa), ceño relajado, reducción de movimientos voluntarios.

Se implementa como un **AnimController "drunk" con parámetro `BAC_level` (0.0–0.25)** que modula la magnitud de cada canal. Permite generar gradientes de intoxicación.

### (b) Mocap de actores simulando intoxicación

- Captura con iPhone TrueDepth + Live Link Face + MetaHuman Animator.
- Actores entrenados (teatro, improv) pueden emular signos razonablemente. **No es tan convincente como ebriedad real**, pero evita el sesgo de "actor sobreactuando" si se les da guía basada en (a).
- Tiempo: ~4 h de captura por identidad, 20 identidades → 80 h. Costo $0 si son compañeros.
- Ventaja: las micro-dinámicas (temblor muscular, micro-expresiones) salen gratis.

### (c) Literatura médica y datasets paralelos

- **NHTSA SFST manual** (Standardized Field Sobriety Test) tiene video de referencia de signos HGN y marcha.
- Papers de **tele-medicina para detección de intoxicación** (Chen et al. 2020; Koukiou & Anastassopoulos varios) listan features térmicos y visuales con ground truth BAC; útiles como referencia biológica aunque no sean datasets de conductor.
- Dataset **"Drunk driving" de la Univ. de Indiana** (acceso restringido IRB) es el único conocido con BAC real medido; escribir al autor puede ser más eficiente que generar todo sintético.

### Consideraciones éticas

- **Nunca publicar el dataset como "real"**: etiquetar claramente `synthetic`.
- **No usar caras de personas reales identificables** sin consentimiento — MetaHuman y Character Creator proveen identidades generadas; evitar scans de terceros.
- **El modelo final debe advertir su alcance**: un clasificador entrenado mayormente con sintético no debe presentarse como diagnóstico de embriaguez forense.

---

## Recomendación para el MVP

Para la feria con recursos limitados (1–2 estudiantes FT, 4–6 semanas, hardware consumer, presupuesto ~$0), la arquitectura realista es **híbrida, 80/20 real/sintético**:

### Plan 4–6 semanas

**Semana 1 — Datos reales baseline**
- Descargar y homogeneizar NTHU-DDD + DMD + YawDD + un subset de DriverGaze.
- Entrenar baseline de somnolencia y distracción (MobileNetV3 o EfficientNet-Lite).
- Reporte de métricas y de **sesgos detectados** (poca piel oscura, poco nocturno).

**Semana 2 — Pipeline sintético mínimo**
- Elegir stack: **Unity Perception + Human-Centric Package** (recomendado por curva más suave) o **Omniverse Replicator** si hay afinidad con Python.
- Setup de 3–5 identidades diversas (género, edad, tono de piel) con Character Creator assets libres.
- Escena: interior de auto genérico, cámara posición DMS (arriba-volante, 30° abajo), luz mixta día/noche.

**Semana 3 — Estados base**
- Animar parpadeos realistas (curva asimétrica), yawning, head turns.
- Generar ~1,500 clips con etiquetas (estado, identidad, iluminación).
- Implementar domain randomization: HDRIs, skins, cámaras intrínsecas.

**Semana 4 — Ebriedad**
- Implementar el AnimController "drunk" con BAC_level (sección anterior).
- Capturar 2–3 mocaps de compañeros con iPhone simulando (guía medica).
- Generar ~500 clips de ebriedad sintética.

**Semana 5 — Fine-tuning + evaluación**
- Pretrain en sintético (2k clips), fine-tune en subset real (~5k imágenes balanceadas).
- Evaluación cruzada: sim-test, real-test, real-test balanceado por demografía.
- A/B: baseline real-only vs. híbrido en el split demográfico infra-representado.

**Semana 6 — Demo + reporte**
- Demo con cámara USB IR en tiempo real mostrando los 5 estados.
- Reporte con límites honestos (no prometer detector de ebriedad robusto).
- Si se llega: exportar modelo a ONNX y correr en notebook del stand.

### Qué NO intentar
- Generar 100k+ clips desde cero — no da el tiempo.
- Simular NIR realista sin referencia — mejor renderizar RGB + filtro grayscale + nota en el reporte.
- Reemplazar datasets reales — el sintético debe complementar, no sustituir, al menos no en 6 semanas.

---

## Para profundizar

### Documentación oficial
- **Unity Perception**: https://github.com/Unity-Technologies/com.unity.perception
- **Unity Human-Centric Package / PeopleSansPeople**: https://github.com/Unity-Technologies/PeopleSansPeople
- **Unreal MetaHuman**: https://dev.epicgames.com/documentation/en-us/metahuman
- **MetaHuman Animator**: https://dev.epicgames.com/community/learning/courses/GVL/unreal-engine-metahuman-animator
- **Omniverse Replicator**: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html
- **NVIDIA Omniverse synthetic-data-examples**: https://github.com/NVIDIA-Omniverse/synthetic-data-examples
- **BlenderProc**: https://github.com/DLR-RM/BlenderProc

### Servicios comerciales
- Synthesis AI: https://synthesis.ai
- Datagen: https://datagen.tech
- Rendered.ai: https://rendered.ai

### Papers clave
- Wood et al. (2021) *Fake it till you make it: face analysis in the wild using synthetic data alone*. ICCV 2021. (FaceSynthetics)
- Tobin et al. (2017) *Domain randomization for transferring deep neural networks from simulation to the real world*. IROS 2017.
- Kortylewski et al. (2019) *Analyzing and Reducing the Damage of Dataset Bias to Face Recognition With Synthetic Data*. CVPRW.
- Ebadi et al. (2022) *PeopleSansPeople: A synthetic data generator for human-centric computer vision*. arXiv:2208.09368.
- Hinterstoisser et al. (2019) *An Annotation Saved is an Annotation Earned: Using Fully Synthetic Training for Object Detection*. ICCVW.
- Borkman et al. (2021) *Unity Perception: Generate Synthetic Data for Computer Vision*. arXiv:2107.04259.
- NHTSA DWI Detection and Standardized Field Sobriety Testing Student Manual (2018).
- Chen et al. (2020) *A Survey on Thermal Imaging for Driver Monitoring*.

### Tutoriales prácticos
- Unity Perception tutorial oficial (SynthDet): https://github.com/Unity-Technologies/com.unity.perception/blob/main/com.unity.perception/Documentation~/Tutorial/TUTORIAL.md
- MetaHuman + Live Link Face (YouTube Epic Games channel, serie 2023).
- NVIDIA Replicator "Getting Started" (Omniverse Launcher → Learn tab).
- *Synthetic Data for Deep Learning* — Sergey I. Nikolenko (Springer 2021), libro de referencia.

### Datasets sintéticos que se pueden descargar como punto de partida
- **FaceSynthetics (Microsoft)** — 100k imágenes, landmarks. https://github.com/microsoft/FaceSynthetics
- **PeopleSansPeople** — escenas humanas Unity. https://github.com/Unity-Technologies/PeopleSansPeople
- **SynFace** — solo código; identidades generadas vía DiscoFaceGAN.

---

*Última revisión: 2026-04-17. Próximo doc sugerido: `05-hardware-camera.md` (cámaras NIR embarcadas, selección de sensor, sincronización con iluminador).*
