# Datasets reales de conducción — somnolencia, distracción, ebriedad

## TL;DR

- **Hipótesis del equipo: CONFIRMADA.** No existe ningún dataset público unificado que cubra los tres estados objetivo (alerta / somnoliento / distraído / ebrio). Cada dataset se especializa en **una sola dimensión**.
- **Somnolencia**: el área mejor cubierta. NTHU-DDD y UTA-RLDD son el estándar de facto; DROZY aporta multimodalidad (EEG/KSS) pero con muy pocos sujetos (14).
- **Distracción**: bien cubierta para el paradigma de "clasificación de actividad" (State Farm, AUC, 100-Driver), pero casi todos se enfocan en **objetos visibles** (celular, comida, radio) y no en distracción por gaze puro.
- **Ebriedad**: prácticamente **no existe** dataset público específico de conductores ebrios, por razones éticas e IRB obvias. Hay estudios aislados con rostro térmico y literatura sobre nistagmo/gaze, pero nada liberado como benchmark.
- **Consecuencia práctica**: el MVP tendrá que combinar ≥3 datasets y/o generar datos propios (ver doc 04). La evaluación honesta del módulo de ebriedad no podrá hacerse con benchmark público — habrá que declarar ese límite en el reporte final.
- **Riesgo de licencias**: varios datasets son *research-only* (NTHU-DDD, DMD, Drive&Act). Si el proyecto alguna vez pretende demo comercial, hay que filtrar desde ya.

---

## Datasets de somnolencia / fatiga

| Dataset | # Sujetos | Tamaño | Modalidad | Etiquetas | Demografía | Licencia | Costo | Estado de acceso |
|---|---|---|---|---|---|---|---|---|
| **NTHU-DDD** (National Tsing Hua Univ. Drowsy Driver Detection) | 36 | ~9.5 h video | RGB + IR (simulador) | drowsy/non-drowsy + yawning, nodding, slow-blink, looking-aside, talking/laughing | 18 M / 18 F, asiáticos | Research-only, requiere formulario | Gratis académico | Solicitud por email al lab (históricamente lento). Mirror en algunos repos. |
| **UTA-RLDD** (UT Arlington Real-Life Drowsiness) | 60 | ~30 h video (180 clips de 10 min) | RGB (webcam, casero) | 3 clases KSS-like: alert, low vigilance, drowsy | 51 M / 9 F, diverso (edad 20-59) | Research-only | Gratis | Disponible vía sitio de Reza Ghoddoosian / UT Arlington. Usable. |
| **YawDD** (Yawning Detection Dataset) | 107 (v1) + 29 (v2) | 322 videos | RGB (cámara en dash y espejo) | yawning / talking / silent | Mixto género y etnia | Research (IEEE DataPort) | Gratis con cuenta IEEE | Disponible en IEEE DataPort. Muy usado para yawn-only. |
| **DROZY** (ULg Multimodal) | 14 | ~36 h multi-sensor | RGB + NIR + EEG + EOG + ECG + EMG + KSS | PVT score, KSS (1-9), fisiológico sincronizado | Estudiantes Uni. Liège | Research-only, EULA | Gratis | Vía Univ. de Liège. Pocos sujetos pero gold-standard multimodal. |
| **CEW** (Closed Eyes in the Wild) | 2,423 imgs | 1,192 ojos cerrados + 1,231 abiertos | RGB still | eye open/closed (crops) | "in the wild" (sin demografía formal) | Research | Gratis | Vía Parkhi et al. / páginas académicas. Usable para clasificador binario de ojo. |
| **FL3D** (Fatigue/Face Large-scale 3D Drowsiness) | ~300 | 10k+ imgs | RGB + 3D head pose | fatigue levels, head pose | Poco reportado | Variable | Gratis en algunas mirrors | Disponibilidad intermitente; verificar Kaggle. |
| **DDD (Kaggle — "Driver Drowsiness Dataset")** | Variable según versión | 10k-40k imgs | RGB | drowsy / non-drowsy (binario) | No documentada formalmente | CC0 / CC-BY en la mayoría | Gratis | Múltiples versiones en Kaggle (buscar "driver drowsiness"). Calidad muy heterogénea; típicamente reempaquetan NTHU + MRL + CEW. |

### Notas

- **NTHU-DDD** sigue siendo el más citado pero es **simulador**, no conducción real — sesgo conocido.
- **UTA-RLDD** es el más cercano a "real world" porque los sujetos grabaron ellos mismos con webcam cuando estaban genuinamente cansados.
- **DROZY** es el único con fisiología; útil para **validar** que las features visuales (PERCLOS, blink rate) correlacionan con cansancio medido objetivamente.
- **CEW + MRL Eye** son los datasets go-to para entrenar el clasificador ojo-abierto/cerrado que alimenta PERCLOS.

---

## Datasets de distracción

| Dataset | # Sujetos | Tamaño | Modalidad | Etiquetas | Demografía | Licencia | Costo | Estado |
|---|---|---|---|---|---|---|---|---|
| **State Farm Distracted Driver** (Kaggle 2016) | 26 | ~22k imgs train + 79k test | RGB (cámara lateral dashboard) | 10 clases: safe driving, texting-left/right, phone-left/right, radio, drinking, reaching behind, hair/makeup, talking to passenger | Mixta | Kaggle competition — uso académico; prohibido comercial | Gratis | Disponible en Kaggle. Benchmark más usado. |
| **AUC Distracted Driver v1 / v2** (American Univ. in Cairo, Eraqi et al. 2019) | 44 (v2) | ~17k imgs (v2) | RGB | 10 clases similares a State Farm | 29 nacionalidades, M/F | Research-only | Gratis | Solicitud por formulario; usualmente concedido. |
| **100-Driver** (Wang et al. 2023) | 100 | ~470k frames | RGB + NIR, 4 vistas | 22 clases de distracción (extiende State Farm) | Multietnia, M/F | Research | Gratis con solicitud | Más grande y moderno; disponible bajo petición a autores. |
| **DMD — Driver Monitoring Dataset** (Vicomtech, Ortega et al. 2020) | 37 | ~41 h video | RGB + IR + depth, 3 cámaras (cara, cuerpo, manos) | Jerarquía anidada: drowsiness, distraction, gaze zone, hands on wheel, objetos | Europa, M/F | Research-only, EULA estricto | Gratis académico | dmd.vicomtech.org — activo. El más completo de la lista. |
| **3MDAD** (Multi-Modal Multi-view Driver Action) | 50 | ~100k frames | RGB + depth, 2 vistas | 16 acciones (incluye comer, beber, texting, ajustar GPS) | Tunicia | Research | Gratis | Disponible vía autores (Jegham et al.). |
| **DriverMVT** (Multi-View Transformer dataset) | ~90 | Variable | RGB multi-view | Acciones de distracción | Poco documentado | Research | Gratis | Menos conocido; verificar paper original. |
| **Drive&Act** (Martin et al. 2019, KIT + Daimler) | 15 | ~12 h | RGB + IR + depth, 5 cámaras | 83 acciones jerárquicas (12 top-level) | Europa | Research-only | Gratis con EULA | driveandact.com — activo. |
| **SynDD1** (Synthetic Distracted Driver, Rahman et al. 2023) | Sintético (Unity) | ~100k frames | RGB sintético | 10+ clases | Avatares sintéticos | Open | Gratis | GitHub. Útil para domain randomization / pretraining. |

### Notas

- **State Farm** tiene el defecto clásico: sujetos del test set también aparecen en train → inflación de métricas. AUC lo resuelve con split por sujeto.
- **DMD** de Vicomtech es, con diferencia, el más útil para un DMS real porque incluye **drowsiness + distraction + gaze zone en los mismos sujetos**. Sigue sin cubrir ebriedad.
- **Drive&Act** usa vehículo real (no simulador) — importante para validación in-the-wild.
- La mayoría etiqueta "distracción" como **acción con objeto visible**. La distracción por **gaze desviado sin objeto** (mirar ventana, zoning out) está sub-representada.

---

## Datasets de ebriedad / impairment

**Veredicto directo**: no existe un dataset público, de escala utilizable y con conductores reales ebrios. Esto es esperable por tres razones:

1. **IRB / ética**: emborrachar sujetos y ponerlos a conducir (aunque sea en simulador) requiere protocolos médicos serios, consentimiento informado agravado, y supervisión.
2. **Legal**: redistribuir caras de personas visiblemente intoxicadas colisiona con dignidad, GDPR y normativas de datos sensibles.
3. **Comercial**: las empresas que sí tienen datos (OEMs automotrices, aseguradoras, Seeing Machines, Smart Eye) los guardan como propiedad industrial.

### Lo que sí existe (con matices)

| Recurso | Qué es | Utilidad real | Acceso |
|---|---|---|---|
| **Sober-Drunk Face Dataset** (Koukiou & Anastassopoulos, Univ. Patras, ~2012) | ~41 sujetos, imágenes térmicas (FIR) antes y después de ingerir alcohol | Térmico, no RGB; muestra pequeña; evalúa enrojecimiento facial vascular | Solicitud a autores; disponibilidad intermitente. Muchos papers lo citan sin link vivo. |
| **Estudios NHTSA** ("Visual Detection of DWI Motorists", Stuster 1997/2010) | Reporte con 24 señales visibles de intoxicación al volante (weaving, slow response, etc.) | **No es un dataset**, es una taxonomía. Útil para definir qué features buscar. | Público en nhtsa.gov |
| **Literatura sobre nistagmo alcohólico (HGN — Horizontal Gaze Nystagmus)** | Protocolo estándar de campo (SFST) | Define que alcohol produce nistagmo horizontal a <45° desviación, seguimiento suave roto y falta de convergencia | Papers médicos / manuales policiales |
| **Estudios de detección de alcohol con rostro térmico RGB-NIR** | Papers de Koukiou, Hermosilla et al., Neves et al. | Usan térmica de nariz/mejillas; pocas muestras; no replicables sin hardware térmico | Papers; datasets privados |
| **MMAct / Kinetics subsets de "intoxicated walking"** | Acciones humanas genéricas | Marginal: caminar ebrio no es conducir ebrio | Público |
| **DriveSim / simuladores con BAC simulado** | Efectos visuales de BAC simulados (no sujetos reales ebrios) | Útil solo si se combina con modelado de signos visuales | Variable |

### Implicación para el proyecto

El módulo de "ebriedad" **no podrá ser entrenado ni benchmarkeado contra un dataset público directo**. Las opciones honestas son:

1. **Proxy por features**: entrenar un detector de **signos visuales correlacionados** con intoxicación (micro-sueño + gaze inestable + saccades erráticas + head bob lento + blink rate anómalo). Se valida contra los otros datasets de somnolencia/gaze, no contra "ebrio" directo.
2. **Self-collected**: grabar compañeros voluntarios (sobrios) simulando signos — **honestamente marcado como sintético/actuado**, no como ground truth de ebriedad real.
3. **Declarar el límite**: en el reporte final decir explícitamente "no validamos contra ebriedad real por falta de dataset — medimos proxies".

Ver doc 04 para detalles de la estrategia proxy.

---

## Datasets adyacentes útiles

Aunque no son de conducción, sirven como piezas de pipeline (landmarks, clasificadores de ojo, pretraining).

| Dataset | Utilidad | Tamaño | Licencia |
|---|---|---|---|
| **MRL Eye Dataset** (Univ. Olomouc) | Crops de ojo abierto/cerrado bajo varias iluminaciones — base para PERCLOS | 84k imgs, 37 sujetos | Research, gratis |
| **BioID Face Database** | Caras con landmarks, útil para validar detección frontal | 1,521 imgs | Gratis |
| **300-W** (300 Faces in the Wild) | 68 landmarks estándar — gold para alinear FaceMesh | ~4k imgs | Research |
| **WFLW** (Wider Facial Landmarks in-the-Wild) | 98 landmarks con atributos (occlusion, pose, illumination, make-up) | 10k imgs | Research |
| **HUST-ALF** (Huazhong Univ. Asian Large-scale Face) | Caras asiáticas; útil si la demografía objetivo es LATAM/Asia | Grande | Research |
| **HELEN / LFPW / AFW** | Landmarks clásicos | Variable | Research |
| **CelebA / FFHQ** | Pretraining genérico de caras | 200k / 70k | CC (con restricciones) |
| **UnityEyes / NVGaze** | Ojos sintéticos con ground truth de mirada | Ilimitado | Research |
| **MPIIGaze / MPIIFaceGaze** | Gaze estimation en webcam — relevante para "mirada desviada" | 15 sujetos, 213k imgs | Research |
| **GazeCapture** | Gaze en móvil; 1,450 sujetos | 2.5M frames | Research |
| **Columbia Gaze** | Gaze en 5 head poses × 21 gaze directions | 56 sujetos | Research |

---

## Análisis comparativo — gap de cobertura

Matriz de cobertura (✓ = soportado directamente; ~ = parcial/proxy; ✗ = ausente):

| Dataset | Alerta / normal | Somnolencia | Distracción con objeto | Distracción por gaze | Ebriedad | Multimodal | Real (no simulador) |
|---|---|---|---|---|---|---|---|
| NTHU-DDD | ✓ | ✓ | ~ (looking-aside, talking) | ~ | ✗ | IR+RGB | ✗ simulador |
| UTA-RLDD | ✓ | ✓ | ✗ | ✗ | ✗ | RGB | ✓ casero |
| YawDD | ~ | ~ (yawn only) | ✗ | ✗ | ✗ | RGB | ~ coche parado |
| DROZY | ✓ | ✓ | ✗ | ✗ | ✗ | RGB+NIR+EEG+EOG+ECG | ✗ lab |
| State Farm | ✓ | ✗ | ✓ | ✗ | ✗ | RGB | ✓ |
| AUC v2 | ✓ | ✗ | ✓ | ✗ | ✗ | RGB | ✓ |
| 100-Driver | ✓ | ✗ | ✓ | ~ | ✗ | RGB+NIR | ✓ |
| DMD (Vicomtech) | ✓ | ✓ | ✓ | ✓ (gaze zone) | ✗ | RGB+IR+depth | ✓ |
| 3MDAD | ✓ | ✗ | ✓ | ~ | ✗ | RGB+depth | ✓ |
| Drive&Act | ✓ | ~ | ✓ | ~ | ✗ | RGB+IR+depth | ✓ |
| SynDD1 | ✓ | ✗ | ✓ | ~ | ✗ | RGB sintético | sintético |
| Sober-Drunk Face | ✗ | ✗ | ✗ | ✗ | ✓ (lab, térmica) | FIR | ✗ lab |

### Observaciones

- **Ningún dataset cubre los tres estados** (somnolencia + distracción + ebriedad). **DMD de Vicomtech** es el que más se acerca: cubre dos (somnolencia + distracción + gaze zone) en los mismos sujetos. Pero ebriedad está ausente en todos.
- **Ninguna fuente pública** combina modalidad multimodal (EEG/fisiología) con escenario de conducción real. DROZY tiene la fisiología pero es lab; DMD tiene el escenario pero no la fisiología.
- La **distracción por gaze desviado sin objeto** solo la cubre bien DMD (con etiqueta "gaze zone") y parcialmente 100-Driver.
- Veredicto firme: **la hipótesis del equipo se sostiene**. El proyecto debe ensamblar múltiples datasets y aceptar que el módulo de ebriedad no tendrá benchmark propio.

---

## Estrategia recomendada para el proyecto

### Módulo de somnolencia

- **Train**: UTA-RLDD (real world, split por sujeto) + MRL Eye (para el clasificador ojo-cerrado que alimenta PERCLOS).
- **Validación cruzada**: NTHU-DDD (para medir generalización entre simulador y real).
- **Sanity check fisiológico**: DROZY (verificar que PERCLOS y blink rate calculados correlacionan con KSS).
- **Métricas**: PERCLOS (P70/P80), blink duration, yawn frequency, head-nod.

### Módulo de distracción

- **Train principal**: DMD (Vicomtech) si la licencia se concede a tiempo. Fallback: AUC v2 (mejor que State Farm por split por sujeto).
- **Augmentation**: SynDD1 para domain randomization.
- **Gaze desviado sin objeto**: DMD gaze-zone + MPIIFaceGaze como pretraining del regresor de mirada.
- **Métricas**: accuracy multiclase + F1 por clase + tiempo de respuesta.

### Módulo de ebriedad (proxy)

- **Sin dataset directo**. Estrategia:
  1. Extraer features: PERCLOS, blink rate, gaze stability (varianza angular), saccade smoothness, head pose tremor, tiempo de reacción a cambios de estímulo.
  2. Entrenar un clasificador binario "patrón normal" vs "patrón anómalo" usando datos **sobrios** como negativo y bootstrappear anomalía con data augmentation temporal (slow-down artificial, gaze jitter sintético).
  3. Validar cualitativamente con videos actuados por voluntarios (declarado como limitación).
- **Referencia teórica**: protocolo HGN + signos NHTSA para justificar qué features buscar.

### Advertencias de licencia

- **No comercial**: NTHU-DDD, UTA-RLDD, DMD, Drive&Act, AUC, 3MDAD, DROZY, 100-Driver, YawDD — todos research-only. Si el proyecto sale a producción, hay que re-licenciar o re-capturar.
- **Kaggle competition data** (State Farm): restringido a uso dentro de la competencia técnicamente; en la práctica se usa académicamente pero no es seguro para comercial.
- **SynDD1** y varios Kaggle sueltos: CC0/MIT, sí son comerciales.
- **Redistribución**: ninguno de los anteriores se puede redistribuir. Lo que sí se puede es publicar los **pesos del modelo** entrenado (con disclaimer) y los **scripts de preprocesamiento**.

---

## Consideraciones éticas y de privacidad

- **IRB**: si el equipo captura datos propios (cara de voluntarios, especialmente simulando ebriedad/cansancio), debe pasar por el comité de ética institucional del Tec. Documentar protocolo, consentimiento informado por escrito, derecho a retirar datos.
- **Consentimiento explícito por modalidad**: grabar cara ≠ grabar fisiología ≠ publicar en paper. Consentimiento granular.
- **GDPR / LFPDPPP** (México): las imágenes faciales son **datos biométricos sensibles**. Tratamiento requiere base legal explícita; no basta con "interés legítimo".
- **Minimización**: guardar features (landmarks, PERCLOS) en vez de video crudo cuando sea posible; anonimizar caras en cualquier demo pública (blur / sustitución).
- **Sesgo demográfico**: casi todos los datasets listados tienen sesgo hacia caucásico o asiático del este. Latinoamericanos están sub-representados → validar explícitamente con muestra local antes de afirmar rendimiento.
- **Uso comercial posterior**: si el proyecto se vuelve producto, la cadena de datos se reconstruye desde cero con licencias comerciales y consentimientos nuevos. Los modelos pre-entrenados con research-only **no se heredan** legalmente.
- **Dual-use y riesgo**: un DMS puede usarse para vigilancia laboral abusiva (tracking de empleados). Declarar propósito acotado (seguridad del conductor, no scoring de productividad).

---

## Para profundizar

### Papers de review recomendados

- Ramzan et al., "A Survey on State-of-the-Art Drowsiness Detection Techniques", IEEE Access 2019.
- Sikander & Anwar, "Driver Fatigue Detection Systems: A Review", IEEE T-ITS 2019.
- Kaplan et al., "Driver Behavior Analysis for Safe Driving: A Survey", IEEE T-ITS 2015.
- Ortega et al., "DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis", ECCVW 2020.
- Eraqi et al., "Driver Distraction Identification with an Ensemble of Convolutional Neural Networks", J. Adv. Transp. 2019 (AUC dataset paper).
- Ghoddoosian et al., "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection", CVPRW 2019 (UTA-RLDD).
- Massoz et al., "The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use", WACV 2016.
- Martin et al., "Drive&Act: A Multi-modal Dataset for Fine-grained Driver Behavior Recognition", ICCV 2019.
- Wang et al., "100-Driver: A Large-Scale, Diverse Dataset for Distracted Driver Classification", IEEE T-ITS 2023.
- Koukiou & Anastassopoulos, "Drunk person identification using thermal infrared images", Int. J. Electron. Secur. Digit. Forensics 2012.
- Stuster, J., "The Detection of DWI at BACs Below 0.10", NHTSA Technical Report DOT HS 808 654, 1997 (y revisión 2010).

### Queries de búsqueda útiles

- `"driver drowsiness" dataset benchmark site:arxiv.org`
- `"distracted driver" dataset multimodal`
- `"driver monitoring" gaze zone annotation`
- `thermal imaging alcohol detection face`
- `PERCLOS benchmark dataset`
- Google Scholar: `"driver monitoring system" survey 2023..2025`
- Papers With Code: tag `driver-drowsiness-detection`, `distracted-driver-detection`

### Links de acceso (verificar en el momento)

- NTHU-DDD: http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- UTA-RLDD: https://sites.google.com/view/utarldd/home
- YawDD: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
- DROZY: http://www.drozy.ulg.ac.be/
- DMD (Vicomtech): https://dmd.vicomtech.org/
- Drive&Act: https://driveandact.com/
- State Farm: https://www.kaggle.com/c/state-farm-distracted-driver-detection
- AUC Distracted Driver: https://abouelnaga.io/projects/auc-distracted-driver-dataset/
- 100-Driver: repositorio de los autores (buscar "100-Driver dataset Wang")
- MRL Eye: http://mrl.cs.vsb.cz/eyedataset
- MPIIFaceGaze: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation
- CEW: http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html

### Benchmarks a consultar

- Papers With Code — "Driver Drowsiness Detection" leaderboard.
- Papers With Code — "Distracted Driver Classification".
- Sin leaderboard público para ebriedad — confirmar el gap.

---

_Documento generado como parte de la fase de investigación del proyecto Attention-Algorithm / DMS. Verificar enlaces y estado de datasets al momento de la entrega — varios proyectos académicos mueven hosting o retiran datos._
