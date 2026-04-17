# Face mesh, landmarks y fairness demográfica

> Documento de investigación 02 — Proyecto DMS (pivote desde Attention-Algorithm)
> Audiencia: equipo universitario, MVP pre-feria
> Estado: borrador para discusión interna

---

## TL;DR

- El sesgo demográfico en visión por computadora está **bien documentado para face detection y face recognition** (Gender Shades 2018, NIST FRVT Part 3 2019), pero la evidencia específica sobre **accuracy de face landmarks / mesh por tono de piel es escasa y dispersa**. Esta ausencia es, en sí misma, un hallazgo: nadie publica NME desagregado por Fitzpatrick de forma sistemática.
- Para un MVP de DMS en CPU/móvil, la frontera práctica hoy es **MediaPipe Face Landmarker v2** (478 landmarks + blendshapes + matriz de transformación 3D). Supera al FaceMesh legacy en blendshapes y es el camino soportado por Google.
- Alternativas serias: **MediaPipe Face Landmarker v2** para tiempo real, **InsightFace/SCRFD + 3DDFA_V2** si se busca 3D denso más robusto, **OpenFace 2.x** solo para análisis offline (no tiempo real en CPU modesta).
- El riesgo real en DMS no es tanto "el mesh no detecta caras oscuras" (MediaPipe detecta bien en la mayoría de condiciones), sino **degradación bajo iluminación lateral fuerte + piel oscura + lentes**, que es exactamente el escenario de un auto.
- Plan de validación barato y ejecutable: tomar ~500 imágenes de **FairFace** balanceadas por raza + etiquetar Fitzpatrick aproximado, medir **tasa de fallo de detección** y **NME de landmarks** por grupo, reportar disparidad máxima. Esto cabe en un fin de semana.
- Si aparece disparidad, mitigar con **CLAHE en canal L de Lab**, gamma adaptativo, y fallback a un segundo detector (SCRFD) antes de rediseñar el modelo.

---

## El problema: sesgo demográfico en visión por computadora

El antecedente canónico es **Gender Shades** (Buolamwini & Gebru, FAT* 2018): auditaron tres APIs comerciales de clasificación de género (IBM, Microsoft, Face++) sobre un dataset balanceado por tono de piel (PPB, escala Fitzpatrick). Encontraron hasta **34.4 puntos porcentuales** de diferencia en error entre hombres de piel clara y mujeres de piel oscura. El paper movió a IBM, Microsoft y Amazon a cambiar o pausar sus productos.

En 2019 NIST publicó el **FRVT Part 3: Demographic Effects** (Grother, Ngan, Hanaoka, NISTIR 8280), evaluando 189 algoritmos de reconocimiento facial sobre millones de imágenes. Resultado: **falsos positivos hasta 100× más altos** para rostros africanos y asiáticos orientales que para caucásicos en muchos algoritmos 1:1, y el efecto se agravaba en 1:N. Los algoritmos chinos rendían mejor en rostros asiáticos, sugiriendo que el dataset de entrenamiento domina.

Casos industriales recordables: cámaras HP con seguimiento facial que no detectaban rostros negros (2009), dispensadores automáticos de jabón que no detectaban manos oscuras, Twitter croppeando preferentemente caras blancas (2020, Twitter publicó su propio análisis).

**Nota importante para este documento**: casi toda esta literatura mide **detección** o **reconocimiento/identificación**, no **regresión de landmarks**. La extrapolación a face mesh no es automática: un modelo puede detectar la cara correctamente pero colocar mal los puntos alrededor de los ojos si no vio suficientes ejemplos de ese fenotipo.

---

## Frameworks de face mesh / landmarks — comparativa

Tiempos aproximados en CPU móvil/laptop modesta (Pixel 6, M1, o i5 laptop), no benchmarks formales — tomar como orden de magnitud. Runtime = inferencia por frame con cara ya detectada, salvo indicación.

| Framework | # landmarks | Modelo underlying | Licencia | Runtime CPU | Android | iOS | Fortalezas | Debilidades documentadas |
|---|---|---|---|---|---|---|---|---|
| **MediaPipe Face Landmarker v2** | 478 (incluye iris) | BlazeFace + FaceMesh V2 + Blendshape + FaceGeometry | Apache 2.0 | ~10-15 ms | Sí (nativo) | Sí | 52 blendshapes, matriz 3D, iris tracking, API unificada, soporte GPU/XNNPACK | Sesgo demográfico no auditado públicamente por Google con desglose; iris tracking puede fallar con lentes |
| **MediaPipe FaceMesh (legacy)** | 468 | BlazeFace + FaceMesh | Apache 2.0 | ~8-12 ms | Sí | Sí | Maduro, mucha documentación comunitaria, rápido | Deprecado en favor de v2; sin blendshapes nativos; sin auditoría fairness publicada |
| **OpenFace 2.x** (Baltrušaitis et al.) | 68 (CE-CLM) | Convolutional Experts CLM + HOG + SVM | BSD-like (académico) | ~40-80 ms | Difícil (C++) | Difícil | Action Units, gaze, head pose de alta calidad; transparente y documentado | No tiempo real en móvil; dependencia de dlib; rendimiento en piel oscura reportado como peor anecdóticamente |
| **Dlib 68-point** | 68 | Ensemble of Regression Trees (Kazemi & Sullivan 2014) | Boost SW | ~3-5 ms (tras detección) | Sí (portado) | Sí | Ligerísimo, sin GPU | Entrenado en iBUG 300-W (sesgado a rostros occidentales); falla en perfiles y oclusiones; sin 3D |
| **FAN (Face Alignment Network)** | 68 o 194 (2D/3D) | Stacked Hourglass | BSD | ~50-150 ms sin GPU | No nativo | No | SOTA histórico en 300-W, versión 3D disponible | Pesado, requiere GPU para tiempo real |
| **3DDFA_V2** (Guo et al. 2020) | 68 + malla densa ~38k | MobileNet + 3DMM regression | MIT | ~15-25 ms | Sí (ONNX) | Sí (ONNX) | 3D denso real, head pose estable, reconstrucción de forma | Entrenado mayormente en 300W-LP sintético; artefactos en perfiles extremos |
| **6DRepNet** (Hempel et al. 2022) | Solo head pose (6D) | RepVGG + 6D rotation | MIT | ~5-10 ms | Sí (ONNX) | Sí | SOTA en head pose, más estable que Euler/quaterniones | No da landmarks — complementa, no reemplaza |
| **MobileFaceNet** | Backbone, no landmarker por sí solo | MobileNetV2-like embedding | MIT | ~5 ms | Sí | Sí | Muy ligero como feature extractor | No produce landmarks; se usa con una cabeza de regresión custom |
| **InsightFace / SCRFD** | 5 o 106 (2d106det) | SCRFD detector + Arcface / 2d106det | MIT (código) | ~10-20 ms (SCRFD-500M) | Sí (ONNX) | Sí | Detector muy preciso, modelos ONNX, comunidad activa, el 106det es rápido y razonable | 106 landmarks menos densos que MediaPipe; modelos de alta calidad requieren GPU |
| **Google ML Kit Face Detection** | Contornos + 6-10 puntos clave | Blaze-family, cerrado | Free / propietaria Google | ~15 ms | Sí (nativo) | Sí | Integración trivial en Android/iOS | Pocos puntos para EAR preciso; cerrado; sin control |
| **Apple Vision VNDetectFaceLandmarksRequest** | ~76 puntos por región | Cerrado (Apple Neural Engine) | Propietaria Apple | ~5-10 ms ANE | No | Sí (excelente) | Calidad alta en dispositivos Apple recientes, gratis en iOS | Solo Apple; sin control del modelo; sin auditoría pública fairness |

**Lectura rápida**: si el target es Android + tiempo real + landmarks densos, **MediaPipe Face Landmarker v2** gana en la relación costo/beneficio. Si el target es análisis forense/offline, OpenFace es más interpretable. Si se quiere 3D denso, 3DDFA_V2.

---

## Evidencia publicada de sesgo en face landmarks

Aquí hay que ser honestos: **la literatura específica sobre fairness de regresión de landmarks es escasa y mucho menos sistemática que la de detección/reconocimiento**. Lo que hay:

1. **Dooley et al., "Robustness Disparities in Face Detection" (NeurIPS 2022)**. Evaluaron detectores (incluido MediaPipe BlazeFace) sobre corruptions y encontraron que la **tasa de detección cae más para rostros de piel oscura y femeninos bajo ruido/blur/iluminación**. Es el paper más cercano a "MediaPipe tiene sesgo". Mide detección, no landmarks.

2. **Yucer et al., "Measuring Hidden Bias within Face Recognition via Racial Phenotypes" (WACV 2022)**. Confirma que modelos entrenados en datasets occidentales tienen peor rendimiento en fenotipos africanos y asiáticos del este, y que el sesgo persiste aunque los datasets estén "balanceados por etiqueta de raza".

3. **Khan et al., "A Comprehensive Study on Face Recognition Biases Beyond Demographics" (TTS 2022)**. Desagrega por atributos no-demográficos (lentes, barba, iluminación) y muestra interacción con piel.

4. **Meta Casual Conversations v2 (2023)**. Dataset con auto-reporte de ancestralidad, género, edad y Fitzpatrick. Meta lo lanzó precisamente porque los datasets existentes no permiten auditar modelos de forma responsable. Usarlo para auditar un landmarker NO está publicado ampliamente todavía.

5. **MediaPipe específicamente**: Google publicó la **Model Card de Face Landmarker** (disponible en el repo mediapipe-solutions), que menciona evaluación en 14 regiones geográficas y balance de género/tono, pero **no publica las tablas de NME desagregado** que un investigador querría. Se reconoce limitación, no se cuantifica públicamente en el detalle necesario.

6. **OpenFace**: Baltrušaitis et al. en el paper original de OpenFace 2.0 (FG 2018) reportan rendimiento en 300-W, Menpo, 300VW — **ninguno de esos datasets está balanceado por tono de piel**. iBUG 300-W está fuertemente sesgado a rostros caucásicos adultos.

7. **Dlib 68**: el shape predictor de Dlib fue entrenado en iBUG 300-W. **No hay auditoría fairness publicada** por el mantenedor. Reportes anecdóticos (GitHub issues, blogs) mencionan peor rendimiento en rostros oscuros y en perfiles, consistente con el sesgo del dataset de entrenamiento.

**Conclusión de esta sección**: no existe, a la fecha de este documento, un benchmark público autoritativo de "MediaPipe vs OpenFace vs 3DDFA_V2 en NME por Fitzpatrick". Esto es una oportunidad (y una responsabilidad) del equipo: publicar aunque sea un microestudio honesto vale más que repetir el claim de "usa MediaPipe, es robusto".

---

## Datasets para validar fairness

| Dataset | Tamaño | Etiquetas demográficas | Licencia / acceso | Uso recomendado |
|---|---|---|---|---|
| **FairFace** (Kärkkäinen & Joo 2021) | ~108k | Raza (7 grupos), género, edad | CC BY 4.0, descarga libre | Baseline inmediato, balanceado por raza por diseño |
| **Casual Conversations v2** (Meta 2023) | ~26k videos, 5k participantes | Auto-reporte: ancestralidad, género, edad, Fitzpatrick I-VI | Licencia restrictiva Meta, formulario | Gold standard para fairness; vale el esfuerzo de registro |
| **BUPT-Balancedface** (Wang et al.) | ~1.3M imágenes, 28k identidades | 4 grupos raciales balanceados | Académico, formulario | Pretraining/validación a escala |
| **RFW — Racial Faces in the Wild** (Wang et al. 2019) | ~40k caras, 12k identidades | 4 grupos raciales balanceados, test 1:1 | Académico, formulario | Benchmark estándar para racial bias en FR |
| **UTKFace** | ~20k | Edad, género, etnia (5 grupos) | Solo investigación | Ligero, útil para prototipo |
| **Fitzpatrick 17k** (Groh et al. 2021) | ~17k | Fitzpatrick I-VI (anotado por dermatólogos) | CC BY-NC | Validación de estimador de Fitzpatrick; imágenes clínicas, no rostros — usar con cuidado |
| **VGGFace2 demographic subsets** | Subconjuntos de VGGFace2 con etiquetas externas | Género, edad aproximada | Académico | Volumen, no balanceado nativamente |
| **PPB — Pilot Parliaments Benchmark** (Gender Shades) | 1270 | Fitzpatrick + género | Académico | Clásico, pequeño pero icónico |
| **DiveFace** | ~24k | 6 grupos demográficos | Académico | Alternativa a BUPT |

**Para el MVP del equipo**: FairFace es el punto de entrada más barato (descarga directa, etiquetas de raza ya listas). Si se quiere desagregar por Fitzpatrick, se puede (a) pedir Casual Conversations v2, o (b) entrenar un clasificador Fitzpatrick aproximado sobre Fitzpatrick 17k y aplicarlo a FairFace con la debida humildad epistémica.

---

## Pipeline de validación recomendado para el proyecto

Objetivo: antes de la feria, poder afirmar con datos "el sistema rinde igual (o no) para los 6 grupos Fitzpatrick". Presupuesto: un fin de semana, una laptop.

### Pasos

**(a) Construir un test set balanceado.** Tomar FairFace validation (~11k) y submuestrear ~50-100 imágenes por grupo racial (las 7 clases FairFace). Si se quiere Fitzpatrick explícito, estimarlo con ITA (Individual Typology Angle) sobre parches de piel en región de frente + mejillas:
- Pasar imagen a espacio Lab
- Tomar mediana de L y b en parches de piel
- Calcular ITA = arctan((L-50)/b) * 180/pi
- Bin: ITA > 55 Fitz I, 41-55 II, 28-41 III, 10-28 IV, -30-10 V, < -30 VI

**(b) Definir ground truth.** FairFace no trae landmarks. Opciones:
- Anotar manualmente ~20 puntos en 50 imágenes por grupo (caro pero honesto)
- Usar un modelo "teacher" fuerte (FAN 3D en GPU) como proxy, sabiendo que introduce el sesgo del teacher. Menos honesto pero más rápido.
- Usar un subset de 300W/Menpo/WFLW que tenga landmarks ground truth y etiquetar Fitzpatrick con ITA — este es probablemente el mejor camino intermedio.

**(c) Métricas.** Para cada grupo g:
- **Detection failure rate**: fracción de imágenes donde el detector no devuelve cara.
- **NME (Normalized Mean Error)**: error medio en landmarks, normalizado por distancia inter-ocular o diagonal del bounding box.
- **AUC de CED (Cumulative Error Distribution)** hasta NME=0.08.
- **Disparity ratio**: max(NME_g) / min(NME_g). Si > 1.2, investigar.

**(d) Reporte.** Tabla con NME por grupo + intervalo de confianza bootstrap (1000 resamples) + disparity ratio. Gráfico de CED por grupo superpuesto.

### Pseudocódigo Python

```python
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

mp_face = mp.tasks.vision.FaceLandmarker
# cargar modelo v2: face_landmarker_v2_with_blendshapes.task

def ita_from_skin_patch(img_bgr, patch_bbox):
    x, y, w, h = patch_bbox
    patch = img_bgr[y:y+h, x:x+w]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = np.median(lab[..., 0]) * 100.0 / 255.0
    b = np.median(lab[..., 2]) - 128.0
    return np.degrees(np.arctan2(L - 50.0, b))

def fitzpatrick_bin(ita):
    if ita > 55:   return 1
    if ita > 41:   return 2
    if ita > 28:   return 3
    if ita > 10:   return 4
    if ita > -30:  return 5
    return 6

def nme(pred, gt, norm):
    # pred, gt: (N, 2); norm: escalar (p.ej. dist inter-ocular)
    return np.mean(np.linalg.norm(pred - gt, axis=1)) / norm

def interocular(gt):
    # asumir indices 36 y 45 en convencion 68-pt
    return np.linalg.norm(gt[36] - gt[45])

results = {g: {"fail": 0, "nme": []} for g in range(1, 7)}
for img_path, gt_landmarks, skin_patch in dataset:
    img = cv2.imread(str(img_path))
    g = fitzpatrick_bin(ita_from_skin_patch(img, skin_patch))
    pred = run_mediapipe_v2(img)   # devuelve (478,2) o None
    if pred is None:
        results[g]["fail"] += 1
        continue
    pred_68 = mediapipe_to_68(pred)   # mapear los 478 a los 68 estandar
    err = nme(pred_68, gt_landmarks, interocular(gt_landmarks))
    results[g]["nme"].append(err)

for g, r in results.items():
    arr = np.array(r["nme"])
    print(f"Fitz {g}: n={len(arr)}, fail_rate={r['fail']/(r['fail']+len(arr)):.3f}, "
          f"NME_mean={arr.mean():.4f}, NME_p90={np.percentile(arr, 90):.4f}")
```

Un par de decisiones de diseño a documentar en el reporte:
- ITA es una aproximación, no reemplaza Fitzpatrick anotado por dermatólogo.
- El mapeo MediaPipe 478 → 68 estilo iBUG introduce un error sistemático; usar siempre los mismos índices y reportarlos.
- Para comparar frameworks con diferentes convenciones de landmarks, fijar un subconjunto común (ej: 5 puntos: ojos, nariz, esquinas de boca) y comparar en ese subconjunto.

---

## Mitigaciones si se detecta sesgo

En orden de costo creciente:

1. **Preprocesado de iluminación**. Antes de inferencia:
   - **CLAHE sobre canal L** de Lab (no sobre BGR directo — deforma color).
   - **Corrección gamma adaptativa** basada en brillo medio del rostro.
   - **White balance** por grey-world o por parche de piel conocido.
   Esto ayuda desproporcionadamente más a rostros oscuros bajo luz lateral, que es el caso de peor rendimiento típico.

2. **Fallback a detector secundario**. Si MediaPipe no detecta cara durante N frames, probar con **SCRFD-500M** (ONNX, liviano). Si SCRFD encuentra cara, pasar el crop a un landmarker separado (3DDFA_V2 o MediaPipe con ROI explícita).

3. **Ensamble de landmarkers**. Correr MediaPipe v2 + 3DDFA_V2 y promediar landmarks comunes (los 68 iBUG). Costoso en compute, pero reduce varianza.

4. **Temporal smoothing agresivo**. Un filtro de Kalman o one-euro filter sobre landmarks estabiliza mucho cuando el detector "parpadea" en frames difíciles — caso típico en piel oscura bajo luz de día lateral.

5. **Re-entrenamiento/fine-tuning**. Solo si 1-4 no alcanzan. Requiere dataset balanceado con landmarks ground truth (caro de anotar). En ese caso, **fine-tunear la cabeza de regresión** del modelo sobre un mix balanceado, congelando el backbone.

6. **Cambiar el modelo de decisión downstream, no el modelo visual**. Si EAR sufre más varianza en un grupo, calibrar el **umbral de somnolencia por persona** (baseline de los primeros 30 segundos de conducción). Esto es más justo y más efectivo que pelear con el landmarker.

---

## Recomendación concreta para el MVP

**Decisión propuesta**:

- **Framework**: MediaPipe Face Landmarker v2 (no el legacy FaceMesh). Razones: 478 landmarks, blendshapes (útiles si se extiende a detección de yawning, frowning), matriz de transformación 3D (útil para head-turn preciso), soporte oficial activo, Apache 2.0, CPU-friendly.
- **Preprocesado**: CLAHE en canal L + corrección gamma adaptativa, activado condicionalmente cuando el brillo medio del ROI facial cae bajo un umbral. Esto no degrada casos buenos y ayuda en los malos.
- **Head pose**: usar la matriz 3D nativa de Face Landmarker v2, no estimación PnP manual con los 6 puntos de FaceMesh legacy — es más estable.
- **EAR**: seguir con la fórmula estándar pero calibrar umbral por-persona en los primeros 30s (baseline) en lugar de umbral fijo global. Esto mitiga parte del sesgo sin tocar el modelo.
- **Validación pre-feria**: ejecutar el pipeline de la sección 7 sobre un FairFace subset de ~500 imágenes. Reportar disparity ratio en el poster/demo como parte de la honestidad científica del proyecto.
- **Contingencia**: tener SCRFD-500M ONNX listo como fallback si aparecen casos problemáticos en la demo en vivo (iluminación del stand de la feria es impredecible).

**No recomendado para MVP**:
- OpenFace (no tiempo real en la laptop promedio del equipo).
- Dlib 68 (sesgo del dataset de entrenamiento está bien documentado, sin iris).
- 3DDFA_V2 solo (más pesado, menos documentación Android; sí como fallback/teacher).

**Hipótesis falsable**: "MediaPipe Face Landmarker v2 + CLAHE logra disparity ratio de NME < 1.2 entre Fitzpatrick I-II y V-VI sobre un subset de FairFace de 500 imágenes". Esto es testeable, y si falla, justifica el fallback.

---

## Para profundizar

### Papers clave
- Buolamwini & Gebru, "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification", FAT* 2018. https://proceedings.mlr.press/v81/buolamwini18a.html
- Grother, Ngan, Hanaoka, "Face Recognition Vendor Test (FRVT) Part 3: Demographic Effects", NISTIR 8280, 2019. https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf
- Dooley et al., "Robustness Disparities in Face Detection", NeurIPS 2022. https://arxiv.org/abs/2211.15937
- Kärkkäinen & Joo, "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age", WACV 2021. https://arxiv.org/abs/1908.04913
- Hazirbas et al., "Casual Conversations v2", Meta 2023. https://arxiv.org/abs/2303.04838
- Grgic et al., "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs" (MediaPipe FaceMesh paper), CVPR 2019 workshop. https://arxiv.org/abs/1907.06724
- Baltrušaitis et al., "OpenFace 2.0: Facial Behavior Analysis Toolkit", FG 2018. https://ieeexplore.ieee.org/document/8373812
- Guo et al., "Towards Fast, Accurate and Stable 3D Dense Face Alignment" (3DDFA_V2), ECCV 2020. https://arxiv.org/abs/2009.09960
- Hempel et al., "6D Rotation Representation for Unconstrained Head Pose Estimation" (6DRepNet), ICIP 2022. https://arxiv.org/abs/2202.12555
- Yucer et al., "Measuring Hidden Bias within Face Recognition via Racial Phenotypes", WACV 2022. https://arxiv.org/abs/2110.09839

### Repos y recursos
- MediaPipe Tasks — Face Landmarker: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- InsightFace (SCRFD, 2d106det): https://github.com/deepinsight/insightface
- OpenFace 2: https://github.com/TadasBaltrusaitis/OpenFace
- 3DDFA_V2: https://github.com/cleardusk/3DDFA_V2
- 6DRepNet: https://github.com/thohemp/6DRepNet
- FairFace dataset: https://github.com/joojs/fairface
- Casual Conversations v2: https://ai.meta.com/datasets/casual-conversations-v2-dataset/
- BUPT-Balancedface: http://www.whdeng.cn/RFW/Trainingdataste.html
- RFW: http://www.whdeng.cn/RFW/index.html
- Fitzpatrick 17k: https://github.com/mattgroh/fitzpatrick17k
- face-alignment (FAN): https://github.com/1adrianb/face-alignment

### Queries útiles para seguir investigando
- `"face landmark" fairness fitzpatrick NME`
- `"facial landmark" demographic bias benchmark`
- `MediaPipe face mesh skin tone evaluation`
- `"3DDFA" OR "FAN" cross-ethnicity evaluation`
- `drowsiness detection EAR "skin tone" OR "dark skin"`
- `DMS "driver monitoring" bias dataset`
- `NIST FRVT landmark OR mesh demographic`

### Lecturas laterales útiles para el proyecto
- Raji & Buolamwini, "Actionable Auditing", AIES 2019 — cómo las auditorías externas cambian productos.
- Mitchell et al., "Model Cards for Model Reporting", FAT* 2019 — plantilla para documentar el modelo final del MVP de forma honesta.
- Regla europea AI Act (2024) — DMS está en la categoría de alto riesgo; aunque el proyecto sea universitario, vale alinear vocabulario con el regulador.
