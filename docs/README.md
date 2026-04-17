# Research Brief — Attention-Algorithm / DMS

*Síntesis ejecutiva · Abril 2026 · branch `research/mvp-direction`*

Este directorio contiene la investigación realizada para aterrizar la dirección del proyecto **Attention-Algorithm** (LEIA-qro), que está pivotando de app de estudio a **Driver Monitoring System (DMS)**. El MVP actual es un único `main.py` con OpenCV + MediaPipe FaceMesh que calcula EAR (Eye Aspect Ratio) y un head-turn score simple sobre webcam.

El objetivo inmediato es **presentar en la feria de ingenierías** y al mismo tiempo responder las preguntas abiertas del equipo sobre innovación, integrabilidad, viabilidad comercial, sesgo demográfico y datasets.

---

## Estructura

| # | Documento | Pregunta que responde |
|---|---|---|
| 01 | [Landscape competitivo](01-competitive-landscape.md) | ¿Qué tan innovador puede ser vs. DMS comerciales existentes? |
| 02 | [Face mesh y fairness](02-face-mesh-and-fairness.md) | ¿Qué framework es robusto entre tonos de piel / razas? |
| 03 | [Datasets reales](03-datasets-real.md) | ¿Qué datasets existen para somnolencia / distracción / ebriedad? |
| 04 | [Datos sintéticos](04-synthetic-data.md) | ¿Es viable generar datos con Unity / Unreal / Omniverse? |
| 05 | [Integración y deployment](05-integration-and-deployment.md) | ¿Qué tan integrable es en una app o embebido? |
| 06 | [Negocio y regulación](06-business-and-regulation.md) | ¿Es vendible a una armadora? ¿a quién más? |

---

## TL;DR por pregunta del equipo

### ¿Qué tan innovador es?

**Técnicamente, poco.** EAR + head-turn + MediaPipe es el MVP canónico académico y no compite con Seeing Machines, Smart Eye, Bosch o Valeo, que usan DNNs propias con NIR. **Estratégicamente, puede serlo** si se posiciona en gaps abiertos que los incumbentes no atacan: detección de ebriedad por visión pura, mercado LATAM de bajo costo, form factor "celular en parabrisas", coaching con LLM, explicabilidad de la alerta. Ver [01](01-competitive-landscape.md).

### ¿Qué tan integrable en una app?

**Muy integrable en Android/iOS.** MediaPipe Tasks tiene SDK nativo para Face Landmarker v2 con latencia 30-70 ms en un Pixel 9 Pro. El port desde el MVP es ~2-3 semanas. Lo que **no es viable** para un proyecto universitario es embebido automotriz real (Jetson, Snapdragon Ride, TDA4VM) — requiere cámara IR calificada, toolchains cerrados, ISO 26262. Para la feria: laptop como demo principal, APK opcional. Ver [05](05-integration-and-deployment.md).

### ¿Vendible a una armadora?

**No, realistamente.** Los ciclos de homologación OEM son 3-5 años, requieren ASPICE L2-L3, ISO 26262 ASIL-B y millones de km de validación. Seeing Machines tardó ~15 años en llegar. **Sí vendible** a: flotillas after-market LATAM (MXN 200-300/veh/mes bajo el piso de Netradyne/Samsara), aseguradoras con UBI (Qualitas, GNP), transporte público / escolar, gobierno. El driver regulatorio clave es **EU GSR 2019/2144** (DAW obligatorio desde 2024, ADDW desde julio 2026). México no exige DMS — eso es oportunidad. Ver [06](06-business-and-regulation.md).

### ¿Preocupación por sesgo racial / tono de piel?

**La preocupación es legítima pero mayormente teórica hasta que se mida.** La literatura de fairness cubre detección y reconocimiento (Gender Shades, NIST FRVT), no regresión de landmarks. MediaPipe no publica NME desagregado por Fitzpatrick. **Plan barato**: validar con ~500 imágenes de FairFace en un fin de semana (pseudocódigo incluido en [02](02-face-mesh-and-fairness.md)). Si aparece disparidad, mitigar con CLAHE + gamma adaptativo + fallback a SCRFD antes de tocar el modelo. **Recomendación**: MediaPipe **Face Landmarker v2** (no el legacy), con preprocesado de iluminación y calibración per-persona del umbral EAR.

### ¿Hay datasets unificados? ¿Hay que generar sintético?

**Hipótesis del equipo confirmada**: no existe un dataset público que cubra los tres estados (alerta / distraído / ebrio). Somnolencia está bien cubierta (NTHU-DDD, UTA-RLDD, DROZY), distracción también (State Farm, AUC, DMD, 100-Driver), **ebriedad está prácticamente vacía por razones éticas**. Ver [03](03-datasets-real.md). Los sintéticos (Unity Perception, Unreal MetaHuman, Omniverse Replicator) son **parcialmente viables**: sirven como pretraining + fine-tuning con datos reales, no como reemplazo. Tienen sim-to-real gap real en iluminación IR y micro-expresiones. Para un equipo universitario, un mini-pipeline de 500-2000 clips enfocado en ebriedad y balance demográfico es factible en 4-6 semanas. Ver [04](04-synthetic-data.md).

---

## Recomendación consolidada

### Para la feria (ahora)

**Posicionamiento:** no "competidor de Seeing Machines", sino **"DMS accesible para el 95% de autos que circulan en México y nunca tendrán un sistema OEM"**.

**Demo:**
- Laptop + webcam, MVP actual mejorado a MediaPipe Face Landmarker v2 (no el legacy) + CLAHE + calibración per-persona.
- Overlay con **explicabilidad de la alerta** ("cerraste los ojos 1.8s, giraste la cabeza 45° 3 veces en 1 min") — diferenciador vs. cualquier app del Play Store.
- Mini-estudio de fairness con FairFace (sección 7 de [02](02-face-mesh-and-fairness.md)) — 1 gráfico en el poster.
- Opcional: APK Android con MediaPipe Tasks — 2-3 semanas de esfuerzo extra, wow-factor alto.

**Narrativa:**
1. Problema: distracción y fatiga son causa # 1 de accidentes viales en México; ningún auto legacy tiene DMS.
2. Solución: detección por visión en el celular, sin hardware adicional, con explicación de la alerta.
3. Diferenciación: LATAM, bajo costo, on-device (privacidad), coaching post-viaje.
4. Honestidad: limitaciones conocidas (solo diurno, fairness medida, no sustituye alcoholímetro).

### Post-feria — roadmap en 3 fases

**Fase 1 (0-2 meses):** consolidar MVP, paper corto, GitHub público, mini-auditoría de fairness.
**Fase 2 (2-12 meses):** piloto con 1 flotilla local (reparto / transporte escolar en Querétaro), recolección de datos reales con consentimiento, validación con la flotilla. Posible spin-off a MXN 200-300/veh/mes.
**Fase 3 (1-3 años):** spin-off universitario, fondos (CONAHCYT, aceleradoras), partnership con aseguradora mexicana para UBI, expandir a detección de impairment como línea de investigación diferenciada.

---

## Ángulo de investigación pendiente

El módulo más **novedoso académicamente** y que justifica publicación es **detección de impairment (fatiga severa / sospecha de ebriedad) por visión pura**. No hay dataset público viable ([03](03-datasets-real.md)), por lo que requiere sintético + proxy con features (nistagmo, variabilidad de parpadeo, gaze errático, ptosis). Tratar como prueba de concepto, **nunca** como diagnóstico — riesgo legal alto.

---

## Decisiones técnicas tomadas a partir de esta investigación

| Decisión | Justificación | Documento |
|---|---|---|
| Migrar a **MediaPipe Face Landmarker v2** (no legacy) | 478 landmarks, blendshapes, matriz 3D, soporte oficial | [02](02-face-mesh-and-fairness.md) |
| Agregar **CLAHE en canal L + gamma adaptativo** antes de inferencia | Mitiga degradación en piel oscura + luz lateral | [02](02-face-mesh-and-fairness.md) |
| **Calibrar umbral EAR per-persona** (baseline 30s) | Más justo que umbral global, mitiga parte del sesgo downstream | [02](02-face-mesh-and-fairness.md) |
| Dataset de **somnolencia**: UTA-RLDD + NTHU-DDD | Mejor cobertura real-world + simulador | [03](03-datasets-real.md) |
| Dataset de **distracción**: DMD + State Farm | DMD cubre gaze zone, State Farm cubre clasificación | [03](03-datasets-real.md) |
| **Ebriedad**: proxy por features + sintético demostrativo | No hay dataset público; declarar límite | [03](03-datasets-real.md), [04](04-synthetic-data.md) |
| Target deployment: **app Android con MediaPipe Tasks** | Único path realista fuera de laptop para feria | [05](05-integration-and-deployment.md) |
| **No intentar** Jetson / Snapdragon / certificación OEM | Fuera de scope universitario | [05](05-integration-and-deployment.md), [06](06-business-and-regulation.md) |
| Modelo de negocio target: **flotillas LATAM + UBI** | OEM inviable para estudiantes; fleet accesible | [06](06-business-and-regulation.md) |

---

## Siguientes pasos sugeridos

1. Migrar el `main.py` a Face Landmarker v2 API (reemplaza FaceMesh legacy).
2. Añadir CLAHE + calibración EAR per-persona.
3. Correr el pipeline de validación de fairness sobre FairFace (fin de semana).
4. Preparar overlay de explicabilidad y poster de feria.
5. (Opcional) Port a Android con MediaPipe Tasks.
6. Decidir si se incluye el módulo exploratorio de impairment en la demo.

Cada documento incluye una sección **"Para profundizar"** con links, papers y queries de búsqueda para retomar el tema con más profundidad de forma agéntica cuando sea necesario.
