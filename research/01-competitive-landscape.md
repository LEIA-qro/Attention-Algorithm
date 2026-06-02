# Landscape competitivo y gap de innovación

*Documento de investigación — Proyecto Attention-Algorithm (LEIA-qro) · Abril 2026*
*Pivote: app de estudio → Driver Monitoring System (DMS) RGB-only de bajo costo*

---

## TL;DR

- **El mercado DMS está consolidado en el Tier-1 OEM**: Seeing Machines y Smart Eye se llevan la gran mayoría de los design-wins con OEMs globales (GM, Ford, BMW, Mercedes, Volvo, Stellantis, Nissan, etc.). Juntas reportan >150 modelos de vehículo en producción o en pipeline hacia 2026-2028. El driver ya no es si hay DMS, sino qué proveedor lo provee.
- **EU GSR 2022/1426 + Euro NCAP 2023-2026** son el catalizador regulatorio: desde julio 2024 todo auto nuevo vendido en la UE debe llevar detección de somnolencia y distracción, y desde 2026 se endurece (DDAW + ADDW). Esto es lo que está moviendo el mercado hoy, no la demanda del consumidor.
- **El stack dominante del OEM es IR NIR + cámara dedicada en la columna de dirección o rearview**, no RGB. Smart Eye y Seeing Machines ya no usan landmarks tradicionales sino DNNs propias entrenadas con millones de horas de video etiquetado. MediaPipe FaceMesh + EAR es lo que hace la academia y los MVPs, no lo que va al auto.
- **After-market flotillero (Netradyne, Samsara, Lytx, Motive, Nauto) es un mercado gigantesco y distinto**: $30-60 USD/vehículo/mes, detección de eventos con IA en la nube, scoring de conductor. Ahí la visión es RGB + dual-facing, y el valor es insurance/liability, no safety en tiempo real.
- **Hay tres gaps reales explotables para un proyecto universitario**: (1) **detección de ebriedad por visión pura** — hoy casi nadie lo vende como feature (Seeing Machines tiene algo en pipeline, Smart Eye ofrece "impairment" pero limitado); (2) **LATAM / México**, donde ningún flotillero tiene producto localizado ni precio accesible; (3) **RGB-only sobre celular montado en parabrisas**, un form-factor que ni OEM ni flotilleros cubren porque no es su negocio.
- **El MVP actual (EAR + head-turn + MediaPipe) no compite técnicamente con Seeing Machines, y no tiene que hacerlo**. El argumento de feria no es "mejor que FOVIO" sino "accesible para el 95% de los autos que circulan en México y que nunca van a tener un DMS OEM". Ese encuadre es defendible.

---

## DMS comerciales (Tier-1 / suppliers)

| Empresa | Producto | Somnolencia | Distracción | Gaze | Emoción | Ebriedad / impairment | Modalidad | Clientes conocidos (OEMs) | Notas de precio / escala |
|---|---|---|---|---|---|---|---|---|---|
| **Seeing Machines** (ASX:SEE, AU) | FOVIO chip + Guardian Gen3 (truck/flotilla) + plataforma automotive | Sí (PERCLOS, microsueño) | Sí (mirada fuera de carretera, uso de celular) | Sí (6-DOF head + eye gaze preciso) | Parcial | **En roadmap** (impairment detection anunciado 2024-2025) | NIR + cámara dedicada | GM, Ford, Mercedes-Benz, BMW, Stellantis (Magneti Marelli tier integrator), Mitsubishi Fuso. Reportaron **>4.7M vehículos OEM en producción** a FY2024 y pipeline de >13M unidades. Revenue FY2024 ~USD 62M | No se vende suelto a consumidor. Integrado vía Tier-1; precio BOM estimado $15-40 USD por vehículo en volumen |
| **Smart Eye** (STO:SEYE, SE) — adquirió **Affectiva** 2021 e **iMotions** 2022 | Automotive Interior Sensing (AIS) = DMS + OMS (occupant monitoring) + emoción | Sí | Sí | Sí (gold-standard eye-tracking, herencia de research) | **Sí** (Affectiva Emotion AI, único diferenciador real) | Parcial ("cognitive load", fatiga; no alcoholemia directa) | NIR principal, soporta RGB en OMS | BMW, Mercedes, Stellantis, Porsche, Geely/Lotus, Polestar, Honda. Reportan **>120 modelos en pipeline** y >100 design wins acumulados | Revenue 2024 ~SEK 400M (~USD 38M). Similar: integrado vía OEM, no retail |
| **Bosch** | Interior Monitoring System (parte de la plataforma Vehicle Computer) | Sí | Sí | Sí | Limitado | No (pero sí detección de cinturón, ocupantes, niño olvidado) | NIR + ToF en algunos módulos | Tier-1 clásico, integración con Audi, Volkswagen Group, varios chinos | No publicado. Se vende como parte de plataforma E/E completa |
| **Valeo** | Driver Monitoring + Cabin Monitoring (parte de "Smart Cocoon") | Sí | Sí | Sí | Básico | No | NIR | Renault, Stellantis, varios chinos (Geely, NIO componentes) | Integrado con SDV platform |
| **Cipia** (antes Eyesight Technologies, IL) | Cipia-FS10 (after-market flotilla), Driver Sense, Cabin Sense (OEM) | Sí | Sí | Sí | No | No | RGB + NIR | Mitsubishi Motors (design win 2022), varios OEMs chinos (Leapmotor reportado) | FS10 retail ~USD 300-500 instalado, modelo SaaS mensual para flotillas |
| **Jungo Connectivity** (IL, spinoff de Cadence) | CoDriver SDK (software puro) | Sí | Sí | Sí | Sí (emoción básica) | Limitado | RGB o NIR, software-agnóstico | Licenciado por varios Tier-1s; relación histórica con Visteon | SDK licensing, no hardware |
| **Tobii** (STO:TOBII, SE) | Eye-tracking automotive (vía subsidiaria) | Sí | Sí | **Sí (best-in-class gaze)** | No | No | NIR | Más fuerte en automotive research / XR que en producción OEM masiva | - |
| **Mitsubishi Electric** | Driver Status Monitor | Sí | Sí | Parcial | No | No | NIR | Integra en vehículos japoneses; también vende a flotilleros vía ADAS kit | - |
| **Aisin** (JP) | DMS integrado en rearview / cluster | Sí | Sí | Parcial | No | No | NIR | Toyota principalmente | Integrado |
| **Continental** | Driver Identification + DMS (parte de cockpit HPC) | Sí | Sí | Sí | No | No | NIR + cámara 3D opcional | VW Group, varios | Integrado |
| **Magna** | Eye Gaze / DMS en espejo retrovisor inteligente | Sí | Sí | Sí | No | No | NIR | OEMs NA | Integrado en módulos de retrovisor |

**Lectura clave de la tabla:** el eje de competencia del Tier-1 es (a) gaze accuracy, (b) robustez con lentes de sol / oscuridad / distintos fenotipos, (c) integración con ADAS L2+/L3 hand-over. **Nadie compite en precio ni en accesibilidad.** La emoción está monopolizada por Smart Eye (vía Affectiva). La ebriedad/impairment está abierta.

---

## Soluciones after-market / flotillas

Mercado distinto al OEM: se monta en el parabrisas, incluye cámara hacia el conductor + hacia afuera, y la propuesta de valor es reducir siniestros + defender al conductor ante litigios ("exoneration video"). La IA corre parcialmente en el edge y parcialmente en la nube.

| Empresa | Producto | Features DMS | Modalidad | Cliente típico | Precio público aproximado |
|---|---|---|---|---|---|
| **Netradyne** (US/IN) | Driveri D-450 / D-510 | Somnolencia, distracción (celular, comer, fumar), cinturón, scoring positivo ("GreenZone") | Dual RGB + NIR de noche, edge AI (Qualcomm) | Flotillas medianas/grandes US, expansión LATAM | **~USD 40-60 / vehículo / mes** (contrato 3 años típico, hardware incluido) |
| **Samsara** (NYSE:IOT) | AI Dash Cam CM31 / CM32 | Somnolencia, distracción, tailgating, señales | Dual RGB, edge AI | Flotillas enterprise (logística, construcción) | **~USD 27-40 / vehículo / mes** + hardware ~USD 400-800. Revenue FY25 >USD 1.2B |
| **Lytx** (US, privado) | DriveCam + Machine Vision + AI (MV+AI) | Somnolencia, distracción, seguimiento, reglas personalizadas | Dual RGB | Flotillas grandes, el jugador más viejo (fundado 1998) | **~USD 30-50 / vehículo / mes**. >1M vehículos bajo servicio reportados |
| **Motive** (ex-KeepTruckin, US) | AI Dashcam | Somnolencia, distracción, celular, comer | Dual RGB, edge | Flotillas trucking US (muy fuerte en ELD) | **~USD 30-45 / vehículo / mes** |
| **Nauto** (US) | Nauto VERA + N3 camera | Predicción de colisión, DMS, VERA behavioral AI | Dual RGB | Flotillas comerciales, insurance partnerships | **~USD 40-70 / vehículo / mes** |

**Observaciones:**
- Todos son **RGB-only o RGB+NIR híbrido** — es decir, el argumento "RGB no sirve para DMS serio" es falso; el OEM prefiere NIR por robustez ante lentes de sol y sol directo, pero el after-market vive de RGB.
- Todos operan en **USD 30-60/vehículo/mes**. Para un transportista mexicano con 20 camiones, eso son USD 600-1200/mes = ~MXN 12,000-24,000/mes. **Prácticamente ninguna flotilla de PyME en México paga eso.**
- Ninguno tiene producto en español-México localizado, facturación en MXN, ni distribuidor fuerte en LATAM. Netradyne es el más avanzado en México (ya opera), los demás venden vía partners.

---

## Open source / académico

La academia sigue dos ramas: (a) landmarks clásicos + heurísticas (EAR, MAR, PERCLOS, head-pose) — lo que usa el MVP actual; (b) DL end-to-end (CNN, transformers, multimodal).

**Datasets públicos relevantes:**
- **NTHU-DDD** (National Tsing Hua University Driver Drowsiness Detection) — clásico, 36 sujetos, escenarios day/night/glasses/sunglasses.
- **DMD (Driver Monitoring Dataset)** — Vicomtech, multimodal (RGB + depth + IR), 41 conductores, distracción + somnolencia. Muy usado 2021+.
- **YawDD** — bostezos, benchmark estándar.
- **UTA-RLDD** — Real-Life Drowsiness Dataset (60 sujetos, video largo).
- **DriverMHG** — Multi-modal hand gesture, útil para distracción (manos fuera del volante).
- **AUC Distracted Driver v1/v2** — clasificación de posturas de distracción.
- **100-Driver** (2023) — dataset grande, 470h, distracción; importante para benchmarks recientes.

**Trabajos recientes 2023-2026 (muestra representativa):**
- Papers que usan MediaPipe FaceMesh + EAR/MAR + SVM/LSTM — abundan en conferencias menores (IEEE Access, MDPI Sensors). Son esencialmente MVPs académicos equivalentes al del proyecto. Útiles como benchmarks de referencia; no son estado del arte.
- Transformers para driver action recognition (TimeSformer, VideoMAE fine-tuned) — empiezan a dominar papers de alto impacto 2024-2025.
- Multimodal fusion (face + pose + volante + CAN bus) — línea fuerte en CVPR/ITSC.
- **Detección de alcohol/drogas por visión**: existe literatura preliminar (ojos rojos, ptosis, temblor ocular, nistagmo, variabilidad de pupila), pero **poquísimos papers con datasets éticamente viables**. Es un hueco académico real, no solo comercial.
- Explicabilidad (XAI) sobre DMS — línea incipiente 2024-2026, interesante como narrativa de feria.

**Proyectos open-source / GitHub notables:**
- **OpenDriverMonitoring** / varios repos tipo "driver-drowsiness-detection" — mayoría son tutoriales con dlib o MediaPipe + EAR, mismo stack del MVP.
- **OpenDBM** (Open Brain-Behavior Digital Biomarkers, NYU) — no es DMS pero tiene pipelines de biomarcadores faciales reutilizables.
- **openpilot (comma.ai)** — incluye un DMS propio (driver monitoring en el Comma 3X), RGB + IR. Open source, MIT-style. **Referencia obligatoria para el proyecto**: hacen RGB-based DMS embebido, con DNN propia, y el código está publicado.

---

## Apps móviles (B2C)

Segmento pobre, mayoritariamente abandonado o de baja calidad. Útil como baseline de "lo que un usuario final puede instalar hoy".

- **Anti Sleep Pilot** (DK) — histórica, combinaba test de reacción + timer, no visión por computadora. Descontinuada / intermitente.
- **Driver Fatigue Monitor / "Stay Awake" / "Drive Awake"** (varias apps Android) — usan cámara frontal, detección de ojos cerrados con OpenCV o ML Kit, alarma sonora. Ratings medios (3-4★), muchas quejas de falsos positivos, consumo de batería, falla con lentes. Ninguna tiene tracción real (<1M descargas la mayoría).
- **Aviva Drive, State Farm Drive Safe & Save, Progressive Snapshot** — telematics de aseguradoras, miden conducta vía acelerómetro/GPS, **no visión**. Relevante como modelo de negocio (descuento en prima).
- **Roadr, Drivvo, etc.** — no son DMS.

**Limitaciones sistemáticas del segmento móvil B2C:**
1. Celular montado mal (no apunta a la cara) → FaceMesh pierde tracking.
2. Batería y térmica: 30 min de procesamiento continuo tumba un gama media.
3. Sin NIR de noche → falla en condiciones reales de manejo nocturno que es cuando más se necesita.
4. Usuario tiene que acordarse de abrir la app. Fricción alta.
5. Falsos positivos → el usuario desinstala a los 2-3 días.

**Conclusión del segmento móvil**: hay espacio, pero el producto tiene que resolver los 5 puntos de arriba para no caer en el mismo pozo.

---

## Gap analysis — dónde hay espacio para innovar

1. **Detección de ebriedad / impairment por visión pura**. Seeing Machines lo anunció en roadmap pero no tiene producto comercial maduro; Smart Eye cubre "cognitive load" pero no alcoholemia. Señales viables: nistagmo horizontal, tiempo de parpadeo alargado, micro-expresiones, variabilidad de pupila, irregularidad en saccades. Un MVP académico que explore esto como *prueba de concepto* (sin prometer precisión clínica) es novedoso y vendible en feria. **Riesgo ético/legal**: alto, hay que encuadrarlo como "alerta a conductor" no "test probatorio".

2. **LATAM / México como mercado**. Ningún after-market tiene producto localizado con precio PyME (<USD 10/vehículo/mes). Para una flotilla de reparto en Querétaro con 15 camionetas, pagar Netradyne no es opción. **Hay hueco para un producto RGB-only, celular+soporte, ~MXN 150-300/mes/vehículo con dashboard web básico**. Esto es defendible académicamente y emprendedurialmente.

3. **RGB-only de muy bajo costo sobre celular en parabrisas**. Nadie en Tier-1 invierte ahí porque no es su negocio; las apps móviles existentes son basura. Si se resuelven los 5 puntos de fricción (montaje, batería, nocturno, falsos positivos, onboarding), **el form-factor es único**.

4. **Coaching en tiempo real con LLM**. Los DMS actuales alertan ("¡Mira al frente!", beep). Ninguno explica *por qué* o da feedback pedagógico post-viaje. Un LLM local (Phi, Gemma 2B) o vía API que genere un resumen del viaje ("tuviste 3 microsueños entre las 14:00 y 14:30, considera parar") es novedoso. Esto conecta con tu pivot original (app de estudio → coaching): **el asset real del proyecto es la pipeline de atención + coaching, no la detección en sí**.

5. **Explicabilidad de la alerta (XAI)**. "Te alerté porque tus ojos estuvieron cerrados 1.8s y giraste la cabeza 45° a la derecha 3 veces en 1 min" es infinitamente más útil que un beep. Fácil de implementar sobre lo que ya tienen (EAR + head-turn ya dan los features para narrarlo). Diferenciador claro vs. apps Play Store.

6. **Integración con aseguradoras LATAM** (Qualitas, GNP, HDI). Telematics ya lo hacen algunas, pero **ninguna con DMS por visión**. Modelo de negocio potencial: descuento en prima a cambio de correr la app. Esto es estrategia, no tecnología, pero da una historia de feria potente.

7. **Detección de uso de celular al volante — específicamente para motos / repartidores**. El boom de delivery (Rappi, Uber Eats, DiDi Food) es enorme en México y los motociclistas son población de alto riesgo. Un DMS para motos (cámara en casco o en dash del scooter) **no existe comercialmente**. Gap grande.

8. **Privacidad on-device total**. Todos los flotilleros suben video a la nube. Diferenciador honesto: "tu cara nunca sale del celular, solo eventos". Relevante regulatoriamente (LFPDPPP en México, GDPR si se exporta).

---

## Recomendación para el proyecto

Para la feria, posicionar el MVP **no como "competidor de Seeing Machines" sino como "DMS accesible para el 95% de los autos que circulan en México y nunca tendrán un sistema OEM"**. El pitch ganador combina tres gaps del análisis: (a) RGB-only sobre celular montado en parabrisas — form factor único, costo cero de hardware; (b) explicabilidad y coaching con LLM post-viaje — aprovecha el pivot desde la app de estudio, donde ya hay experiencia en retroalimentación pedagógica; (c) una *feature de laboratorio* de detección de impairment (fatiga severa / sospecha de ebriedad) como prueba de concepto académica, encuadrada cuidadosamente como alerta, no diagnóstico. Post-feria, el camino realista es convertirlo en un piloto con una flotilla PyME local (reparto, taxis, transporte escolar en Querétaro) a MXN 200-300/vehículo/mes, compitiendo por debajo del piso de Netradyne/Samsara. El MVP técnico (EAR + head-turn + FaceMesh) es suficiente para ese piloto; el valor del proyecto está en el producto, la localización y la narrativa de coaching, no en superar al estado del arte en landmarks.

---

## Para profundizar

**Reportes financieros y de producto (datos duros):**
- Seeing Machines — Annual Report FY2024 y Investor Presentations trimestrales: https://seeingmachines.com/investors/ (buscar "FY24 results", "Automotive pipeline units")
- Smart Eye — Annual Report 2024 y Q4 report: https://smarteye.se/investors/ (buscar "design wins", "AIS pipeline")
- Cipia — Investor relations (TASE:CPIA): https://cipia.com/investors/
- Samsara — 10-K FY2025 (NYSE:IOT): buscar "AI Dash Cam ARR"
- Netradyne — reportes de prensa y funding rounds (privado, Series D-E 2023-2024)

**Regulación:**
- UNECE R159 / UN R152 (AEB) y R157 (ALKS) — contexto ADAS
- EU General Safety Regulation 2019/2144 + Implementing Reg. 2021/1341 (DDAW) + 2023/2590 (ADDW)
- Euro NCAP Roadmap 2025-2030 — "Occupant Status Monitoring" protocol

**Papers clave (búsquedas sugeridas en Google Scholar / arXiv):**
- `"driver drowsiness" "MediaPipe" 2024..2026`
- `"driver monitoring" transformer 2024..2026`
- `"alcohol impairment" "facial landmarks" driver`
- `"gaze estimation" in-cabin automotive 2024..2026`
- `"distracted driving" dataset benchmark 2024`
- `explainable AI driver monitoring`
- Buscar autores: Juan Diego Ortega (Vicomtech, DMD dataset), Cristina Bustos, Miguel Bordallo (eye-tracking + drowsiness).

**Queries de búsqueda (web/news) para mantener actualizado el landscape:**
- `"driver monitoring system" OEM adoption 2025`
- `Seeing Machines design win 2025`
- `Smart Eye AIS pipeline 2025`
- `Netradyne Mexico fleet`
- `Samsara AI dash cam LATAM`
- `Euro NCAP 2026 driver monitoring protocol`
- `alcohol detection camera vehicle DADSS` (sistema DADSS de NHTSA — alcoholímetro infrarrojo en auto, contexto regulatorio para ebriedad)
- `in-cabin sensing market forecast 2030`

**Competidores/proyectos a monitorear activamente:**
- openpilot (comma.ai) — repo `commaai/openpilot`, carpeta `selfdrive/monitoring/`
- OpenDBM — biomarcadores faciales reutilizables
- DMD dataset (Vicomtech) — para benchmark propio
- DADSS Program (NHTSA + ACTS) — alcohol sensing en auto, competencia indirecta para el ángulo de ebriedad

**Analistas y reportes de mercado (pago, pero abstracts gratis):**
- ABI Research — "In-Cabin Monitoring and Sensing" (anual)
- Yole Group — "Automotive Camera Market" y "Driver Monitoring System"
- Strategy Analytics / TechInsights — design-win trackers OEM
