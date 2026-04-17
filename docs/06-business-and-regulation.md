# Viabilidad comercial, mercado y regulación

> Documento de estrategia para el pivot del proyecto "Attention-Algorithm" hacia un Driver Monitoring System (DMS).
> Pregunta central: **¿es vendible a una armadora? ¿a quién más? ¿qué regulación aplica?**
> Alcance: análisis realista orientado a un proyecto universitario, no a un deck de inversionistas.

---

## TL;DR

- **Vendérselo directo a una armadora (OEM) en 2026 siendo estudiantes es prácticamente imposible.** Los ciclos de homologación son de 3-5 años, requieren ASPICE L2-L3, ISO 26262 ASIL-B, ISO 21448 (SOTIF), y una trayectoria de validación con millones de km. Seeing Machines y Smart Eye (los dos líderes globales) tardaron ~15-20 años en llegar ahí.
- **Pero sí hay mercado real en segmentos accesibles:** flotillas de transporte (after-market), aseguradoras con productos UBI, gobierno/transporte público en LATAM, y partnerships con Tier 2/3 locales. Los ciclos de venta ahí son de 3-9 meses, no 5 años.
- **La regulación es el driver más potente del mercado.** La EU GSR 2019/2144 vuelve obligatorio DAW (drowsiness) desde 2024 en todos los autos nuevos y ADDW (distracción) desde julio 2026. Euro NCAP ya exige DMS para 5 estrellas desde enero 2023. China C-NCAP lo incorporó en julio 2024. EE.UU. viene atrás pero el HALT Act presiona. México/LATAM van atrasados, lo cual es oportunidad.
- **El mercado global de DMS vale ~USD 2.0-3.5 B en 2025-2026 y se proyecta a USD 9.6-10.8 B para 2035, CAGR ~10-12%.** Está dominado por Seeing Machines, Smart Eye, Valeo, Bosch, Aptiv, Continental. Entrar como competidor directo es suicida; entrar como complemento regional o de nicho, viable.
- **Ángulo diferenciador defendible para un equipo universitario en México:** precio accesible para flotillas LATAM, integración con coach LLM en tiempo real, o focalización en detección de ebriedad (si NHTSA finalmente publica la regla del HALT Act). Sin diferenciador claro, el proyecto muere frente a los incumbentes.
- **Estrategia pragmática propuesta:** Fase 1 — demo + paper + GitHub para la feria. Fase 2 — piloto real con 1 flotilla de carga local, datos con consentimiento, auditoría de fairness. Fase 3 — spin-off universitario con fondos CONAHCYT y/o partnership con aseguradora mexicana (Qualitas, GNP) para UBI.

---

## Regulación — el driver más fuerte del mercado

Cualquier análisis de viabilidad comercial de DMS tiene que empezar por la regulación. Desde 2022-2024 el mercado dejó de ser "nice-to-have" y pasó a ser obligatorio en varias geografías. Esto crea demanda estructural pero también la barrera de entrada más dura: homologación.

### Unión Europea — el estándar que arrastra al resto del mundo

#### UNECE Regulation No. 152 y 159

- **UN R152** — Advanced Emergency Braking Systems (AEBS) para M1/N1. Obligatorio en UE para tipos nuevos desde 2022.
- **UN R159** — Moving Off Information System (MOIS). Complemento de AEBS que reduce atropellos de peatones en arranque.
- No son DMS directamente, pero son el marco de referencia para sistemas activos de seguridad en UE y conviven con los requisitos GSR.

#### EU General Safety Regulation (GSR) 2019/2144 — la pieza clave

La Regulación (UE) 2019/2144 revisó el marco de seguridad vehicular y obliga, entre otros, a sistemas de monitoreo del conductor:

- **Driver Drowsiness and Attention Warning (DAW / DDAW):**
  - Obligatorio para **nuevos tipos** de vehículos M y N desde **6 de julio de 2022**.
  - Obligatorio para **todos los vehículos nuevos** (matriculados) desde **7 de julio de 2024**.
  - DAW es relativamente simple: detección indirecta de fatiga (patrones de steering, hora del día, duración de viaje). Se cumple incluso sin cámara.

- **Advanced Driver Distraction Warning (ADDW):**
  - Obligatorio para **nuevos tipos** desde **7 de julio de 2024**.
  - Obligatorio para **todos los vehículos nuevos** desde **7 de julio de 2026**.
  - ADDW sí requiere sensado directo: cámara infrarroja mirando al conductor, tracking de gaze y head pose.
  - Requisitos técnicos del Reglamento Delegado: a velocidades 20-50 km/h, warning si la mirada permanece en la "zona de distracción" >6 s; a velocidades >50 km/h, >3.5 s.

Esto es lo que ha detonado la demanda masiva de DMS en Europa. Todo OEM que venda en UE necesita esto en cada modelo nuevo a partir de 2026.

#### Euro NCAP — el segundo driver (voluntario pero comercialmente mandatorio)

- Desde **enero 2023**, Euro NCAP incluye DMS dentro del protocolo "Safety Assist — Safe Driving".
- Evalúa tres estados: **distracción** (larga, múltiples cortas, uso de celular), **fatiga** y **conductor no responsivo**.
- En la práctica, **obtener 5 estrellas sin DMS directo con cámara es casi imposible desde 2023**, y el protocolo 2026 sube la vara aún más (requisitos más estrictos sobre sensing, state detection y vehicle response).
- Euro NCAP es "voluntario", pero comercialmente las armadoras compiten por la calificación; es un driver de compra tan fuerte como la regulación.

### Estados Unidos — atrás, pero con presión

- **NHTSA** todavía no tiene un FMVSS específico para DMS. Hay propuestas de NCAP para incluir driver distraction/fatigue como recommended tech.
- **IIHS** publica ratings de ADAS y desde 2024 incluye evaluación de sistemas de monitoreo como parte de su protocolo de Partial Automation Safeguards.
- **HALT Act (parte de la Bipartisan Infrastructure Law, 2021)** — ordena a NHTSA emitir un rulemaking obligando tecnología pasiva de prevención de conducción con discapacidad alcohólica en vehículos nuevos. Plazo original: noviembre 2024; implementación posible desde 2026.
- En la práctica, NHTSA reportó al Congreso en febrero 2026 que la tecnología de detección pasiva de alcohol alrededor del límite legal todavía tiene error rate "inaceptablemente alto". El rulemaking está retrasado.
- De cualquier modo, el HALT Act es una oportunidad: si hay que medir estado del conductor para inferir impairment, DMS visual (fatiga, patrón de gaze, micro-movimientos) es una pieza natural.

### China — el más agresivo después de UE

- **MIIT** publicó GB/T 41796-2022 (durabilidad de módulos de sensing) y GB/T 41797-2022 (requisitos de performance y métodos de test para sistemas de monitoreo de atención del conductor). Son estándares "recomendados" pero en la práctica C-NCAP los exige.
- **C-NCAP 2024** (vigente desde julio 2024) incorpora DMS formalmente:
  - Driver Fitness Monitoring (DFM): fatiga fisiológica (ojos cerrados prolongados, bostezos).
  - Driver Attention Monitoring (DAM): distracción (head pose anormal, celular, fumar).
  - DMS aporta 2 puntos del scoring, solo detrás de AEB (3 puntos).
  - Umbral de detección requerido: ≥90% de accuracy.

### México y LATAM — atrasados, y por eso hay ventana

- **NOM-194-SE-2021** (publicada en octubre 2022, reemplazó NOM-194-SCFI-2015). Regula partes y componentes de seguridad para vehículos ligeros en México. Permite certificar contra FMVSS (US), UNECE (UE) o NMX.
- **NOM-194 NO exige DMS**. Cubre temas clásicos: frenos, iluminación, cinturones, airbags. Esto significa que un auto vendido en México en 2026 puede legalmente no tener DMS.
- **Latin NCAP** actualizó su protocolo en 2024 y planea incluir ADAS y tecnologías de asistencia a partir de 2026. La inclusión de DMS todavía no es tan exigente como Euro NCAP.
- **Implicaciones para el proyecto:** el mercado local mexicano no está forzado a comprar DMS, pero las armadoras que exportan a UE ya tienen que integrarlo. Más interesante: **hay un vacío en flotillas y transporte público** donde nadie los obliga, pero el ROI (reducción de siniestros, primas de seguro) justifica la compra. Ese es el hueco donde un proyecto universitario puede entrar.

### Resumen regulatorio (tabla)

| Región | Norma / Programa | DMS exigido | Fecha clave | Rigor |
|---|---|---|---|---|
| UE | GSR 2019/2144 — DAW | Sí (indirecto) | Jul 2024 (todos los vehículos nuevos) | Obligatorio |
| UE | GSR 2019/2144 — ADDW | Sí (cámara) | Jul 2024 (tipos) / **Jul 2026 (todos)** | Obligatorio |
| UE | Euro NCAP Safe Driving | Sí (de facto para 5★) | Ene 2023, actualización 2026 | Comercialmente mandatorio |
| China | GB/T 41796/41797-2022 | Recomendado | 2022-2023 | Referencia técnica |
| China | C-NCAP 2024 | Sí (scoring 2 pts) | Jul 2024 | Obligatorio de facto |
| US | NHTSA FMVSS | No específico | — | Pendiente |
| US | HALT Act | Detección de impairment | Rulemaking atrasado, ~2026-2028 | En desarrollo |
| US | IIHS Partial Automation Safeguards | Sí (para rating top) | 2024 | Comercialmente relevante |
| México | NOM-194-SE-2021 | No | 2022 | No aplica |
| LATAM | Latin NCAP | Parcial | 2024-2026 | Emergente |

**Conclusión regulatoria:** vender en UE o China en 2026 **obliga** a DMS. Vender en México no. Esto define dónde puede competir un proyecto universitario y dónde no.

---

## Tamaño de mercado y crecimiento

Las cifras varían significativamente por consultora (metodología, alcance, inclusión de OMS, in-cabin sensing, etc.). Rangos representativos:

| Consultora | Mercado actual | Proyección | CAGR | Año |
|---|---|---|---|---|
| SNS Insider | USD 3.51 B (2025) | USD 9.60 B (2035) | 10.74% | 2026-2035 |
| Spherical Insights | USD 3.52 B (2025) | USD 10.84 B (2035) | 11.9% | 2026-2035 |
| Grand View Research / GMInsights | USD 2.2 B (2023) | USD 5.6 B (2030) | ~12.5% | 2025-2030 |
| ResearchAndMarkets | USD 2.07 B (2026) | USD 2.94 B (2030) | 9.2% | 2026-2030 |

Rango razonable a citar: **CAGR 9-12%, mercado de ~USD 3-3.5 B en 2025 creciendo a ~USD 9-11 B hacia 2035**. La variación viene de si se incluye solo el hardware/software del módulo DMS o también el sensing de ocupantes (OMS), gesture recognition, etc.

**Drivers del crecimiento** (consistentes entre reportes):
1. GSR UE (ADDW 2026) + Euro NCAP.
2. C-NCAP 2024.
3. Vehículos autónomos L2+/L3 que requieren handover monitoring.
4. Aseguradoras UBI (usage-based insurance) que valoran data de comportamiento.
5. Flotillas comerciales buscando reducir siniestralidad.

**Fuentes para citar en el paper/deck:**
- SNS Insider — "Driver Monitoring System (DMS) Market", abril 2026.
- Spherical Insights — "Top 20 Companies in Global DMS Market 2026-2035".
- Grand View Research — "Driver Monitoring System Market Size Report".
- Yole Développement / Yole Group — reportes de in-cabin sensing (~USD 4k cada uno, en biblioteca de la universidad a veces accesibles).
- MarketsandMarkets — "Driver Monitoring System Market — Global Forecast".

> **Nota honesta:** las consultoras venden reportes de USD 4-8k cada uno. Las cifras en notas de prensa son teasers. Para el paper universitario basta citar los rangos y la fuente, no hay que comprar el reporte.

---

## Canales de venta

Para un equipo universitario en México en 2026, no todos los canales son realistas. Tabla comparativa:

| Canal | Ticket típico | Ciclo de venta | Requisitos | Realismo para proyecto universitario |
|---|---|---|---|---|
| OEM (Tier 1 supplier) | USD 5-20 / vehículo licencia | 3-5 años | ASPICE L2-3, ISO 26262 ASIL-B, ISO 21448 SOTIF, MISRA C, historial | Muy bajo. No empezar aquí. |
| After-market fleet | USD 30-70 / vehículo / mes | 3-9 meses | Hardware + SaaS + soporte | Medio-alto. Entry point ideal. |
| Insurance / UBI | Rev share o licencia | 6-18 meses | Integración telemática, validación actuarial | Medio. Requiere partner. |
| Apps B2C | USD 0-10/mes freemium | Inmediato | UX pulida, viralidad | Bajo ROI, alto como showcase. |
| Gobierno / transporte público | Licitación (USD 100k-5M) | 12-24 meses | Cumplimiento de bases, capacidad de entrega, fianza | Medio. Con socio local. |

### OEM (Tier 1 supplier)

- El camino de Seeing Machines y Smart Eye: integrarse como proveedor de software al Tier 1 (Magna, Valeo, Aptiv, Continental, Bosch) que a su vez vende al OEM (Ford, GM, VW, Toyota, Stellantis).
- Requiere cumplir **ASPICE nivel 2-3** (Automotive SPICE, proceso de desarrollo auditable), **ISO 26262** con ASIL-B como mínimo para sistemas de warning (B para DAW/ADDW), **ISO 21448 SOTIF** (Safety Of The Intended Functionality, crítico para sistemas ML), MISRA C/C++, ciberseguridad ISO/SAE 21434.
- Proceso típico: RFQ → A-sample → B-sample → C-sample → SOP (Start Of Production). Entre 3 y 5 años. Millones de USD en gastos de desarrollo.
- Costos de homologación: USD 1-5M por programa.
- **No es para un proyecto universitario sin socio industrial.** Lo que sí se puede: publicar un paper y/o IP que eventualmente sea comprada/licenciada por un Tier 1.

### After-market fleet

- El canal más accesible y el que tiene mayor tracción comercial visible en LATAM.
- Jugadores globales: **Netradyne** (Driveri), **Samsara** (AI Dash Cam), **Lytx** (DriveCam), **Motive** (ex KeepTruckin), **Nauto**, **Azuga**, **Geotab**.
- En México: **LoJack / Brickhouse**, **Copiloto Satelital**, **Detektor**, **Track Plus**, integradores locales de Samsara/Motive.
- Pricing típico: **USD 30-60 por vehículo por mes** para Samsara/Lytx/Netradyne (Premium), USD 25-35 para Motive (mid-range), USD 15-25 para integradores locales genéricos.
- Ciclo de venta: 3-9 meses. Demo → piloto 10-50 unidades → rollout.
- **Ventaja para el proyecto:** se puede entrar con hardware barato (Raspberry Pi / Jetson Nano + cámara IR) y software propio. El cliente no pide ISO 26262, pide que funcione y que el ROI sea claro (menos siniestros, menos primas).

### Insurance / UBI (Usage-Based Insurance)

- Las aseguradoras pagan por "señal" que permita tarificar mejor o reducir siniestros.
- Jugadores globales: **Cambridge Mobile Telematics** (white label para muchas aseguradoras), **Root Insurance**, **Progressive Snapshot**, **Allstate Drivewise**, **Metromile**.
- En México: **Qualitas** tiene programas de telemática para flotas; **GNP**, **AXA**, **Mapfre** están explorando UBI pero aún incipiente.
- Modelo: la aseguradora paga rev share sobre primas o un fee por conductor monitoreado. Requiere integración con su stack actuarial.
- Buena opción para partnership, pero requiere acceso a una aseguradora dispuesta. Un profesor con contacto en Qualitas/GNP puede abrir esa puerta.

### Apps B2C

- Difícil monetizar. El conductor promedio no paga por una app que le grite "te estás distrayendo".
- Sirve como **showcase** — subirla a Play Store/App Store da credibilidad, data y tráfico.
- Modelos que a veces funcionan: integración con gamificación para jóvenes (descuentos en seguro si mantienes score), coach de viajes largos, integración con Waze/Google Maps.
- Competidores: **Nexar**, **CarSafe**, **Sober Steering**, apps de ADAS para celulares.

### Gobierno / transporte público

- Licitaciones de SCT/SICT, gobiernos estatales, sistemas de transporte público (Metrobús CDMX, transporte urbano Monterrey/Guadalajara, RTP).
- Tickets potencialmente grandes (cientos de miles a millones de USD) pero ciclos largos (12-24 meses) y requieren socio local con experiencia en licitaciones.
- Oportunidad realista: SEMOVI CDMX, ATT Nuevo León, empresas de transporte de carga con contratos federales (Pemex, CFE).

---

## Pricing benchmarks

Precios reales observados en el mercado para calibrar expectativas.

### OEM (licencia por vehículo)

- **Seeing Machines (FY2025):** revenue ~USD 62-63M, 4.2M vehículos equipados (62% YoY growth). ARR (Annualised Recurring Revenue) ~USD 13.4M. Implica **~USD 3-5 por vehículo** en royalty OEM, más servicios de ingeniería.
- **Smart Eye:** revenue similar (SEK ~300M / ~USD 28M), creciendo 20-30% YoY. Mix parecido de royalty + NRE (Non-Recurring Engineering).
- **Orden de magnitud:** **USD 3-10 de licencia por vehículo** al OEM, con NRE de USD 1-5M por programa.

### After-market fleet (suscripción)

- **Samsara / Lytx / Netradyne:** **USD 30-60 / vehículo / mes**. Incluye hardware, SaaS, IA, soporte.
- **Motive:** USD 25-35 / vehículo / mes.
- **Integradores LATAM (revendedores):** USD 15-35 / vehículo / mes para productos más simples.
- Hardware upfront: USD 300-900 por unidad, a veces subsidiado en contratos de 3 años.

### Consumer dashcams con DMS básico

- **Nexar Pro / Beam:** USD 150-250.
- **Owl Car Cam (descontinuada pero benchmark):** USD 300-400.
- **Vantrue N5 / E3:** USD 200-400.
- **Garmin Dash Cam con driver alerts:** USD 150-300.
- Mercado saturado, margen bajo.

### Implicación para el proyecto

- Si el target es OEM: USD 3-10 / vehículo. Para tener revenue interesante (USD 1M/año) necesitas **100k-300k vehículos equipados**. Inaccesible sin Tier 1 socio.
- Si el target es flotilla LATAM: USD 20 / vehículo / mes → **una flotilla de 500 unidades = USD 120k/año ARR**. Alcanzable con 2-3 flotillas piloto. Este es el tamaño real al que un proyecto universitario puede aspirar en 2-3 años.

---

## Competidores directos por segmento

Mapa simplificado de "con quién se pelea el mercado" por canal.

| Segmento | Competidores globales | Competidores regionales (LATAM/MX) | Foso principal |
|---|---|---|---|
| OEM (algoritmo) | Seeing Machines, Smart Eye, Jungo, Cipia, Tobii, Mitsubishi Electric | — (no hay) | IP, validación, ASPICE/ISO 26262, contratos plurianuales |
| OEM (hardware/Tier 1) | Valeo, Bosch, Continental, Aptiv, Magna, Denso, Visteon | — (integradores) | Relación con armadora, capacidad de producción |
| After-market fleet | Samsara, Netradyne, Lytx, Motive, Nauto, Azuga, Geotab | Copiloto Satelital, LoJack, Detektor, Track Plus | Red comercial, integración ERP/telemática |
| UBI / aseguradoras | Cambridge Mobile Telematics, Arity (Allstate), Zendrive, TrueMotion | Pocos. Qualitas in-house parcial | Partnership con aseguradora, modelo actuarial |
| Apps B2C | Nexar, DriveWell, Flo | Pocos | Distribución, UX, data |
| Consumer dashcam | Nexar, Garmin, Vantrue, BlackVue, Thinkware | Steren, marcas chinas genéricas | Canal retail, marca |

**Lectura rápida:** los OEM/Tier 1 son un bloque cerrado e inaccesible. El after-market fleet en LATAM no está saturado por locales — los players son globales con revendedores, lo que deja hueco para un producto local competitivo en precio y soporte.

---

## Barreras para un proyecto universitario

Ser honestos sobre lo que no se puede hacer como equipo de estudiantes en 1 año.

1. **Certificaciones automotrices (OEM path):**
   - **ASPICE L2-3:** requiere un proceso de desarrollo auditado por años. Costo: USD 200k-1M solo en consultoría y auditoría inicial.
   - **ISO 26262 ASIL-B/C:** requiere FMEA, FTA, hardware-software co-design, tooling certificado. Tiempo típico 18-36 meses.
   - **ISO 21448 SOTIF:** especialmente dura para ML, requiere ODD (Operational Design Domain) bien definido y cobertura estadística.
   - **Fuera de alcance.**

2. **Datos de validación:**
   - Seeing Machines y Smart Eye han recolectado **decenas de millones de km** de datos reales con conductores consintientes, múltiples demografías, iluminación, razas, lentes, etc.
   - Validar fairness (gender, skin tone, age, glasses, ethnicity) es condición sine qua non en 2026 para cualquier OEM o regulador serio.
   - Los datasets públicos (DriveAHEAD, DMD, DAD, Drive&Act) son útiles para research pero no suficientes para claims de accuracy en producción.
   - **Sí es posible** recolectar dataset propio modesto (cientos de horas) con una flotilla piloto. No es suficiente para OEM, sí para after-market.

3. **Relación con OEMs:**
   - Requiere track record, estabilidad financiera, fábrica con capacidad, equipos de 50+ ingenieros dedicados por programa.
   - **Camino realista:** research → paper → IP → licenciamiento a Tier 1 existente. Nunca venta directa a OEM.

4. **Ciberseguridad automotriz (ISO/SAE 21434):**
   - Obligatorio en UE desde 2024 para nuevos tipos. Requiere SOC, SBOM, vulnerability management, penetration testing.
   - Fuera de alcance para la Fase 1, manejable para Fase 2-3 con socio.

5. **Fondeo:**
   - Seeing Machines levantó ~USD 150M+ en ~15 años antes de ser rentable. Smart Eye similar.
   - Un spin-off universitario puede acceder a CONAHCYT (programa PRONACES, Fondos Sectoriales), 500 Startups LATAM, Endeavor, Mountain Nazca, Angel Ventures. Tickets semilla USD 50k-500k.
   - Suficiente para un piloto fleet, insuficiente para un play OEM global.

---

## Estrategia pragmática — 3 fases

### Fase 1 (ahora — feria académica, horizonte 1-3 meses)

Objetivo: demostrar que el equipo sabe hacerlo y construir credibilidad.

- Demo en vivo sólido: cámara + modelo corriendo en Jetson Nano/RPi o laptop, detectando fatiga, distracción, y celular en mano.
- Benchmark contra datasets públicos (DMD, DAD) con métricas defendibles (precision/recall, no solo accuracy).
- GitHub público con código reproducible, dataset synthetic o licenciado, README decente.
- Paper corto (4-8 páginas, estilo IEEE) subido a arXiv. No tiene que ser publishable, pero sí defendible.
- Video corto (2-3 min) mostrando el sistema funcionando en escenarios reales (distracción con celular, ojos cerrados, cabeza girada).
- Nada de ASPICE/ISO. No aplica en feria.

**Entregable clave:** demo + paper + repo. Esto ya vende la tesis sin prometer lo que no se puede cumplir.

### Fase 2 (6-12 meses post-feria)

Objetivo: cerrar 1 piloto real con cliente pagando algo (aunque sea simbólico) y recolectar datos.

- Identificar 1-2 empresas de flotilla local (transporte de carga Querétaro/Bajío, transporte de personal industrial, última milla urbana). Tamaño típico: 50-300 unidades.
- Ofrecer piloto 10-30 unidades gratis o a costo durante 3-6 meses, con clausula clara de recolección de datos con consentimiento informado del conductor.
- Instrumentación: cámara IR + Jetson Nano/Orin + LTE. BOM USD 200-400 por unidad.
- Backend: dashboard web básico (Next.js + Supabase/Postgres), eventos clasificados (drowsiness, distraction, phone), reporte semanal al fleet manager.
- Validación de fairness: asegurar que el modelo no degrada performance por tono de piel, género, lentes, edad. Auditar y publicar resultados.
- Paper 2 (publishable): resultados del piloto. CIARP, CVPR Workshop, ITSC son targets razonables.
- Considerar partnership con aseguradora (Qualitas/GNP) para explorar UBI: si el piloto muestra reducción de siniestros verificable, hay caso para rev share.

**Entregable clave:** contrato/MoU con flotilla + dataset propio + paper con resultados reales. Esto ya es "startup que funciona", no "proyecto escolar".

### Fase 3 (1-3 años)

Objetivo: convertirlo en entidad real (startup, spin-off del Tec, centro de investigación aplicada).

- Spin-off universitario con apoyo del Tec (OTT, Incmty, Contigo Seguros si aplica).
- Fondos: CONAHCYT (PRONACES, ProInnova), aceleradoras (Mountain Nazca, Angel Ventures, 500 LATAM), potencial strategic investment de aseguradora mexicana.
- Producto comercial v1 para flotillas LATAM: hardware integrado, SaaS, pricing USD 20-35 / vehículo / mes. Target 500-2,000 unidades activas en año 2.
- Paralelamente, explorar **licenciamiento de IP a Tier 1 regional** (Magna México, Valeo Ramos Arizpe, Continental San Luis Potosí) si el algoritmo demuestra ventaja competitiva específica (por ejemplo, mejor performance en conductores latinos, o costo menor).
- Opcional y ambicioso: partnership con aseguradora para lanzar producto UBI co-branded ("Qualitas + nombre_del_proyecto — Seguro para flotillas con DMS incluido"). Tamaño de mercado LATAM suficiente para justificar.
- Certificaciones incrementales: ISO 9001 → ISO/IEC 27001 (ciberseguridad) → ISO 26262 (solo si hay camino OEM claro). **No perseguir ISO 26262 sin cliente que la requiera y pague por ella.**

---

## Contra-argumento honesto — ¿vale la pena?

Antes de comprometerse con el pivot a DMS, el equipo debería responder 3 preguntas incómodas:

1. **¿El ángulo diferenciador es defensible?**
   - Si el pitch es "hacemos lo que Seeing Machines pero más barato": van a perder. Seeing Machines tiene 20 años de IP y USD 150M de R&D.
   - Ángulos potencialmente defensibles:
     - **LATAM-focus:** producto diseñado para el mercado mexicano/latinoamericano — precio 30-50% menor, soporte en español, integración con ERPs locales (SIIT, plataformas de flotilla mexicanas), adaptado a vehículos usados (lo que nadie en el top tier quiere tocar).
     - **LLM coach:** un agente conversacional en cabina que no solo alerta sino que "coachea" al conductor en tiempo real ("oye, llevas 5 horas manejando, hay un OXXO a 3 km, te recomiendo parar"). Esto no lo hacen los players actuales y es posible hoy con Whisper + LLM barato + TTS.
     - **Detección de ebriedad:** si NHTSA finalmente mueve el HALT Act, quien tenga un modelo decente de impairment visual (micro-movimientos oculares, gaze erratic, reacción lenta a estímulos) tiene una ventana. Requiere research serio, no implementación rápida.
     - **Integración con cámara de celular:** un SDK que convierte el celular del conductor en un DMS después-mercado de bajo costo. Target: conductores de Uber/DiDi, flotillas pequeñas sin hardware dedicado.
   - Si ninguno de estos convence al equipo con honestidad, **hay que revisar el pivot**.

2. **¿Hay alguien del equipo con contacto real en una flotilla, aseguradora, o Tier 1 local?**
   - Sin un "warm intro" en Fase 2, cerrar un piloto toma 6-12 meses extra de trabajo comercial frío que no le corresponde a un equipo de ingeniería.
   - Es más fácil revisar esto al inicio que descubrirlo en el mes 8.

3. **¿El equipo se compromete a 2-3 años o es un proyecto de un semestre?**
   - Un DMS que solo vive en feria es un portfolio piece, no un negocio. Perfectamente válido, pero hay que etiquetarlo como tal.
   - Si el objetivo real es aprender ML aplicado y tener algo lindo en el CV: fase 1 basta y el documento lo refleja.
   - Si el objetivo es construir una startup: hay que encuadrar fase 2 desde ya (qué flotilla, qué profesor conecta, quién hace el business dev).

**Respuesta corta a "¿es vendible a una armadora?":** no, no en 2026, no por este equipo, no directo. **¿A quién sí?** Flotillas de transporte LATAM, aseguradoras con producto UBI emergente, integradores de telemática locales, gobierno subnacional vía licitación con socio. Ahí sí hay mercado real.

---

## Para profundizar

### Regulación (textos oficiales)

- **EU GSR 2019/2144** — texto consolidado: https://eur-lex.europa.eu/eli/reg/2019/2144/oj
- **Reglamento Delegado ADDW** — C(2023)4523: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=intcom:C(2023)4523
- **Euro NCAP protocols** (Safe Driving 2023+ y roadmap 2026): https://www.euroncap.com/protocols/
- **UNECE R152, R159:** https://unece.org/transport/vehicle-regulations
- **NHTSA HALT rulemaking / ANPRM (enero 2024):** https://www.federalregister.gov/documents/2024/01/05/2023-27665/advanced-impaired-driving-prevention-technology
- **NOM-194-SE-2021** (DOF, México): buscar en https://www.dof.gob.mx
- **Latin NCAP protocols 2024-2026:** https://www.latinncap.com/en/our-tests
- **China C-NCAP 2024:** https://www.c-ncap.org

### Mercado y competidores

- **Seeing Machines Annual Report 2024 y FY2025 Trading Update:** https://seeingmachines.com/investors/
- **Smart Eye investor relations (SEK reports):** https://smarteye.se/investors/
- **Understanding DMS blog (analista independiente):** https://www.understandingdms.com
- **MarketsandMarkets — Driver Monitoring System Market Report** (teaser gratis, full USD 4-8k).
- **Yole Group — Automotive In-Cabin Monitoring / DMS & OMS report** (USD 6k+, buscar acceso vía biblioteca Tec).
- **Strategy Analytics / TechInsights** — reportes de ADAS market.
- **IDTechEx — "DMS & OMS 2024-2034"** — buen overview regulatorio (teaser gratis).

### Papers y datasets académicos

- **DMD (Driver Monitoring Dataset), Vicomtech:** https://dmd.vicomtech.org
- **DAD (Driver Anomaly Detection):** https://github.com/okankop/Driver-Anomaly-Detection
- **Drive&Act:** https://driveandact.com
- **DriveAHEAD (Karlsruhe):** head pose estimation in driving.
- **AUC Distracted Driver Dataset:** classificación de 10 clases de distracción.
- Reviews recientes: búsqueda en IEEE Xplore "driver monitoring system review 2024-2026".

### Queries útiles para seguir investigando

- `"driver monitoring system" "ISO 26262" site:seeingmachines.com OR site:smarteye.se`
- `"Euro NCAP" "2026" "driver monitoring" protocol`
- `"GSR" "ADDW" "July 2026" implementing regulation`
- `"Netradyne" OR "Samsara" "driver monitoring" pricing fleet`
- `"NOM-194" ADAS Mexico autos`
- `"Qualitas" OR "GNP" telematics UBI flotilla`
- `"CONAHCYT" mobility safety startup grant`
- `"NHTSA" "alcohol impaired driving prevention" "advanced notice of proposed rulemaking"`
- `site:arxiv.org "driver distraction" detection fairness skin tone`

### Fuentes consultadas para este documento

- InterRegs — EU Regulation on Advanced Driver Distraction Warning Systems.
- EUR-Lex — Regulation (EU) 2019/2144 y Delegated Regulation C(2023)4523.
- Seeing Machines FY2025 Trading Update y Annual Report 2024.
- Understanding DMS — 2024 Results for Seeing Machines and Smart Eye.
- SNS Insider, Spherical Insights, Grand View Research, ResearchAndMarkets — reportes de mercado DMS 2025-2035.
- Euro NCAP protocols documentation.
- Anyverse — Euro NCAP, C-NCAP y Global DMS standards.
- Smart Eye blog — Euro NCAP 2026 updates y GSR vs Euro NCAP differences.
- NHTSA — Advanced Impaired Driving Prevention Technology ANPRM (Federal Register).
- MADD — Impaired Driving Prevention Tech ANPRM Fact Sheet.
- ATIC-TS — Mexican Light Vehicle Safety Regulation NOM-194-SE-2021.
- Latin NCAP — protocolo actualizado 2024-2026.
- MIIT — GB/T 41796-2022 y GB/T 41797-2022 (vía ResearchInChina y Anyverse).
- Samsara, Motive, Freightwaves — benchmarks de pricing fleet dashcams.
- IDTechEx — "Regulations: Drivers for Mandating Driver Monitoring Systems".

---

*Documento vivo. Actualizar cifras de mercado y fechas regulatorias cada 6 meses — este sector se mueve rápido.*
