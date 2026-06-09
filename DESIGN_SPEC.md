# Spec de Diseño + Stack — DMS (Driver Monitoring System)
### Documento único consolidado · Español de México · MVP demo Expo Ingenierías

> Este spec resuelve los 3 veredictos de jurado y todo el trabajo de exploración en una sola dirección coherente, sin contradicciones. Donde los jurados chocaron, la resolución se marca explícitamente con **[RESUELTO]** y su razón.

---

## 1) Estilo elegido: **SENTINEL** (chasis) + injertos

**Ganador: SENTINEL — Cinematic Road-Safety HUD**, el "copiloto vigilante" en un *cockpit* oscuro casi-negro donde la información es luz que emerge de la oscuridad y el color está **racionado como una lámpara de advertencia**, no salpicado como app de consumo.

### 1.1 La gran contradicción resuelta: oscuro vs claro **[RESUELTO]**
Los jurados se dividieron: J1→SENTINEL (oscuro), J3→Cobalt (oscuro), J2→Faro (claro). **Gana oscuro** por dos razones que sobreviven al escrutinio:

- **El argumento de Faro a favor del canvas claro se autodestruye con sus propios números.** El mismo análisis (J2/J3) mide que el ámbar (`drowsy`) sobre papel claro da **1.94:1 — ilegible**, y obliga a re-empaquetar todo como *fills* sólidos. En cambio, **texto blanco grande sobre fill de estado saturado sobre fondo casi-negro supera AA-large** en los tres estados. Para una pantalla montada bajo sol, el contraste de luminancia (blanco sobre color profundo) sobrevive al glare mejor que tinta de color sobre papel.
- **Lo que SÍ se rescata de Faro no es el color del canvas, es la disciplina del wow:** el "respiro" debe ser **CSS `@keyframes` puro (transform/opacity), acelerado por GPU, cero JS por frame, con kill-switch `prefers-reduced-motion`**. Esa es la regla anti-jank que adopta SENTINEL. Framer Motion es **opcional**, nunca *load-bearing*.

**Conclusión:** Vista conductor = **siempre oscura** (`#0A0E14`). Dashboard manager = oscuro por defecto, con **modo claro opcional** (es desktop, sin glare; ahí el claro es seguro y se ve "oficina diurna").

### 1.2 Por qué SENTINEL y no los rivales
Cinco direcciones (SENTINEL, Cobalt, Cluster Noir, Nocturne, parte de Cinética) **convergen** en la misma receta: canvas casi-negro, un acento neutro, anillo de estado que "respira", tipografía grotesca, mapa oscuro, Recharts. En esa familia, SENTINEL gana porque:
- Tiene el **wow universal del HUD oscuro** (calma → urgencia inequívoca sin flashear al juez), Y
- Es el único que **pone al frente la "traza de atención"** alimentada por el `get_attention_weights()` real del Bi-LSTM (ventana temporal de 3s) — el diferenciador que un jurado de **ingeniería** premia: convierte un blob de color en una **demo de IA creíble**.
- Es **el más shippable** del clúster oscuro: el anillo es **SVG puro + 2 keyframes CSS, sin librería de animación, sin canvas/WebGL**.

### 1.3 Injertos obligatorios (de los runners-up)
| Injerto | Origen | Qué es |
|---|---|---|
| **Regla "color = SOLO estado"** | Tablero Editorial | Todo el *chrome* (marca, botones, links, focus) va en tinta neutra/cian. Verde/ámbar/rojo aparecen **únicamente** para el estado del conductor. Una pantalla monocroma literalmente significa "todo bien, sigue manejando". Regla **dura**, no guía. |
| **Trip Ribbon** | Tablero Editorial | Una barra horizontal de 8px que es la línea de tiempo completa del viaje, pintada por estado, alineada pixel-a-pixel sobre el track GPS. Hover sincroniza punto en mapa + snapshot. Es el "ohh" del manager. Flex de divs / SVG — librería ligera. |
| **Traza de atención (sparkline)** | SENTINEL + Cluster Noir | Micro-sparkline de los últimos ~30s de estado/confianza dentro del anillo, mapeada al output real del modelo. **Hero central del conductor, NO nice-to-have.** Sin endpoint nuevo (confianza ya viaja en `state_samples`). |
| **Disciplina de contraste/CVD medida** | Faro | Ratios por estado, *fills* separados para texto blanco, regla dura "ámbar nunca es texto/línea fina". Se hornea en los tokens. |
| **Trío de color más separable** | Cinética Editorial (medido por J3) | El trío teal/ámbar/magenta tiene la **mejor separabilidad grayscale** y distancia CVD. Reemplaza al verde/ámbar/rojo ingenuo. Ver §2. |
| **Rojo reservado SOLO para peligro** | Cluster Noir | Errores de sistema/conectividad usan el acento cian o un toast neutro, **nunca rojo**. El rojo del estado queda semánticamente puro. |
| **Háptico + heartbeat "en vivo"** | Cobalt + SENTINEL | `navigator.vibrate()` en transición de estado (money-shot en el volante Forza). Dot "en vivo" atado al último poll exitoso, para distinguir pantalla congelada (WiFi) de un Alert real. |
| **Encuadre "caja negra"** | BLACKBOX (solo el framing) | Snapshots de incidentes en el manager presentados como "frames recuperados" con velocidad + coordenada estampadas. **Solo el encuadre narrativo, SIN scanlines/CRT/mono-mayúsculas.** Solo manager, jamás conductor. |
| **Redundancia: color + icono + palabra + movimiento** | Todas | Cada estado se codifica por 4 canales. Nunca un dot de color solo. **No se corta bajo presión de tiempo.** |

---

## 2) Paleta final (con hex)

### 2.1 Neutros y marca (chrome mudo)
| Rol | Hex | Uso |
|---|---|---|
| `canvas-void` | `#0A0E14` | Fondo base app / pantalla conductor |
| `surface-panel` | `#121821` | Cards, paneles, bloques de telemetría |
| `surface-raised` | `#1B2430` | Hover, superficie secundaria, track del timeline, base del mapa |
| `hairline` | `#2A3543` | Bordes 1px, divisores, grid, detalle de crosshair |
| `ink-high` | `#EAF0F7` | Texto primario, números grandes, label de estado (16.9:1 sobre void) |
| `ink-mid` | `#9DAEC2` | Texto secundario, labels de eje, metadata |
| `ink-low` | `#8696AC` | Terciario, timestamps, footnotes (subido de `#5E6F84`, que reprobaba AA → ahora 6.4:1 void / 5.9:1 panel) |
| `brand-signal` (acento) | `#3DA9FC` | Logo, nav activa, links, focus ring, **línea de ruta**, botón primario, heartbeat "en vivo". **NO es color de estado** → nunca compite con una alerta |
| `brand-signal-deep` | `#1E6FB8` | Pressed/active del acento |
| `manager-light-canvas` (opcional) | `#F5F7FA` | Modo claro **solo del dashboard manager** (desktop, sin glare) |

### 2.2 Colores de estado — colorblind-safe **[RE-DERIVADO con contraste WCAG medido]**

> **Corrección sobre el draft del sintetizador:** los ratios "~5.0:1" eran aspiracionales y varios reprobaban AA (p. ej. blanco sobre teal-fill `#0F7E86` daba 4.21:1, no 5.0), y los *deep fills* que se pintaban tenían **peor** separabilidad CVD que el trío base. Esto se re-calculó con la fórmula WCAG real.

**Decisión clave que elimina el problema de contraste (resuelve A1+A2):** en la **vista conductor** el color del estado se porta en el **anillo + icono + palabra grande sobre el canvas oscuro `#0A0E14`** — **NUNCA como texto blanco sobre un fill saturado**. Así el texto siempre es alto-contraste sobre oscuro y el color es señal gráfica, no fondo de texto.

Trío base (contraste medido sobre `#0A0E14`):

| Estado (ES) | Modelo | Color (anillo/icono/palabra) | Sobre void | Fill de badge (manager) | Texto del badge |
|---|---|---|---|---|---|
| **Alerta** | `Alert` | `#2DD4BF` teal | **10.4:1** | `#0B6A66` | blanco (5.6:1) |
| **Somnoliento** | `Drowsy` | `#FFB020` ámbar | **10.6:1** | `#FFB020` | navy `#06090F` (10.9:1) |
| **Distraído** | `Distracted` | `#FF6B81` coral | **7.1:1** | `#B83250` | blanco (5.1:1) |
| **Sin señal / Buscando** | — | `#94A3B8` slate | 7.5:1 | — | — |

**Incidente / `harsh_event`:** hereda el color del estado del momento (incidente somnoliento = ámbar, distraído = coral) + un *glow* efímero del mismo tono. **No existe un color de "peligro" aparte** (resuelve A3).

**Por qué es CVD-safe:** teal↔(ámbar/coral) se separan por el **eje azul-amarillo** (preservado en deuteranopia/protanopia, ~8% de hombres); ámbar↔coral se separan por **luminancia** (gap medido 0.19). El caso grayscale puro (acromatopsia, rarísimo) lo cubre la **redundancia obligatoria de icono + palabra** (§2.4). Se usa **coral, no rojo puro**, para no leer como alarma de peligro y para evitar el colapso del magenta en protanopia.

### 2.3 Reglas duras de color (no negociables)
1. **Los colores de estado solo se pintan sobre oscuro como anillo/icono/palabra grande** (todos ≥7:1 sobre void). En el manager, si se necesita *badge con fill*, se usan los fills medidos de §2.2 con su texto correspondiente (ámbar→navy, teal/coral→blanco). El ámbar **nunca** lleva texto blanco encima.
2. **`Distracted`/coral solo en trazos ≥3px y display grande**, jamás texto fino de cuerpo sobre oscuro.
3. **No existe "rojo de peligro" separado** (resuelve A3): los 3 colores de estado son **exclusivos del estado del conductor**; un incidente hereda el color de su estado. Errores de sistema/conectividad/WiFi → `brand-signal` cian o toast neutro, **nunca** un color de estado.
4. **El color saturado SOLO existe para el estado.** Cero color de estado en botones, links, decoración o chrome. (Regla de Tablero Editorial, override al 60-30-10 suave.)
5. **Teal de estado NO compite con el acento de marca cian** (`#3DA9FC` azul ≠ `#2DD4BF` teal-verde) → "azul de marca" nunca se confunde con "estás alerta".

### 2.4 Redundancia por forma/icono (canal obligatorio además del color)
Cada estado lleva **silueta de icono distinta** (no tres variantes del mismo ojo) + **palabra en español** siempre visible + **cadencia de movimiento** propia (ver §8.3):

| Estado | Icono (silueta) | Palabra | Movimiento |
|---|---|---|---|
| Alerta | Escudo-check / ojo abierto, anillo estable | `ALERTA` | respira lento (4s) |
| Somnoliento | Párpado caído / media-luna | `SOMNOLENCIA` | droop-blink lento (~1.5s) |
| Distraído | Mirada desviada / flechas off-axis | `DISTRACCIÓN` | un flash agudo único (no estrobo) |
| Sin señal | Cámara buscando / señal tachada | `BUSCANDO ROSTRO` | spinner mínimo |

Iconos custom dibujados a la **geometría exacta de Lucide** (viewBox 24×24, stroke 2px, linecap redondo) para que se integren sin costura con el resto del set. Se renderizan como SVG inline con `currentColor`.

---

## 3) Tipografía

**Dos familias + una mono, todas self-hosted vía `@fontsource` (cero CDN, cero licencia, sirven desde el mismo EC2/Caddy).**

| Rol | Fuente | Uso |
|---|---|---|
| **Display** | **Space Grotesk** (700/500) | Palabra de estado gigante (`ALERTA`), números KPI, wordmark, titulares. Geométrica, "instrument-cluster", legible a 120px en el HUD y sobrevive glare (strokes gruesos). |
| **Body / UI** | **Geist Sans** (400/500/600) | Todo el texto de UI, labels, tablas, botones, párrafos. Cobertura completa es-MX (á é í ó ñ ü). |
| **Mono (numéricos en vivo)** | **Geist Mono** (500) | Velocidad (km/h), confianza %, lat/lng, timestamps, ticks de chart, session IDs. **`tabular-nums` obligatorio** para que los números del polling (cada 1-2s) NO bailen/reflowen. |

**[RESUELTO] — Inter queda fuera:** varias skills prohíben Inter como "default premium" y Nocturne fue penalizado por usarlo. Space Grotesk da el carácter de instrumento que Inter no; Geist es el caballo de batalla legible. **Regla de pareja:** Space Grotesk solo display/números-como-titular; Geist Sans para lo que se lee; Geist Mono solo valores medidos en vivo. Nunca las tres en una misma línea.

**[RESUELTO] — Clash Display / Fraunces descartadas:** Clash (Cobalt) y Fraunces (Tablero) son hermosas pero suben el riesgo (verificación de glifos de acentos en `DISTRACCIÓN`/`SOMNOLENCIA`, una 3a familia, Fraunces a 22vw "pretencioso"). Space Grotesk + Geist es el par de menor riesgo con suficiente carácter.

**Escala:** body 16px base · label de estado conductor `clamp(64px, 18vw, 160px)` peso 800 · KPIs 32-40px · jerarquía de headings ratio 1.25 · nada informativo <16px en la vista conductor.

---

## 4) Stack frontend completo

**UN solo proyecto Vite + React + TypeScript, una SPA con dos superficies (`/conductor`, `/manager`), build estático servido por el Caddy que YA corre en el EC2.** Llamadas relativas a `/api` (mismo origen, cero CORS).

### 4.1 Base, framework, build/deploy
| Pieza | Elección | Versión/nota |
|---|---|---|
| Build | **Vite** | `npm create vite -- --template react-ts`. Build = `vite build` → `dist/`. **Sin Next.js** (el backend ya es FastAPI+Caddy; SSR sería peso muerto y un runtime más que cuidar). |
| Runtime | **React 18 + react-dom + TypeScript strict** | Pinear Node 20 LTS (`.nvmrc`) para que el build del EC2 y la laptop coincidan. |
| Routing | **React Router v6** (`createBrowserRouter`, modo SPA) | Rutas `/` (selector), `/conductor`, `/manager`, `/manager/viajes/:id`. **Code-split por superficie con `React.lazy`** → el celular del conductor NO baja el código de mapas/charts del manager. |
| Styling | **Tailwind** + tokens en CSS custom properties | Toda la paleta (§2) como variables HSL → una sola fuente de verdad para ambas superficies. |
| Componentes | **shadcn/ui** (Radix, copy-in) | Card, Badge, Table, Tabs, Dialog, Sheet, Tooltip, Skeleton, Switch, Select, Sonner. Se restilizan solo con tokens. Accesibilidad (focus trap, ARIA, teclado) incluida. |
| Iconos | **lucide-react** | Set base de UI; los 3 glifos de estado custom se dibujan a su geometría. |
| Deploy | **Caddy sirve `dist/`** (reemplaza el bloque `handle {}` placeholder actual) | `encode zstd gzip` + **`try_files {path} /index.html`** (SPA fallback — obligatorio o los deep-links `/manager/viajes/123` dan 404). HTTPS Let's Encrypt ya vivo → la PWA cumple origen seguro. |

**Caddyfile (concreto, reemplaza el placeholder):**
```
34-205-126-89.nip.io {
  encode zstd gzip
  handle_path /api/* { reverse_proxy api:8000 }
  handle {
    root * /srv
    try_files {path} /index.html
    file_server
  }
}
```
Servicio `web` con build multi-stage (`docker compose up -d --build web`) para que el redeploy del front use el mismo flujo que el API. Alternativa: montar `./web/dist:/srv:ro` como volumen.

### 4.2 Data / polling
**TanStack Query v5** es la pieza más cargada de responsabilidad (el backend es polling, sin websockets).
- `QueryClient` global: `refetchOnWindowFocus:false, retry:2, staleTime:1000`.
- **Conductor (en vivo):** `useQuery({ queryKey:['alerts', sessionId, since], refetchInterval:1500, refetchIntervalInBackground:false })` contra `GET /sessions/{id}/alerts?since=` con cursor `since` que avanza al `ts` más nuevo en cada poll → payload incremental. El pausado en background ahorra batería.
- **Intervalo dinámico** (protege el t3.small y la batería): `refetchInterval: (q) => !q.state.data?.active ? false : (q.state.data?.state === 'Alert' ? 2000 : 1200)`.
- **Manager:** un viaje **finalizado es inmutable** → `staleTime: Infinity`, sin polling. Un viaje **activo** poll cada 1.5-2s (mapa + timeline creciendo en vivo = segundo wow). Pausar con `document.hidden`.
- **Mutaciones** (`POST /sessions`, `POST /sessions/{id}/end`) vía `useMutation`, invalidando `['current-session']` y `['sessions']`.
- **NUNCA cachear `/api/*` en el service worker** (un "Alerta" viejo mostrado como vivo es el peor fallo de demo).

### 4.3 Mapas
**MapLibre GL JS + `react-map-gl/maplibre`** (open-source, **sin token, sin facturación** — crítico con presupuesto AWS topado).
- **Estilo claro/gris** (CARTO Positron / Protomaps grayscale) para que **ruta + incidentes (en color de estado) sean lo único saturado**.
- Ruta = **un solo GeoJSON `LineString`** en una capa `line`, coloreada por estado con expresión `["match", ["get","state"], "alert", "#1FB6C1", ...]` (NO N capas → mantiene 60fps). Casing blanco de 8px debajo de la línea de 5px = look premium.
- Incidentes = markers de punto con color+forma del estado; click → popup con snapshot.
- Fit-to-route con `@turf/bbox`. Decimar con `@turf/simplify` si el track pasa de ~2000 puntos.
- En vivo: `map.getSource('track').setData(geojson)` por poll (re-render incremental, sin parpadeo).

**[RESUELTO] — MapLibre vs Leaflet:** la exploración se dividió (estabilidad→Leaflet vs wow→MapLibre). **Se elige MapLibre** porque el mapa vive **solo en el dashboard desktop** (no en el celular), así que el riesgo de WebGL en GPU débil **no aplica**, y el render vectorial es el "wow" premium. Leaflet+OSM queda como **plan B documentado** si MapLibre da problemas.

### 4.4 Charts
**Recharts** vía el wrapper oficial `Chart` de shadcn (`ChartContainer`/`ChartConfig`) → heredan tipografía, theming por CSS-vars y los hex de estado, sin pelear con el styling default.
- **Trip Ribbon / timeline de estados:** **NO** es line/area chart. Es una banda categórica — SVG plano a mano (`<rect>` por *run* de estado colapsado, escala con `d3-scale`) o BarChart horizontal apilado. ~150 líneas, se ve mejor que cualquier componente genérico. Presupuestar ~1 día solo para esto.
- Velocidad vs tiempo: `AreaChart` con `ReferenceDot` en incidentes.
- Analítica de flota: `BarChart`/donut (no pie con muchas rebanadas).

**[RESUELTO] — Recharts y no visx/Tremor/Chart.js:** visx es low-level (demasiada plomería para 2 semanas); Tremor trae su propio design system que choca con shadcn; Chart.js es canvas y se integra peor. Recharts crudo bajo shadcn evita el doble design system.

### 4.5 PWA
**`vite-plugin-pwa`** (Workbox), `registerType:'autoUpdate'`, precache **solo del app-shell**.
- Manifest **name-agnostic** (`name`/`short_name` desde un solo `app.config.ts`): `display:'standalone'`, `theme_color:#0A0E14`, set de iconos 192/512 + 512 maskable.
- Runtime caching: `/api/*` → **`NetworkOnly`** (jamás cachear estado en vivo); snapshots S3 → `CacheFirst` con expiración; tiles → `StaleWhileRevalidate`.
- **NO offline-first** (se asume WiFi constante). Banner "Reconectando…" desde `isPaused`/`navigator.onLine`.
- **Screen Wake Lock API nativa** (sin librería, ~25 líneas) para que el celular montado no se apague. **Detalle crítico:** el lock se libera al ir a background → **re-adquirir en `visibilitychange`**. Adquirir en el tap de "Iniciar viaje" (gesto de usuario).

### 4.6 Animación
**Framer Motion (`motion/react`) usado con moderación** — solo para los 3-4 momentos hero. El anillo de estado base es **SVG + CSS keyframes puro** (no depende de Framer). Ver §8.3 para el sistema completo.

---

## 5) UX vista conductor (in-car)

### 5.1 Principios
1. **AMBIENT por defecto, FOREGROUND solo en evento.** El estado Alerta (~95% del tiempo) es **visualmente aburrido a propósito**: fondo sólido calmo, un anillo que respira lento, cero números corriendo. El presupuesto de wow va a la **transición** hacia alerta y al dashboard. Una pantalla que el conductor NO necesita mirar es la pantalla bien diseñada.
2. **Regla de la ojeada única (<1.5s, estilo NHTSA).** El estado se entiende de reojo sin fijar la vista: color de pantalla + un icono grande + una palabra. Nada que requiera leer o contar.
3. **Redundancia obligatoria (nunca solo color):** color + icono + palabra + movimiento, simultáneos. (Cubre daltonismo, glare y visión periférica.)
4. **Escalar la intrusión con la severidad** y **auto-despachar** la alerta al volver a Alerta. El conductor **nunca toca la pantalla para silenciar** (tocar = distraerse = la paradoja). El único tap legítimo en viaje es "Terminar viaje", fuera del flujo visual.
5. **Diseñar para el peor entorno, no para el screenshot:** alto contraste de luminancia, tipografía masiva, legible a 60-80cm de reojo con sol pegando.

### 5.2 La paradoja anti-distracción — cómo se resuelve a nivel de sistema **[RESUELTO]**
Tres mecanismos, no uno:
- **Color = solo estado** (regla de Tablero): pantalla monocroma/calma = "sigue manejando". El color irrumpe solo cuando importa.
- **Presupuesto de movimiento atado a eventos:** se respira solo en Alerta; el movimiento se "gasta" únicamente en el instante en que la atención debe redirigirse. (Implementación literal de la paradoja en código, gated tras `prefers-reduced-motion`.)
- **Refuerzo audio/háptico** (porque un conductor somnoliento/distraído por definición NO está mirando bien la pantalla): tono Web Audio + voz `SpeechSynthesis` es-MX (frases ≤4 palabras: "Ojos al camino") + `navigator.vibrate()`. El `AudioContext` se desbloquea en el tap de "Iniciar viaje". Toggle de mute disponible.

### 5.3 Pantallas
| Pantalla | Descripción |
|---|---|
| **Pre-viaje / Splash** | Logo placeholder + check "Cámara y sensor conectados" + botón grande **"Iniciar viaje"** (la app es dueña de la sesión: `POST /sessions`; arma Wake Lock + AudioContext aquí). |
| **Estado — ALERTA** (base, ~95%) | Fondo `#0A0E14`; **anillo/orbe de estado teal** dead-center con la palabra `ALERTA` + traza de atención. Respira a ~0.25Hz (señal de "vivo y vigilando"). Banda inferior: chips discretos de velocidad + dot `#3DA9FC` "en vivo". Debe dar ganas de NO mirarla. |
| **Estado — SOMNOLENCIA** | Anillo/fondo viran a ámbar (cross-fade 400ms). Icono párpado/media-luna + `SOMNOLENCIA`. Pulso lento envolvente. Audio: 2 pulsos cálidos + voz "Atención, somnolencia". |
| **Estado — DISTRACCIÓN** | Vira a magenta `#FF3D6E`. Icono mirada-desviada + `DISTRACCIÓN` + sub "Ojos al camino". Edge-glow periférico + un flash agudo único. Audio: beeps cortos agudos + voz "Ojos al camino". **Umbral de disparo más conservador** (es la clase más débil del modelo). |
| **Overlay de incidente** (efímero, sobre cualquier estado) | Borde de pantalla se ilumina (frame glow 6-8px) + toast "Incidente registrado · 14:32 · 82 km/h" + micro-thumbnail + 1 vibración firme. Auto-dismiss ~2s. NO bloquea, NO requiere tap. Refuerza al juez que se captura evidencia. |
| **Sin señal / Reconectando** | Estado neutro gris `#5E6F84` + "Buscando rostro…". **NUNCA congelar el último estado** (un teal "Alerta" viejo durante una caída engaña al juez y es peligroso). Se dispara si el último sample tiene >3-4s. |

### 5.4 Anti-flicker (histéresis) — requisito de seguridad Y de pulido
Con polling 1-2s y confianza ruidosa (modelo ~63%), **NO** renderizar el sample crudo. Máquina de estados con histéresis asimétrica:
- **Subir de severidad:** requiere 2 samples consecutivos del mismo estado **O** confianza ≥0.75 (≥0.8 para `Distracted`).
- **Bajar a Alerta:** requiere ~3 samples consistentes o 4-5s sin alerta.
- Suavizar confianza con EMA (α≈0.4).
Una pantalla que parpadea teal/rojo/teal por ruido se ve rota y, peor, distrae. **La calma visual es una feature de seguridad.**

### 5.5 Accesibilidad / glare
- Texto blanco `#EAF0F7` o tinta `#0A0E14` sobre fills de estado, ratio ≥4.5:1 (objetivo 7:1 para el estado principal).
- Iconos con relleno sólido (no líneas finas que el glare borra).
- PWA fullscreen (`display:standalone`), orientación bloqueada según montaje, Wake Lock activo, sin chrome del navegador.
- **Validar antes de la expo (check de 30 min):** pasar las 3 pantallas por simulador de daltonismo (DevTools vision-deficiency: protan/deuter/tritan + escala de grises) y probar el celular físico bajo luz fuerte. Si Somnoliento y Distraído se confunden, subir diferencia de icono/movimiento (no solo hue).
- **Modo demo determinista** (`?demo=1`, gesto oculto): reproduce la secuencia Alerta→Somnoliento→incidente→Distraído→Alerta, por si el modelo se pone ruidoso frente a jueces. Estabilidad > realismo en 2 min.

---

## 6) UX dashboard manager (desktop)

### 6.1 Arquitectura de información (progressive disclosure, 3 niveles)
`Resumen de flota` → `Lista de viajes` → `Detalle de viaje`. El manager nunca ve los 3 niveles a la vez. **El detalle de viaje es la pantalla estrella** (70% del pulido va ahí). Shell: sidebar fijo 240px (nav + brand token + link "Vista conductor") + breadcrumb.

### 6.2 Pantalla 1 — Resumen de flota
Fila de 3-4 stat cards calculadas de `GET /sessions`: **Viajes registrados**, **Incidentes totales** (suma `incident_count`), **Viajes activos ahora**, **Viaje más reciente**. Debajo: BarChart "Incidentes por viaje" + tabla compacta de viajes recientes.

**[RESUELTO — restricción dura]:** **NO existe endpoint de analítica de flota.** `GET /sessions` solo da `id, driver_id, started_at, ended_at, status, incident_count`. Cualquier KPI más allá de "count de viajes" y "suma de incident_count" (ej. % tiempo somnoliento, incidentes/hora) **debe computarse client-side** haciendo fan-out de `GET /sessions/{id}` por cada sesión (cachear; en demo son pocas). **No prometer métricas que la API no soporta barato.**

### 6.3 Pantalla 2 — Lista de viajes
Tabla (TanStack Table v8 headless) con: Estado (chip activo/finalizado), Conductor (`driver_id` o "Sin asignar"), Inicio (es-MX `6 jun, 14:32`), Duración (computada, "—" si activo), **Incidentes** (badge rojo si >0), mini state-mix bar (60px), chevron. Ordenable; fila clickeable → detalle. Filtro "Solo con incidentes" + rango (Hoy/7d/Todo).

### 6.4 Pantalla 3 — Detalle de viaje (LA ESTRELLA)
Layout 2 columnas + franja inferior, **acopladas por un cursor de tiempo compartido** (brushing & linking — el wow nº1):
- **Izquierda (~55%) — Mapa:** MapLibre, polyline coloreada por estado, pins de incidente (color+forma), marcador "fantasma" que sigue el scrub del ribbon.
- **Inferior full-width — Trip Ribbon / timeline de estados:** banda de 8px+ con *runs* de estado coloreados sobre el eje de tiempo, ticks de incidente clickeables, cursor vertical con tooltip (hora + estado + confianza). Click → fija `selectedTs` (sincroniza mapa + snapshots). Para viaje activo, **crece en vivo**.
- **Derecha (~45%) — Incidentes/snapshots:** cards con chip de estado, hora, velocidad, **thumbnail del snapshot**, badge "Frenado brusco" si `harsh_event`. Hover resalta pin en mapa + tick en ribbon. Click → **Sheet lateral** (no modal, mantiene contexto del mapa) con foto grande + metadata + "Ver en mapa". **Encuadre "caja negra"** (frame recuperado con velocidad+coordenada estampada).
- **Header:** KPIs (duración, distancia Haversine sobre `track`, vel. máx, # incidentes por tipo, "índice de riesgo" = % tiempo no-Alerta).

**Derivar, no asumir:** `track[]` y `states[]` son streams independientes con `ts` propios y cadencias distintas. Para colorear la polyline: por cada track point, hallar el estado vigente = último `state_sample` con `ts ≤ track.ts` (sort + búsqueda binaria, `useMemo`). Para el ribbon: colapsar `states[]` consecutivos en *runs* (NUNCA un rect por sample). `state` es **TEXT libre** → normalizar con un mapa canónico y fallback gris "Desconocido". Timestamps son **UTC (TIMESTAMPTZ)** → mostrar en `America/Mexico_City` con `date-fns-tz`.

### 6.5 Analítica
Donut "Distribución de estados" + stacked bar "Estado por hora del día" + "Velocidad vs incidentes" (bucketear velocidad en bandas, **NO** scatter — la data es escasa; nulls van a "sin dato", **nunca** coercer a 0). Todos con la paleta de §2 + etiqueta directa (no leyenda por color sola).

### 6.6 Estados vacíos/carga/error (entregable crítico, no pulido opcional)
Skeletons que matchean el layout final; empty positivo ("Sin incidentes en este viaje. Conducción 100% en alerta." con check teal); error por región con "Reintentar" (scoped, no full-page crash); "Actualizado hace Xs" + distinción **En vivo** (badge pulsante) vs **Finalizado**.

### 6.7 BLOQUEADOR conocido (en ruta crítica) **[RESUELTO — acción para el equipo]**
`incidents.snapshot_key` es una **S3 key, NO una URL**, y el bucket tiene Block Public Access ON. El contrato actual solo tiene `POST /uploads/presign` (para PUT/subida), **no hay endpoint de lectura**. **Sin esto, las fotos de incidentes NO cargan** — y son parte del wow del detalle.
- **Acción requerida al backend:** agregar `GET /sessions/{id}/incidents/{iid}/snapshot` que **302-redirige a una URL GET presignada** (el front solo pone `<img src>`), o `GET /uploads/presign-get?key=`.
- **Mientras tanto:** el slot degrada a placeholder con icono de cámara + estado. **Nunca un broken-image glyph frente a jueces.**

---

## 7) Modo conductor vs manager sin auth

**[RESUELTO] Approach: rutas separadas por path + dos shells visuales independientes, sin login.**

- `react-router` con 3 rutas: `/` = **lobby** (selector con 2 cards grandes: "Soy conductor" → `/conductor`, "Panel de flota" → `/manager`); `/conductor` = PWA in-car (oscuro, fullscreen, glanceable, Wake Lock); `/manager` = dashboard (oscuro o claro, denso).
- `/` **auto-redirige por ancho de viewport SOLO la primera vez** (`matchMedia('(max-width: 768px)')` → conductor; si no → manager), pero **siempre** deja un link visible para cambiar a discreción del demo. Persistir elección en `localStorage`.
- **Cambiar de modo desde el conductor:** gesto deliberado y difícil de hacer por accidente — **long-press 1.5s** sobre el wordmark → hoja inferior con las dos opciones. **Nunca un botón suelto** que se toque manejando.

**Por qué rutas y no toggle in-app ni responsive único:**
1. La paradoja exige que el conductor **no tenga chrome de manager** (menús/switches que distraigan) → dos shells limpios.
2. En la demo: el celular abre directo `…/conductor` (QR en el volante Forza) y el laptop abre `…/manager` → **cero clics de setup** frente a jueces, cero riesgo de caer en la vista equivocada.
3. **Bundle-split natural** (`React.lazy`): `/conductor` carga ligero (crítico en celular), `/manager` carga su mapa solo al entrar.

**Descartados explícitamente:** detección por user-agent (frágil), un layout responsive que se reacomoda conductor↔manager (acopla dos UX opuestas → más bugs, viola "estable en 2 semanas"), auth/login real (fuera de scope MVP).

**Name-agnostic:** un único `export const BRAND` consumido por title, sidebar, lobby y manifest. Cambiar el nombre comercial = editar **1 línea**. Logo = un solo SVG reemplazable. Nada de marca hardcodeado en componentes.

---

## 8) Marca: logo / iconografía / motion

### 8.1 Concepto
**"El copiloto que mira el camino contigo"** (NO un ojo de vigilancia). Este reframe mata el cliché distópico del face-scan y hace al producto **protector, no acusador** — resuelve la paradoja a nivel de marca. Registro: entre HMI automotriz (Tesla/Polestar/Mobileye) y minimalismo dev-tool (Linear/Vercel).

### 8.2 Logo — "El Arco de Atención" (recomendado, name-agnostic)
Un **arco abierto + dot focal** que lee simultáneamente como (a) ojo entreabierto calmo, (b) carretera al horizonte, (c) cono de radar/atención. Sin palabras, escala a favicon 16px. **La "apertura" del arco codifica estado en producto** (abierto=Alerta, angosto=Somnoliento, off-axis=Distraído) → **el logo ES el indicador de estado**.
- Construcción: un solo SVG path sobre grid de 8px, stroke 2px vía `currentColor`, **monocromo como fuente de verdad** (color aplicado por contexto, nunca horneado), sin gradiente en la marca primaria.
- Rutas alternas si el nombre lo fuerza: "El Foco" (retícula/viewfinder) o "La Senda" (carriles convergentes + chevron).

### 8.3 Motion — "Respiración calmada, alarma decidida" (dos regímenes)
**Motor único: Framer Motion (`motion/react`)** + transiciones Tailwind para micro-hovers triviales. Solo `transform`/`opacity`/`filter` (GPU, fuera del main thread — crítico porque el hilo principal ya hace polling). **No GSAP, no react-spring** (fragmenta el bundle/mental model de un estudiante en 2 semanas).

| Régimen | Superficie | Carácter |
|---|---|---|
| **A — Conductor** | in-car | Lento, suave, **periférico**. El estado base respira; transiciones = **crossfades de color tweenados** de 400ms (esconden el jitter del polling: el ojo lee fluidez, no un salto). |
| **B — Manager** | desktop | Ágil tipo data-app premium: stagger de listas, shared-element (`layoutId`) al abrir detalle, dibujado progresivo de la ruta (`pathLength` 0→1), springs cortos en hover. Aquí vive el wow visual del juez. |

**Tokens:** una sola curva marca `cubic-bezier(0.22,1,0.36,1)` (ease-out suave); durations 80ms (tap) / 160ms (hover) / 240ms (entrada) / 400ms (crossfade de estado conductor); respiración base 2200ms. Spring `{stiffness:400, damping:30}` **solo** en el manager. **PROHIBIDO** `ease-in-out` simétrico salvo el breathing, y `linear` salvo shimmer de skeleton.

**Ley de movimiento in-car (guardrail de la paradoja):**
- Alerta = respira lento (vivo, tranquilo). El breathing **se detiene** durante una alerta — el contraste calma↔firmeza es lo que hace legible la gravedad.
- Drowsy/Distracted = **edge-glow periférico** (no demanda lectura foveal) + morph del icono + **un** pulso/flash único. **PROHIBIDO:** parpadeo repetido, flash a pantalla completa, shake de toda la UI, estrobo. La intensidad se logra con **saturación y tamaño**, no con frecuencia de parpadeo.
- `useReducedMotion()` → todo cae a swap instantáneo de color/label, conservando la info. **No opcional.**

**Prohibiciones duras (del "don't overdo it"):** sin parallax, sin scroll-reveals en el dashboard de datos, sin glassmorphism/`backdrop-filter` animado (mata FPS en el celular — penalizó a Nocturne), sin confetti, sin 3D, sin animar `width/height/top/left`. Premium = timing y easing impecables sobre un vocabulario pequeño.

### 8.4 Iconografía
Lucide (MIT) como set base de UI; los 3 glifos de estado custom dibujados a su geometría exacta (24×24, stroke 2px, caps redondos), SVG inline con `currentColor`. **Componente reutilizable `StatusBadge`** (icono + pill de color + label) compartido por ambas superficies.

---

## 9) Orden de construcción sugerido (~2 semanas, 1 estudiante)

**Filosofía: construir primero la superficie segura/de menor riesgo, dejar el wow frágil al final, blindar la demo.**

| Días | Trabajo | Por qué en este orden |
|---|---|---|
| **1** | Scaffold Vite+React+TS+Tailwind+shadcn. `QueryClient` + `api.ts` tipado (los ~11 endpoints) + tokens de color (§2) como CSS vars + React Router. **Poner `GET /current-session` polleando en pantalla contra el EC2 vivo el día 1.** `.nvmrc` Node 20. | Prueba la espina de polling mismo-origen antes de cualquier UI. Node mismatch es el fallo silencioso nº1 de Vite. |
| **2-5** | **Dashboard manager:** shell + resumen + lista de viajes + detalle (mapa + Trip Ribbon + lista de incidentes con placeholders). | Es CRUD-read sobre `GET /sessions` + `/sessions/{id}` — el sweet spot de shadcn y lo que carga la mayor impresión "premium". Menor riesgo, primero. **Levantar el bloqueador de snapshots (§6.7) con el backend HOY.** |
| **6-9** | **Vista conductor:** anillo de estado (SVG+CSS) + traza de atención + máquina de histéresis + crossfades + audio/háptico + Wake Lock (con re-adquisición en `visibilitychange`) + botones Iniciar/Terminar. | Visualmente más simple pero UX de mayor riesgo (glanceable, no distraer). Es la cara que ven los jueces → pulir al máximo. |
| **10-11** | PWA (manifest+SW, **`/api` NetworkOnly**) + install en el celular real + **pase de daltonismo/glare** (DevTools CVD + sol físico) + modo demo determinista (`?demo=1`). | El check de accesibilidad evita el peor failure de demo y debe hacerse en el dispositivo real. |
| **12-14** | Buffer: deploy de `dist/` por Caddy (+ `try_files`), testing en el rig Forza sobre el WiFi real, pulido de estados vacíos/error, ensayo general. **No deployar el día de la demo** (`autoUpdate` SW puede servir build viejo). | |

**Si algo se corta por tiempo:** se corta el **Resumen de flota** (no el detalle de viaje), y la analítica elaborada — **nunca** la redundancia color+icono+palabra ni el anillo de estado.

---

### Resumen de contradicciones resueltas (índice)
1. **Oscuro vs claro** (§1.1): chasis oscuro + vista conductor siempre oscura; wow en CSS puro (disciplina de Faro); manager claro opcional.
2. **Trío de color** (§2.2): teal/ámbar/magenta (mejor CVD medido) sobre verde/ámbar/rojo ingenuo.
3. **Tipografía** (§3): Space Grotesk + Geist (+ Geist Mono); Inter, Clash y Fraunces fuera.
4. **Mapa** (§4.3): MapLibre (solo desktop → sin riesgo WebGL); Leaflet plan B.
5. **Charts** (§4.4): Recharts bajo shadcn, no visx/Tremor.
6. **Modo conductor/manager** (§7): rutas separadas + long-press, no toggle/responsive/auth.
7. **Bloqueador de snapshots** (§6.7): requiere endpoint GET presignado del backend; placeholder mientras tanto.
8. **Analítica de flota** (§6.2): client-side por fan-out; no prometer métricas no soportadas.

**Regla de equipo (override):** es proyecto de clase — **cero atribución de IA** en commits, PRs, comentarios o docs. Todo se entrega como trabajo del equipo.

---

# Punch-list del crítico de coherencia

Confirmed: the API and frontend don't exist yet — this is a greenfield spec. The backend contract is entirely on-paper, which sharpens the contract-risk findings. I have everything I need. The contrast math contradicts the spec's own central claims, which is the highest-value finding.

Key measured facts vs. the spec's claims:
- White on teal-fill `#0F7E86` = **4.21:1** (spec claims ~5.0:1). For 120px display text this clears AA-large (3:1) but the spec's number is wrong and it fails AA-normal (4.5:1).
- White on magenta-fill `#C42852` = **4.85:1** (spec claims ~5.0:1) — passes AA-normal, fine.
- Navy on amber = **10.58:1** — excellent, spec correct.
- The **fills** trio (`#0F7E86`/`#FFB020`/`#C42852`) the spec actually paints has grayscale min-gap **0.029** and deuteranopia distance **61.7** — i.e. it *fails* the very separability test the spec used to reject other palettes. Only the *base* trio scores well (min-gap 0.119). The spec conflates the two.

Here is the punch-list.

---

# Punch-list crítica — Spec DMS (priorizada)

Revisión adversarial. No repito el spec; cada ítem es accionable. Donde corregí math, los números son WCAG 2.x / Machado-2009 CVD calculados sobre los hex del propio spec.

---

## ALTA (rompen accesibilidad, contradicen el spec, o ponen en riesgo la demo)

**A1 — El argumento central "oscuro gana porque blanco sobre fill supera AA" es parcialmente falso con los hex elegidos.**
Verifiqué los ratios reales:
- Blanco `#EAF0F7` sobre teal-fill `#0F7E86` = **4.21:1**, no "~5.0:1" como afirma §2.2. Pasa AA-large (≥3:1) solo porque la palabra de estado es enorme; **reprueba AA-normal (4.5:1)**. Cualquier texto blanco no-gigante sobre ese fill (sub-labels, "ALERTA" si baja de ~24px bold) es no-conforme.
- Blanco sobre magenta-fill `#C42852` = **4.85:1** (ok, pero tampoco "≥5.0").
- Navy sobre ámbar = **10.58:1** (correcto).

Acción: o subes el fill teal a algo como `#0B6A70`/`#0A5A5F` para cruzar 4.5:1 con blanco, o **fijas como regla** que el texto sobre teal-fill es siempre ≥ display-large y nunca cuerpo. Corregir los números "~5.0:1" en la tabla §2.2 — son aspiracionales, no medidos.

**A2 — Contradicción dura: el trío de *fills* que el spec pinta REPRUEBA la prueba de CVD que el spec usó para elegirlo.**
El spec elige teal/ámbar/magenta porque el trío *base* tiene min-gap grayscale **0.119** y buena distancia CVD. Pero la vista conductor pinta **fondo/anillo con los fills** `#0F7E86`/`#FFB020`/`#C42852`, cuya separabilidad real es:
- grayscale min-gap = **0.029** (peor que el Cobalt 0.075 que el spec descartó).
- deuteranopia, par teal↔magenta = **61.7** (el spec rechazó Tablero Editorial por colapsar a 45.8; 61.7 es del mismo orden de riesgo, no "sin ningún par cerca del umbral").

O sea: la separabilidad medida aplica a los *base*, pero la superficie que el conductor ve de reojo usa los *deep*. Acción: definir explícitamente **qué tono pinta el fondo/anillo del conductor**. Recomendación: el conductor usa los **base saturados** (`#1FB6C1`/`#FFB020`/`#FF3D6E`) como color de superficie (con texto navy o un chip de texto sobre placa oscura), reservando los *deep* solo para fills-con-texto-blanco-encima en el manager. Y re-correr el pase CVD sobre los tonos que realmente se pintan, no sobre los base.

**A3 — `#FF3D6E` ("magenta-rojo") como "Distracción" colisiona semánticamente con la regla "rojo = solo peligro del conductor".**
§2.3 regla 3 dice "el rojo se reserva para peligro del conductor" y errores van en cian/gris. Pero `#FF3D6E` **es percibido como rojo** (sobre todo en protanopia, donde el magenta pierde el componente azul). Entonces: Distracción ya está "gastando" el rojo. Si además un incidente/harsh_event usa el mismo `#FF3D6E` (como dice §2.2 fila incidente), el conductor ve el mismo rojo para dos cosas distintas (estado distraído vs. evento registrado). Acción: decidir si Distracción y "peligro/incidente" comparten color (entonces la regla "rojo solo peligro" es falsa, hay que reescribirla) o separarlos. No pueden coexistir como están escritas.

**A4 — `ink-low #5E6F84` reprueba contraste y se usa para info real.**
`#5E6F84` sobre void = **3.76:1**; sobre panel `#121821` = **3.46:1**. §2.1 lo asigna a "timestamps, footnotes" y §2.2 lo usa como **color del estado "Sin señal / Buscando rostro"**. Reprueba AA (4.5:1) e incluso AA-large (3:1) está al borde sobre panel. Un timestamp es texto ≤16px → no-conforme. Peor: "Buscando rostro" es un estado de seguridad que debe leerse claro. Acción: subir `ink-low` a ~`#7A8BA0` (cruza 4.5:1) o prohibirlo para texto <18px. El gris de "Sin señal" debe ser legible, no el más tenue de la rampa.

**A5 — El bloqueador de snapshots (§6.7) está en ruta crítica y depende de un backend que NO existe en el repo.**
Confirmé: no hay código de API en el repositorio (solo `legacy/main.py`, que es el script OpenCV de MediaPipe). Todo el contrato (`POST /sessions`, `GET /sessions/{id}/alerts?since=`, presign, etc.) es **on-paper**. El spec asume que "la API ya existe" pero no hay nada que lo respalde aquí. Esto convierte A5 y A6 de "coordinación" a "riesgo de cronograma": el endpoint GET-presign para leer snapshots **y** todos los demás endpoints podrían no estar listos el día 1. Acción inmediata: validar contra el EC2 vivo (no contra el repo) que cada endpoint del §4.6 responde, **antes** de comprometer el plan de §9. Si el backend lo mantiene otra persona (Pasaye, según memoria), el endpoint de lectura de snapshot es una dependencia bloqueante de un tercero — escalarla hoy.

**A6 — `?demo=1` determinista es indispensable, pero el spec lo trata como ítem de día 10-11.**
§5.5 lo llama "por si el modelo se pone ruidoso". Con un modelo a ~63% de accuracy (memoria del proyecto) y `Distracted` siendo la clase más débil, la probabilidad de que la demo en vivo se vea errática frente a jueces es **alta**, no marginal. El modo demo no es red de seguridad, es probablemente **el modo principal de la presentación**. Acción: subirlo a infra de día 1-2 (un mock del cliente de API que reproduce un guion scriptado), para que TODA la UI se desarrolle contra datos deterministas y el "modelo real" sea el camino opcional. Esto también desbloquea desarrollo aunque el backend (A5) llegue tarde.

**A7 — Histéresis (§5.4) descrita pero ausente del plan de construcción y subestimada.**
§5.4 es un requisito de seguridad ("la calma visual es feature de seguridad") pero el cronograma §9 la mete embebida en "días 6-9" junto con anillo+audio+háptico+wake-lock+botones. Una máquina de estados con histéresis asimétrica + EMA + de-bounce de polling, bien hecha y testeada contra streams ruidosos, es **fácilmente 1.5-2 días sola** y es la lógica con más bugs potenciales (race conditions entre poll, EMA, y transición de UI). Acción: presupuestarla aparte y testearla con el guion de `?demo=1` que incluya ruido inyectado. Si esto falla, la pantalla parpadea frente a jueces — el peor resultado.

---

## MEDIA (riesgos de shippability, "overdone", o decisiones faltantes)

**M1 — El "Arco de Atención que cambia de forma según el estado" (§8.2) es scope creep disfrazado de logo.**
"El logo ES el indicador de estado" suena elegante pero implica: un SVG con tres morfologías animadas, sincronizado con la máquina de estados, presente en favicon/wordmark/lobby/manifest. Es exactamente el tipo de "wow frágil" que el dueño pidió evitar ("don't overdo it"). Un favicon no comunica estado a un conductor. Acción: cortar a un logo **estático** monocromo. El indicador de estado vive en el anillo (§5.3), no en la marca. Esto también simplifica el requisito name-agnostic.

**M2 — Tres familias tipográficas self-hosted es más riesgo del que el spec admite.**
§3 descarta Clash/Fraunces por "verificación de glifos de acentos" pero luego mete **tres** familias (Space Grotesk + Geist Sans + Geist Mono) vía `@fontsource`. Space Grotesk tiene cobertura latina pero **conviene verificar explícitamente** `Ó`, `Í`, `Ñ` en `DISTRACCIÓN`/`SOMNOLENCIA` a 120px (es display, los defectos de hinting se ven). Tres familias = 3 fallback chains, 3 FOUT potenciales, más peso. Acción: confirmar que Space Grotesk renderiza los acentos display antes de comprometerla; considerar si Geist Sans + Geist Mono bastan (Space Grotesk solo para el número/palabra hero). Es un "wow" caro para 2 semanas.

**M3 — `navigator.vibrate()` no funciona en iOS Safari — el "money-shot háptico" puede no existir en la demo.**
§5.2 y la tabla de injertos venden `navigator.vibrate()` como momento clave en el volante Forza. **iOS Safari no soporta la Vibration API** (solo Android Chrome/algunos). Si el celular montado es un iPhone, el háptico simplemente no ocurre — sin error, sin feedback. Acción: confirmar HOY qué dispositivo va montado. Si es iPhone, quitar el háptico de la narrativa de venta o cambiar a refuerzo audio (que sí funciona). No prometer al jurado algo que el hardware no hará.

**M4 — Wake Lock + AudioContext + Vibration todos requieren gesto de usuario y HTTPS — encadenados a "Iniciar viaje", pero iOS PWA standalone los limita.**
El spec adquiere los tres en el tap de "Iniciar viaje" (correcto en teoría). Pero en iOS, una PWA en modo `standalone`: Wake Lock API llegó tarde y es inconsistente, `SpeechSynthesis` es-MX puede no tener voz instalada (cae a voz default o silencio), y `AudioContext` se suspende agresivamente al cambiar de foco. Acción: probar la cadena completa en el **dispositivo y navegador reales de la demo** mucho antes del día 10. Tener fallback visual puro si audio/wake-lock fallan (la pantalla puede apagarse a mitad de demo — desastre silencioso).

**M5 — Modo claro opcional del manager (§2.1) es trabajo doble sin pedido explícito.**
El dueño dijo "pulido y estable, don't overdo it". Un segundo tema (claro) para el manager duplica el QA de contraste, de charts, de mapa, de todos los estados — y nadie lo pidió. §1.1 lo justifica como "seguro en desktop" pero seguridad no es razón para construirlo. Acción: cortar el modo claro del MVP. Un solo tema oscuro, bien pulido, es más "premium y estable" que dos temas a medias. Mover a "post-demo".

**M6 — Brushing & linking con cursor de tiempo compartido (§6.4) es el feature de mayor riesgo del manager y se presupuesta junto con todo lo demás.**
Sincronizar Trip Ribbon ↔ marcador fantasma en mapa ↔ cards de incidente, con `selectedTs` compartido, scrub en vivo, y búsqueda binaria de estado-vigente-por-track-point, es ingeniería de estado no trivial. §9 lo mete en "días 2-5" con TODO el manager (shell+resumen+lista+detalle). Acción: presupuestar el linking como su propio bloque (~1.5 días) y tener una versión degradada (hover-only sin scrub) como fallback. El Trip Ribbon estático ya es "wow"; el scrubbing bidireccional es donde se va el tiempo.

**M7 — "Distancia Haversine sobre track" y "índice de riesgo" (§6.4) asumen densidad de datos que un track GPS de simulador Forza puede no tener.**
Distancia por Haversine sumada punto-a-punto **sobreestima** con GPS ruidoso y **subestima** con sampling escaso; en un simulador la "ruta GPS" es sintética/inyectada, así que la cifra puede ser absurda (0 km o miles). El "índice de riesgo = % tiempo no-Alerta" depende de que `state_samples` cubra uniformemente el viaje. Acción: validar con datos reales del rig antes de mostrar números duros a jueces; si la distancia sale rara, mostrar duración + #incidentes (robustos) y omitir km. Un KPI obviamente incorrecto destruye credibilidad de ingeniería.

**M8 — Decisión faltante: ¿cómo se asocia el celular del conductor a una sesión que el manager pueda ver en vivo?**
§7 dice "el celular abre `…/conductor` por QR, el laptop abre `…/manager`". Pero sin auth, ¿cómo sabe el manager **cuál** es la sesión activa para verla en vivo? `GET /current-session` sugiere que hay UNA sesión global activa (singleton). Acción: confirmar que el backend asume **una sola sesión activa a la vez** (modelo singleton para la demo) y documentarlo. Si dos personas pudieran iniciar viaje, el modelo se rompe. Esto afecta directamente el "segundo wow" (manager viendo el viaje activo crecer).

**M9 — `state` como TEXT libre (§6.4) + normalización con fallback "Desconocido" implica que el contrato conductor↔backend↔manager NO está fijado.**
Si el backend acepta texto libre en `state`, nada garantiza que el modelo escriba exactamente `Alert`/`Drowsy`/`Distracted`. La normalización defensiva es correcta, pero revela que **no hay enum compartido**. Acción: fijar un enum canónico en el contrato (no normalizar después). El spec mismo usa `Alert`/`Drowsy`/`Distracted` (inglés) en §2.2 pero las palabras de UI son español (`ALERTA`/`SOMNOLENCIA`/`DISTRACCIÓN`) — confirmar que el mapeo string→label está centralizado y que el modelo emite los strings esperados.

**M10 — Intervalo de polling dinámico (§4.2) lee `q.state.data?.state` pero el endpoint es `/alerts?since=`.**
El snippet `refetchInterval: (q) => q.state.data?.state === 'Alert' ? 2000 : 1200` asume que la respuesta de `/alerts?since=` trae un campo `state` plano del estado actual. Pero un endpoint `alerts?since=` típicamente devuelve un **array de alertas nuevas desde el cursor**, no el estado actual escalar. Acción: confirmar la forma de respuesta de `/alerts?since=`. Si es un array, el estado "actual" hay que derivarlo (último elemento / o un endpoint separado), y el `refetchInterval` necesita otra fuente. Es un detalle que rompe el ahorro de batería si está mal.

---

## BAJA (pulido, consistencia, o nice-to-have mal etiquetado)

**B1 — `staleTime: Infinity` para viajes finalizados + SW `autoUpdate` (§4.2, §9 día 12-14) pueden mostrar un viaje "finalizado" que en realidad siguió.**
Si un viaje se marca finalizado pero el backend luego corrige (raro pero posible en demo con reinicios), el cliente nunca lo refetchea. Bajo riesgo en demo controlada. Acción: aceptar el riesgo o un `staleTime` alto finito (5 min) en vez de Infinity.

**B2 — Wake Lock "re-adquirir en `visibilitychange`" (§4.5) es correcto pero incompleto.**
También se libera en cambios de orientación y en algunos `blur`. Acción: re-adquirir también tras `fullscreenchange` y verificar tras el primer poll. Detalle menor.

**B3 — `theme_color: #0A0E14` en el manifest aplica a AMBAS superficies, pero el manager puede ser claro (§2.1/M5).**
Inconsistencia menor: si el manager fuera claro, la barra de estado del SO seguiría oscura. Se resuelve solo si se corta el modo claro (M5).

**B4 — "Iconos custom dibujados a la geometría exacta de Lucide" (§2.4, §8.4) es trabajo de diseño no trivial para un estudiante.**
Dibujar 4 glifos custom (párpado caído, mirada off-axis, etc.) que se vean tan pulidos como Lucide y a 24×24 stroke-2 requiere habilidad de diseño vectorial. Acción: usar glifos **existentes de Lucide** que ya comunican (`eye`, `eye-off`, `eye-closed` si existe, `scan-face`, `alert-triangle`) en vez de custom. Menor riesgo, mismo efecto. Custom solo si sobra tiempo.

**B5 — `@turf/simplify` a >2000 puntos (§4.3) — el umbral es arbitrario y la simplificación puede borrar el punto exacto de un incidente.**
Si decimas la línea, el vértice donde ocurrió un incidente puede desaparecer, desalineando el pin del trazo. Acción: simplificar solo la **línea base**, nunca los vértices de incidente; o no simplificar (un viaje de demo difícilmente pasa 2000 puntos). Bajo riesgo.

**B6 — Long-press 1.5s sobre el wordmark (§7) para cambiar de modo no tiene affordance visible — un juez que tome el celular no lo descubrirá, y podría dispararse accidentalmente.**
Es deliberadamente oculto (bien para la paradoja) pero §7 también dice "siempre deja un link visible para cambiar". Contradicción menor: ¿oculto (long-press) o visible (link)? Acción: en `/conductor` activo, oculto; en splash/pre-viaje, link visible. Aclarar cuándo aplica cada uno.

**B7 — La afirmación "Recharts hereda los hex por CSS-vars" (§4.4) es parcialmente optimista.**
El wrapper `Chart` de shadcn pasa colores vía `ChartConfig` (props), no todo por CSS-vars; algunos elementos (tooltips, ejes) sí toman vars, otros necesitan props explícitas. Acción: esperar ~medio día de fricción de theming en charts, no "sin pelear con el styling default". Expectativa, no bug.

**B8 — Idioma del código de estado: el spec mezcla `Alert`/`Drowsy`/`Distracted` (modelo, inglés) con labels español. El "harsh_event" e "incident" también en inglés.**
Consistencia: confirmar que los **strings del API** son inglés (`Alert`, `harsh_event`) y solo la **capa de presentación** traduce. Si el modelo o el backend emiten español en algún lado, habrá mismatches silenciosos en la normalización. Centralizar en un solo `STATE_LABELS` y `INCIDENT_LABELS`.

---

### Top 5 que arreglaría antes de escribir una línea de código
1. **A2 + A1**: re-derivar y fijar los tonos que *realmente* se pintan en el conductor, y re-correr el pase CVD/contraste sobre ESOS (no sobre los base). Los números del spec están mal y el trío pintado reprueba su propia prueba.
2. **A5 + A6**: construir el mock determinista (`?demo=1`) como cliente de API día 1, y validar el contrato real contra el EC2 vivo — el backend no existe en el repo.
3. **A7**: presupuestar y testear la histéresis como bloque propio con ruido inyectado.
4. **A3**: resolver la colisión semántica rojo-distracción vs rojo-peligro (reescribir la regla o separar tonos).
5. **M1 + M5 + M6**: recortar el scope "overdone" (logo que cambia de forma, modo claro del manager, scrubbing bidireccional como must-have) para honrar "estable en 2 semanas, don't overdo it".
