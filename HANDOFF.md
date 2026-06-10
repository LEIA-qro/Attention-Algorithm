# DMS — Handoff (2026-06-07)

Documento para retomar el proyecto DMS (Driver Monitoring System) en una sesión nueva.
Captura el estado actual + el trabajo pendiente que pidió Felipe. **PLANEADO, no ejecutado.**

---

## 0. Cómo retomar (leer primero)

- **Memoria del proyecto** (auto-carga): `~/.claude/projects/D--TEC-LEIA-Attention-Algorithm/memory/`
  - `project_cloud-integration.md` ← la más detallada (infra, fixes, gotchas, todo el historial).
  - `project_architecture.md`, `project_team-roles.md`.
- **App viva**: https://34-205-126-89.nip.io
- **Reglas duras**: español MX en UI; **cero atribución de IA** (commits/código/docs); no over-engineer; "premium pero estable, don't overdo it".
- Felipe = nube/integración. Diego/Santiago = modelo. Pasaye = hardware (RPi). Josué = presentación.

---

## 1. Estado actual (qué funciona, en vivo)

- **Backend** (EC2 `34.205.126.89`, us-east-1, cuenta edu): FastAPI + Postgres + S3 + Caddy(HTTPS), docker-compose en `/home/ubuntu/dms`. Endpoints: health, POST/GET sessions, current-session, **/sessions/{id}/state** (estado continuo), states, incidents, track, alerts, /uploads/presign, /devices(+heartbeat).
- **Frontend** (React+TS+Vite+Tailwind, `C:\Users\luis\dms-deploy\web`): SPA rutas `/` `/conductor` `/manager` `/manager/viajes/:id` `/sensores`. Tema light/dark acromático ("Instrumento", chrome sin color, color=solo estado), i18n es/en, SettingsButton (engrane: tema/idioma/sonido/vibración/link a sensores), pill "Cámara disponible", GPS real del celular (POST /track), snapshots con lightbox, mapa MapLibre (ruta acromática + O/D markers + click-incidente→flyTo), Trip Ribbon, gráfica de velocidad.
- **Edge / "RPi" = la laptop**: `C:\Users\luis\dms-deploy\tools\rpi_live.py` corre el **pipeline del equipo** (SurveillancePipeline de `09_surveillance_custom.py`: YOLO objetos + MediaPipe + LSTM + motor de eventos + clips locales) y le agrega la capa de nube (headless, pola current-session, postea state/incidents/fotos + heartbeat, auto-calibra al iniciar viaje, `--show` = ventana local con overlay).
- **Confirmado esta sesión**: el flujo end-to-end funciona (celular inicia viaje → laptop-RPi se engancha → estados/incidentes/fotos a la nube + detecta objetos incl. celular). El **fix de desync app↔local SÍ quedó** (la app lee `/sessions/{id}/state` continuo, no incidentes).

---

## 2. Infra, deploy y GOTCHAS (críticos)

- **SSH**: `ssh -i ~/.ssh/dms-key ubuntu@34.205.126.89` (llave ed25519 local; usuario `ubuntu`).
- **Deploy FRONTEND (in-place, NO romper el mount)**: `cd web && npm run build` → `scp dist/index.html ec2:/home/ubuntu/dms/web/dist/index.html` + `scp dist/assets/* ec2:.../web/dist/assets/`. Verificar el hash del JS servido. **Recargar el celular** para bajar el bundle.
- **Deploy BACKEND**: `scp api/main.py ec2:.../api/main.py` + `ssh ec2 "cd /home/ubuntu/dms && docker compose up -d --build api"`.
- **GOTCHAS**:
  - `npm install` truena por MSIX VFS → usar `npm install --cache-dir D:/pip-cache`/`D:/npm-cache`. Igual `pip install --cache-dir D:/pip-cache`.
  - `scp -r dist host:.../dist` cuando el dir existe → ANIDA `dist/dist`. Usar el método in-place (archivos sueltos) de arriba. Si haces `rm -rf` del dist montado → reiniciar caddy (`docker compose restart caddy`) o sirve 404 (inode borrado).
  - Reiniciar el container `api` RESETEA el registro in-memory de `/devices` (rpi_live lo repuebla en 3s).
- **Edge / modelo**:
  - Worktree del repo del modelo: `D:/TEC/LEIA/Attention-Algorithm-model` (branch `attention_algorithm`).
  - venv: `D:/TEC/LEIA/Attention-Algorithm-model/.venv-live` (Python 3.12) con: mediapipe onnxruntime opencv-python numpy pyyaml **ultralytics flask torch**.
  - Python default de la lap es 3.14 (MediaPipe no soporta) → SIEMPRE el venv 3.12.
  - `config/yolo_config.yaml` traía `device: "cuda"` → la lap no tiene NVIDIA → cambiado a `device: "cpu"` en el worktree (local, NO commitear).
  - **Cámara EXCLUSIVA**: corre `rpi_live.py` O el dashboard Flask del equipo (`scripts/10_web_dashboard.py`), no ambos.
- **Comandos**:
  - Edge a la nube + ventana: `cd D:/TEC/LEIA/Attention-Algorithm-model && .venv-live/Scripts/python.exe C:/Users/luis/dms-deploy/tools/rpi_live.py --repo D:/TEC/LEIA/Attention-Algorithm-model --show` (Ctrl+C o `q` para salir).
  - Dashboard del equipo (vista local + calibración, sin nube): `.venv-live/Scripts/python.exe scripts/10_web_dashboard.py --source 0 --selfie --port 8080` → http://localhost:8080.
  - Limpiar BD: `ssh ec2 "cd /home/ubuntu/dms && docker compose exec -T db psql -U dms -d dms -c 'TRUNCATE sessions, state_samples, incidents, track_points RESTART IDENTITY CASCADE;'"`.

---

## 3. TRABAJO PENDIENTE (lo que pidió Felipe) — priorizado

> **Avance 2026-06-07**: P5, P6, P7, P4, P2 y **P3 HECHOS y DEPLOYADOS**. Además: **Zona de pruebas** en vivo y **alarmas reforzadas**. Solo queda **P1 (latencia)**.
> - **P3 (NUBE)**: tabla `config` (umbrales por estado, editables desde la app), flag `confirmed` en incidentes, lógica en `post_state` que confirma un estado sostenido ≥ umbral (Drowsy/Distracted) y lo registra como incidente `confirmed=true`. `/state` devuelve `confirmed`. `GET/PUT /config`. Helpers `_run_start`/`_state_confirmed`/`_maybe_confirm` en `api/main.py`.
> - **Alarmas**: `audio.ts` con `startAlarm/stopAlarm` (square de dos tonos, gain alto, repetida sonido+vibración) que suena mientras el estado esté **confirmado**, hasta que el conductor reaccione. `useDriverState` expone `confirmed`; `Conductor.tsx` arranca/detiene la alarma. Badge "CONFIRMADO".
> - **Zona de pruebas**: toggle `testMode` en Configuración. `web/src/components/TestZone.tsx` (forzar estado postea cada 1s → sostiene → confirma+alarma; disparar eventos puntuales). Reusa `POST /states` y `/incidents`.
> - **Config UI**: steppers de segundos en SettingsButton (Somnolencia/Distracción).
> - **Marca**: el producto se llama **DAVE** (Driver Attention & Vision Evaluator). Integrado en `index.html` (title/desc), Landing (wordmark + subtítulo) y footer de Configuración. (Antes "DMS"; quedan refs internas en código/infra como `dms-deploy`, `/home/ubuntu/dms`, bucket — NO renombrar infra.)
> - **Easter egg**: toggle anidado en Configuración → Pruebas → "Voz de Crazy Dave" (solo visible con testMode on, persistido como `daveVoice`). Cuando está on, la alarma sostenida usa `web/public/crazy-dave.mp3` (servido en `/crazy-dave.mp3`) en loop en vez del beep. `audio.ts` lo desbloquea en `initAudio()` (gesto de Iniciar viaje) para esquivar autoplay.
> - **Fix clips (fps real)**: el pipeline del equipo en external-capture usaba `features.fps`=30 por default, pero la laptop procesa a ~9fps → clips a 30fps con duplicación de frames inconsistente (cada uno cubría distinto tiempo real, movimiento a saltos). FIX en `tools/rpi_live.py` → `build_pipeline` setea `dms_cfg["features"]["fps"]` al fps real, expuesto como `--fps` (default 9). Validado: clips ahora 9fps/73frames/~8s reales y consistentes. También mejora las ventanas temporales (PERCLOS/parpadeo) → estado menos errático.
> - **Eliminar viaje (cascada total)** [deployado]: `DELETE /sessions/{id}` borra en una transacción incidents+state_samples+track_points+session y los objetos S3 (snapshot_key + clip_key, vía `delete_objects`). Frontend: botón 🗑️ en lista del manager y header del detalle, con `ConfirmDialog` (componente nuevo) y mutación TanStack que invalida `["sessions"]` / navega a /manager. `api.deleteSession`. Validado end-to-end (404 + BD 0 + S3 borrado).
> - **DAVE corriendo en segundo plano**: el comando para la prueba local es `rpi_live.py --repo ... [--show] [--fps 9]`; log de prueba en `tools/rpi_live_test.log`. El `[diag]` cada 5s reporta fps del pipeline (~8) + RTT del post (~205ms, = RTT de red, server responde <10ms).

### P1 — Bajar la latencia de la conexión  [Acción 0 + 1 HECHAS — falta SSE (Acción 2)]

**MEDICIÓN (2026-06-07, 15 muestras c/u):** `/health` avg 224ms (1er req 558ms = TLS frío), `/current-session` (toca DB) avg 210ms ≈ igual → **DB/API <10ms, NO es el cuello**. Los ~200ms son **RTT de red** laptop→us-east-1. Cuello real = cadencia del edge + gap del polling + RTT. Bonus: el edge posteaba **bloqueante** en el loop del pipeline → bajar cadencia sin más habría reducido fps.

**Acción 1 HECHA (en `tools/rpi_live.py`, sin deploy — es el archivo local):**
- POSTs de estado + heartbeat ahora **no-bloqueantes** (`post_async`, hilo daemon) → no roban fps al pipeline.
- `--state-every` default 1.0 → **0.4s**; `poll_every` (current-session) 2.0 → **1.0s**.
- Polling del conductor ya estaba en **0.7s** (de la sesión anterior) + bypass de histéresis al confirmar.
- **Instrumentación en vivo**: línea `[diag]` cada 5s con fps del pipeline + RTT del post; fps también en la línea de estado. (Sirve para ver el cuello durante una corrida real.)
- Resultado esperado: staleness end-to-end de ~2.7s worst → ~1.3s worst (~0.8s avg). SSE lo bajaría a ~0.2-0.4s.

**Falta — Acción 2 (SSE, "la buena"):**
- **Acción 2 (la buena)**: **SSE** (Server-Sent Events) para empujar el estado en vez de pollear: endpoint `GET /sessions/{id}/stream` (text/event-stream) que emite el último state_sample; el conductor usa `EventSource`. Latencia <1s. Mantener polling como fallback. (FastAPI: `StreamingResponse` con generador.)
- **Medir primero**: añadir timestamps (postear→DB→servir) para ver dónde está el cuello (¿el t3.small? ¿el fps del pipeline? ¿el polling?). Archivos: `tools/rpi_live.py`, `api/main.py`, `web/src/lib/useDriverState.ts`, `web/src/lib/api.ts`.

### P2 — Consumir VIDEOS (clips) en vez de solo foto  [HECHO ✓ deployado]
> Correlación clip↔incidente por `event_type` (el nombre del clip trae el tipo; la telemetría del equipo solo expone event_type, no el trigger dict). El clip cierra ~5s post-evento → watcher del directorio con check de estabilidad (mtime >2s). Incidente sin clip = solo foto (fallback). NOTA: no hay video en modo demo (mock sin `clip_url`) — validar con rpi_live real.
"Un evento no se evalúa bien con una sola foto." El pipeline del equipo **ya graba clips ±5s MP4** en `output/clips/*.mp4` (ClipWriter, con dedup + HUD).
- **Backend**: `_CONTENT_TYPES` en presign → agregar `mp4`/`webm` (video/mp4). Incidente: columna/campo `clip_key`; `get_session` resuelve `clip_key`→presigned GET (como snapshot_url). Endpoint para **asociar el clip después** (el clip se cierra ~5s post-evento): p.ej. `POST /sessions/{id}/incidents/{inc_id}/clip {key}` o `PATCH`.
- **Edge (rpi_live)**: tras el incidente, cuando el ClipWriter cierra el MP4 (vigilar `output/clips/` por archivo nuevo, o engancharse al event_logger), subirlo a S3 (presign mp4 → PUT) y asociarlo al incidente. Reto: correlacionar el archivo del clip con el incidente (por timestamp/nombre — el nombre trae `YYYYMMDD_HHMMSS_ms_<evento>_<pct>.mp4`).
- **Frontend**: el lightbox de `TripDetail` muestra `<video controls>` cuando hay `clip_url`; foto como poster/fallback. Archivos: `api/main.py`, `tools/rpi_live.py`, `web/src/pages/TripDetail.tsx`, `web/src/lib/types.ts`.

### P3 — Capa de filtro / thresholds configurables ("X seg distraído = distraído real")  [HECHO ✓ deployado — NUBE]
Felipe: no solo detectar eventos y mostrarlos; agregar una capa que, **tras X tiempo en un estado, lo confirme como real** y lo registre como tal, **configurable por estado** (drowsy, distracted, eyes_off, etc.).
- **DECISIÓN DE DISEÑO**: ¿la capa vive en el **edge** (rpi_live, sobre el estado continuo — simple: contador de racha por estado, al cruzar threshold postea incidente `confirmed=true`) o en la **nube** (API procesa el stream de state_samples)? Felipe dijo "por parte de la api y la app" → favorece nube, pero el edge es más simple y ya tiene el stream. **Recomendación**: lógica en rpi_live, con thresholds **leídos de la API** (`GET /config` editable) para que sea configurable sin tocar el edge.
- **Config**: thresholds por estado (default sugerido: drowsy=1.5s, distracted=2.5s, eyes_off=2s, phone=1s) — guardar en una tabla `config` o un JSON; endpoint `GET/PUT /config`. La app/dash muestran/filtran por `confirmed`.
- **Nota**: el motor de eventos del equipo (`event_engine.py`, `sustain_seconds` en `yolo_config.yaml`) YA hace sustain/cooldown — evaluar si se expone esa config en vez de duplicar. Pero Felipe quiere una capa propia configurable desde la nube/app. Archivos: `tools/rpi_live.py`, `api/main.py` (config + flag confirmed en incidente), frontend (filtro/etiqueta confirmed).

### P4 — Detección de objetos en la app (campo `event_type`)  [HECHO ✓ deployado]
Hoy la nube mapea TODOS los triggers (phone/food/danger/eyes_off/distracted) a "Distracted" → se pierde el tipo. (`EVENT_TO_STATE` en rpi_live.)
- **Backend**: agregar campo `event_type` (TEXT) al incidente (columna + `IncidentIn` + select en get_session/alerts).
- **Edge**: rpi_live ya tiene el `evt` (event_type real) en el loop de triggers → mandarlo en `event_type`.
- **Frontend**: mostrar el tipo en lista de incidentes / lightbox / popup del mapa con icono+label i18n ("Celular detectado", "Comida", "Objeto peligroso", "Ojos fuera del camino", "Somnolencia", "Distracción"). Archivos: `api/main.py`, `tools/rpi_live.py`, `web/src/lib/types.ts`, `web/src/lib/stateColors.ts` (labels de evento), `TripDetail.tsx`, `TripMap.tsx`, `TripRibbon.tsx`.

### P5 — Light mode en la vista de driver  [HECHO ✓ deployado]
El Conductor está **dark-locked** a propósito: `<div className="theme-dark ...">` en `Conductor.tsx` (regla de glare). Por eso el toggle no cambia el conductor ("no se ve light mode"). El botón NO está mal wired — el conductor ignora el tema global por el scope `theme-dark`.
- **Acción**: quitar (o condicionar) el `theme-dark` del root del Conductor para que siga el tema global. Felipe quiere ver light mode ahí → quitar el lock. (Trade-off: perdemos "siempre oscuro por glare"; es su decisión.) Archivo: `web/src/pages/Conductor.tsx` (línea del div raíz).

### P6 — Speedometer en la vista de driver (estilo Waze/Maps)  [HECHO ✓ deployado]
> Componente `web/src/components/Speedometer.tsx` (gauge acromático). `Conductor.tsx` captura `speed` del watchPosition y lo muestra abajo-izquierda durante el viaje. Muestra "—" sin fix GPS (lap parada).
Mostrar la velocidad actual grande en el conductor.
- La velocidad ya está disponible: el `useEffect` de GPS en `Conductor.tsx` lee `p.coords.speed` (m/s) — hoy solo la postea, no la guarda en estado. Agregar `const [speed, setSpeed] = useState<number|null>(null)` y setearlo en el watchPosition (`speed*3.6`). Componente `Speedometer` (número grande + "km/h", opcional arco/gauge). Mostrar `—` si speed es null (parado/sin fix). Archivo: `web/src/pages/Conductor.tsx` (+ posible `web/src/components/Speedometer.tsx`).

### P7 — Velocidad con decimales (no trimmed) en el manager dash  [HECHO ✓ deployado]
> `.toFixed(1)` en SpeedChart (serie sin redondear + tooltip), TripDetail (lista + lightbox) y TripMap (popup).
Hoy se redondea (`Math.round`) en `SpeedChart.tsx` (data.speed) y en `TripDetail.tsx` (lista de incidentes `${inc.speed_kmh} km/h`).
- **Acción**: mostrar 1 decimal (`.toFixed(1)`) en la gráfica y en las filas de incidente / KPIs. Archivos: `web/src/components/SpeedChart.tsx`, `web/src/pages/TripDetail.tsx`.

---

## 4. Decisiones pendientes (preguntar a Felipe / al equipo)
- **P3**: ¿capa de confirmación en el edge o en la nube? ¿usar la config del motor del equipo o una propia?
- **P5**: ¿el conductor sigue el tema global del todo, o dark por default con override?
- **Dashboard del equipo vs nuestra nube**: coexisten (Flask local en la RPi + nube remota). Alinear cuál es el "oficial" del expo.
- **event_type → estado**: ¿los eventos de objeto (phone/food) cuentan como "Distracted" para el estado del conductor, o son una categoría aparte?

## 5. Limitaciones conocidas
- **Modelo**: la confianza siempre sale `0.90` porque el pipeline del equipo usa un **override heurístico de pose de cabeza** que pisa las probs del LSTM. El estado es **muy pegajoso a la pose de cabeza** (se queda en Distracted mientras la cabeza esté fuera del neutral calibrado → recalibrar/ajustar histéresis). La clase Distracted del LSTM es la más débil.
- **GPS estático**: en stand quieto el GPS del celular da ~1 punto (sin línea de ruta) — es lo real. `rpi_live --fake-gps` da ruta sintética si se quiere para el demo.
- **t3.small**: aguanta la demo (2-5 users) pero es chico; el pipeline corre en la lap, no en el EC2.

## 6. Orden sugerido de ataque
P5 + P7 (rápidos, visibles) → P4 (event_type, mejora visible) → P2 (videos) → P1 (latencia, medir+SSE) → P3 (capa de confirmación, requiere decisión). Verificar con `tsc --noEmit` + `npm run build` + deploy in-place tras cada uno; smoke-test endpoints nuevos.
