import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapPinned, MapPin } from "lucide-react";
import type { Incident, TrackPoint } from "@/lib/types";
import { NO_SIGNAL, STATE_COLOR, stateLabel, eventLabel } from "@/lib/stateColors";
import { useTheme, type Theme, pick, useLang } from "@/lib/prefs";

const STYLE_URL: Record<Theme, string> = {
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
};

// Hex acromatico por tema; el unico color en el mapa es el marcador de incidente.
const THEME_HEX: Record<Theme, { route: string; casing: string; halo: string; od: string }> = {
  dark: { route: "#F2F4F7", casing: "#0B0C0F", halo: "rgba(11,12,15,.85)", od: "#F2F4F7" },
  light: { route: "#16181D", casing: "#FFFFFF", halo: "rgba(255,255,255,.92)", od: "#16181D" },
};

function colorFor(state: Incident["state"]): string {
  return STATE_COLOR[state] ?? NO_SIGNAL;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&quot;",
  );
}

// Marcadores de origen y destino, distintos de los puntos de incidente.
function originEl(ink: string, halo: string): HTMLDivElement {
  const el = document.createElement("div");
  el.style.cssText =
    `width:16px;height:16px;border-radius:50%;background:transparent;` +
    `border:3px solid ${ink};box-shadow:0 0 0 3px ${halo};box-sizing:border-box;`;
  return el;
}
function destEl(ink: string, halo: string): HTMLDivElement {
  const el = document.createElement("div");
  // Gota: cuadrado redondeado rotado 45 con una punta hacia abajo.
  el.style.cssText =
    `width:16px;height:16px;background:${ink};` +
    `border-radius:50% 50% 50% 0;transform:rotate(45deg);` +
    `box-shadow:0 0 0 3px ${halo};box-sizing:border-box;`;
  return el;
}

export function TripMap({
  track,
  incidents,
  focusedIncidentId,
}: {
  track: TrackPoint[];
  incidents: Incident[];
  focusedIncidentId?: string | null;
}) {
  const theme = useTheme();
  const lang = useLang();
  const t = pick(lang);
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<maplibregl.Marker[]>([]);
  // Marcadores de incidente por id para que el zoom abra el popup correcto.
  const incidentMarkersRef = useRef<Map<string, maplibregl.Marker>>(new Map());
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);

  // Crea el mapa; se recrea al cambiar de tema para intercambiar estilo y ruta.
  useEffect(() => {
    if (!container.current) return;
    setReady(false);

    const ink = THEME_HEX[theme];
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: container.current,
        style: STYLE_URL[theme],
        center: [-100.3899, 20.5888],
        zoom: 11,
        attributionControl: false,
      });
    } catch {
      setFailed(true);
      return;
    }
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    const onLoad = () => {
      map.addSource("route", {
        type: "geojson",
        // lineMetrics permite el gradiente de progreso a lo largo de la linea.
        lineMetrics: true,
        data: { type: "FeatureCollection", features: [] },
      });
      // Contorno ancho de baja opacidad que da un glow suave.
      map.addLayer({
        id: "route-casing",
        type: "line",
        source: "route",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": ink.casing, "line-width": 9, "line-opacity": 0.55, "line-blur": 1 },
      });
      // Linea principal acromatica con fade-in segun el progreso.
      map.addLayer({
        id: "route",
        type: "line",
        source: "route",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-width": 4,
          "line-gradient": [
            "interpolate",
            ["linear"],
            ["line-progress"],
            0,
            ["to-color", ink.route + "55"],
            0.5,
            ["to-color", ink.route + "cc"],
            1,
            ["to-color", ink.route],
          ],
        },
      });
      setReady(true);
    };

    // Si el estilo/CDN falla, muestra un fallback visible en vez de un lienzo vacio.
    const onError = (e: { error?: { message?: string } }) => {
      if (!map.isStyleLoaded()) setFailed(true);
      console.warn("MapLibre error:", e?.error?.message ?? e);
    };

    map.on("load", onLoad);
    map.on("error", onError);

    return () => {
      for (const m of markersRef.current) m.remove();
      markersRef.current = [];
      incidentMarkersRef.current.clear();
      map.off("load", onLoad);
      map.off("error", onError);
      map.remove();
      mapRef.current = null;
    };
  }, [theme]);

  // Actualiza geometria, marcadores y viewport al cambiar los datos o el idioma.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || failed) return;

    const ink = THEME_HEX[theme];
    const coords = track
      .filter((p) => Number.isFinite(p.lng) && Number.isFinite(p.lat))
      .map((p) => [p.lng, p.lat] as [number, number]);

    const src = map.getSource("route") as maplibregl.GeoJSONSource | undefined;
    src?.setData({
      type: "Feature",
      properties: {},
      geometry: { type: "LineString", coordinates: coords },
    });

    // Refresca todos los marcadores (extremos e incidentes).
    for (const m of markersRef.current) m.remove();
    markersRef.current = [];
    incidentMarkersRef.current.clear();

    // Extremos de origen y destino de la ruta.
    if (coords.length > 0) {
      const start = coords[0];
      const end = coords[coords.length - 1];
      const startMarker = new maplibregl.Marker({ element: originEl(ink.od, ink.halo) })
        .setLngLat(start)
        .setPopup(
          new maplibregl.Popup({ offset: 14, closeButton: false }).setHTML(
            `<div style="font-family:system-ui;font-size:12px;color:#16181D"><b>${escapeHtml(
              t("Inicio", "Start"),
            )}</b></div>`,
          ),
        )
        .addTo(map);
      markersRef.current.push(startMarker);
      // Solo dibuja el pin de destino si esta lejos del origen.
      const apart = Math.abs(end[0] - start[0]) > 1e-6 || Math.abs(end[1] - start[1]) > 1e-6;
      if (apart) {
        const endMarker = new maplibregl.Marker({
          element: destEl(ink.od, ink.halo),
          // Compensa la rotacion de 45 para anclar el popup sobre la punta.
          offset: [0, -6],
        })
          .setLngLat(end)
          .setPopup(
            new maplibregl.Popup({ offset: 14, closeButton: false }).setHTML(
              `<div style="font-family:system-ui;font-size:12px;color:#16181D"><b>${escapeHtml(
                t("Fin", "End"),
              )}</b></div>`,
            ),
          )
          .addTo(map);
        markersRef.current.push(endMarker);
      }
    }

    // Marcadores de incidente: el unico color del mapa.
    for (const inc of incidents) {
      if (inc.lat == null || inc.lng == null) continue;
      if (!Number.isFinite(inc.lng) || !Number.isFinite(inc.lat)) continue;
      const el = document.createElement("div");
      el.style.cssText = `width:14px;height:14px;border-radius:50%;background:${colorFor(
        inc.state,
      )};box-shadow:0 0 0 3px ${ink.halo};cursor:pointer;`;
      const speed = inc.speed_kmh != null ? `${inc.speed_kmh.toFixed(1)} km/h` : "—";
      const evt = eventLabel(inc.event_type, lang);
      const evtLine =
        evt && evt !== stateLabel(inc.state, lang)
          ? `${escapeHtml(evt)}<br/>`
          : "";
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([inc.lng, inc.lat])
        .setPopup(
          new maplibregl.Popup({ offset: 14, closeButton: false }).setHTML(
            `<div style="font-family:system-ui;font-size:12px;color:#16181D"><b>${escapeHtml(
              stateLabel(inc.state, lang),
            )}</b><br/>${evtLine}${escapeHtml(speed)}</div>`,
          ),
        )
        .addTo(map);
      markersRef.current.push(marker);
      incidentMarkersRef.current.set(inc.id, marker);
    }

    // Ajusta a la ruta; protege los casos degenerados (sin puntos o todos iguales).
    if (coords.length === 0) return;
    const lons = coords.map((c) => c[0]);
    const lats = coords.map((c) => c[1]);
    const minX = Math.min(...lons);
    const maxX = Math.max(...lons);
    const minY = Math.min(...lats);
    const maxY = Math.max(...lats);

    if (maxX - minX < 1e-6 && maxY - minY < 1e-6) {
      map.jumpTo({ center: [minX, minY], zoom: 14 });
    } else {
      map.fitBounds(
        [
          [minX, minY],
          [maxX, maxY],
        ],
        { padding: 56, duration: 0, maxZoom: 15 },
      );
    }
    // lang es dep a proposito para re-renderizar los popups en el idioma activo.
  }, [track, incidents, ready, failed, theme, lang, t]);

  // Al enfocar un incidente desde el padre, vuela a el y abre su popup.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || failed || !focusedIncidentId) return;
    const inc = incidents.find((i) => i.id === focusedIncidentId);
    if (!inc || inc.lat == null || inc.lng == null) return;
    if (!Number.isFinite(inc.lng) || !Number.isFinite(inc.lat)) return;

    map.flyTo({ center: [inc.lng, inc.lat], zoom: 15, duration: 900, essential: true });
    const marker = incidentMarkersRef.current.get(focusedIncidentId);
    const popup = marker?.getPopup();
    if (popup && marker && !popup.isOpen()) marker.togglePopup();
  }, [focusedIncidentId, incidents, ready, failed]);

  return (
    <div className="relative h-[360px] w-full overflow-hidden rounded-xl border border-hairline bg-raised">
      <div
        ref={container}
        role="img"
        aria-label={t(
          "Mapa de la ruta del viaje con marcadores de incidentes",
          "Trip route map with incident markers",
        )}
        className="h-full w-full"
      />

      {!ready && !failed && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center bg-raised">
          <div className="flex flex-col items-center gap-2 text-ink-low">
            <MapPinned size={28} className="animate-pulse" />
            <span className="text-xs">{t("Cargando mapa…", "Loading map…")}</span>
          </div>
        </div>
      )}

      {failed && (
        <div className="absolute inset-0 grid place-items-center bg-panel p-6 text-center">
          <div className="flex max-w-xs flex-col items-center gap-3">
            <MapPin size={32} className="text-ink-low" />
            <p className="text-sm text-ink-mid">
              {t("No se pudo cargar el mapa base.", "Couldn't load the base map.")}
            </p>
            <p className="text-xs text-ink-low">
              {t(
                "La ruta y los incidentes siguen disponibles en la lista y la línea de tiempo.",
                "The route and incidents are still available in the list and timeline.",
              )}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
