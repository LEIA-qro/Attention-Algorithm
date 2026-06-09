import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ChevronLeft, ShieldCheck, Camera, X, MapPin, Play, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { TripMap } from "@/components/TripMap";
import { TripRibbon } from "@/components/TripRibbon";
import { SpeedChart } from "@/components/SpeedChart";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { NO_SIGNAL, STATE_COLOR, stateLabel, eventLabel } from "@/lib/stateColors";
import { pick, useLang, type Lang } from "@/lib/i18n";
import type { Incident, Session, StateSample } from "@/lib/types";

function fmt(ts: string, lang: Lang) {
  return new Date(ts).toLocaleString(lang === "en" ? "en-US" : "es-MX", {
    timeStyle: "short",
    dateStyle: "short",
  });
}
function fmtTime(ts: string, lang: Lang) {
  return new Date(ts).toLocaleTimeString(lang === "en" ? "en-US" : "es-MX", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function duration(s: Session, lang: Lang): string {
  const t = pick(lang);
  if (!s.ended_at) return t("En curso", "In progress");
  const min = Math.round((Date.parse(s.ended_at) - Date.parse(s.started_at)) / 60000);
  if (min < 1) return t("< 1 min", "< 1 min");
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)}h ${min % 60}m`;
}

// % del viaje fuera del estado "Alerta" (índice de riesgo). Robusto a estados
// desconocidos: cuentan como no-alerta.
function riskIndex(states: StateSample[]): number | null {
  if (states.length === 0) return null;
  const nonAlert = states.filter((s) => s.state !== "Alert").length;
  return Math.round((nonAlert / states.length) * 100);
}

function colorFor(state: Incident["state"]): string {
  return STATE_COLOR[state] ?? NO_SIGNAL;
}

function Chip({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-hairline bg-panel px-4 py-3">
      <div className="text-lg font-bold tabular-nums text-ink-high">{value}</div>
      <div className="text-xs uppercase tracking-widest text-ink-low">{label}</div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-6" aria-hidden="true">
      <div className="h-5 w-40 rounded bg-raised" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-[68px] rounded-xl bg-raised" />
        ))}
      </div>
      <div className="h-[360px] rounded-xl bg-raised" />
      <div className="h-24 rounded-xl bg-raised" />
      <div className="h-[220px] rounded-xl bg-raised" />
      <div className="space-y-2">
        <div className="h-16 rounded-xl bg-raised" />
        <div className="h-16 rounded-xl bg-raised" />
      </div>
    </div>
  );
}

export default function TripDetail() {
  const { id = "" } = useParams();
  const lang = useLang();
  const t = pick(lang);
  const [lightbox, setLightbox] = useState<Incident | null>(null);
  const [focusedIncidentId, setFocusedIncidentId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const mapWrapRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ["session", id],
    queryFn: () => api.getSession(id),
  });

  const del = useMutation({
    mutationFn: () => api.deleteSession(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["sessions"] });
      navigate("/manager");
    },
  });

  const isActive = data?.session.status === "active";
  const risk = useMemo(() => (data ? riskIndex(data.states) : null), [data]);
  const incidents = data?.incidents ?? [];

  // Focus an incident on the map: fly + open its popup (handled in TripMap) and
  // scroll the map into view. Re-set even if already focused so a repeat click
  // re-triggers the flyTo.
  function focusOnMap(inc: Incident) {
    if (inc.lat == null || inc.lng == null) return;
    setFocusedIncidentId(null);
    requestAnimationFrame(() => setFocusedIncidentId(inc.id));
    mapWrapRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <div className="mx-auto max-w-5xl p-6">
      <Link
        to="/manager"
        className="mb-4 flex w-fit items-center gap-1 text-ink-low transition-colors hover:text-ink-mid"
      >
        <ChevronLeft size={18} /> {t("viajes", "trips")}
      </Link>

      {isLoading && <Skeleton />}

      {isError && (
        <div className="rounded-xl border border-hairline bg-panel p-8 text-center">
          <p className="text-ink-high">{t("No se pudo cargar el viaje.", "Couldn't load the trip.")}</p>
          <p className="mt-1 text-sm text-ink-low">
            {t("Revisa la conexión con la API.", "Check the API connection.")}
          </p>
          <button
            onClick={() => refetch()}
            className="mt-4 rounded-full bg-brand px-6 py-2 font-semibold text-void transition-transform hover:scale-[1.03]"
          >
            {t("Reintentar", "Retry")}
          </button>
        </div>
      )}

      {data && (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <h1 className="font-mono text-lg text-ink-mid">{data.session.id}</h1>
            {isActive ? (
              <span className="flex items-center gap-1.5 rounded-full bg-raised px-2.5 py-1 text-xs font-semibold text-state-alert">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-state-alert" />
                {t("EN VIVO", "LIVE")}
              </span>
            ) : (
              <span className="rounded-full bg-raised px-2.5 py-1 text-xs font-semibold text-ink-mid">
                {t("FINALIZADO", "ENDED")}
              </span>
            )}
            <button
              type="button"
              onClick={() => setConfirmDelete(true)}
              className="ml-auto flex items-center gap-1.5 rounded-full border border-hairline px-3 py-1 text-xs font-semibold text-ink-mid transition-colors hover:border-state-distracted hover:text-state-distracted"
            >
              <Trash2 size={14} /> {t("Eliminar viaje", "Delete trip")}
            </button>
          </div>

          <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Chip label={t("Inicio", "Start")} value={fmt(data.session.started_at, lang)} />
            <Chip label={t("Duración", "Duration")} value={duration(data.session, lang)} />
            <Chip label={t("Incidentes", "Incidents")} value={incidents.length} />
            <Chip
              label={t("Índice de riesgo", "Risk index")}
              value={risk != null ? `${risk}%` : "—"}
            />
          </div>

          <div className="mb-5" ref={mapWrapRef}>
            <TripMap
              track={data.track}
              incidents={incidents}
              focusedIncidentId={focusedIncidentId}
            />
          </div>

          <div className="mb-8 rounded-xl border border-hairline bg-panel p-5">
            <TripRibbon states={data.states} incidents={incidents} />
          </div>

          <div className="mb-8 rounded-xl border border-hairline bg-panel p-5">
            <h2 className="mb-3 text-sm uppercase tracking-widest text-ink-low">
              {t("Velocidad", "Speed")}
            </h2>
            <SpeedChart track={data.track} />
          </div>

          <h2 className="mb-3 font-sans text-lg font-semibold text-ink-high">
            {t("Incidentes", "Incidents")} ({incidents.length})
          </h2>

          {incidents.length === 0 ? (
            <div className="flex flex-col items-center gap-3 rounded-xl border border-hairline bg-panel px-6 py-10 text-center">
              <span
                className="grid h-12 w-12 place-items-center rounded-full"
                style={{ background: "#0B6A66" }}
              >
                <ShieldCheck size={24} className="text-state-alert" />
              </span>
              <p className="font-medium text-ink-high">
                {t("Sin incidentes en este viaje", "No incidents on this trip")}
              </p>
              <p className="text-sm text-ink-mid">
                {t("Conducción 100% en alerta.", "100% alert driving.")}{" "}
                {isActive ? t("Sigue en marcha.", "Still in progress.") : t("Buen viaje.", "Safe trip.")}
              </p>
            </div>
          ) : (
            <div className="overflow-hidden rounded-xl border border-hairline">
              {incidents.map((inc) => {
                const located = inc.lat != null && inc.lng != null;
                const label = stateLabel(inc.state, lang);
                const evt = eventLabel(inc.event_type, lang);
                const showEvt = evt && evt !== label;
                return (
                  <div
                    key={inc.id}
                    className="flex flex-col gap-2 border-b border-hairline bg-panel px-4 py-3 last:border-b-0 transition-colors hover:bg-raised sm:flex-row sm:items-center sm:justify-between sm:gap-4 sm:px-5"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      {inc.snapshot_url ? (
                        <button
                          type="button"
                          onClick={() => setLightbox(inc)}
                          className="relative shrink-0 overflow-hidden rounded border border-hairline transition-transform hover:scale-[1.04] focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand"
                          aria-label={
                            inc.clip_url
                              ? t(`Reproducir video de ${label}`, `Play ${label} clip`)
                              : t(`Ampliar captura de ${label}`, `Enlarge ${label} snapshot`)
                          }
                        >
                          <img
                            src={inc.snapshot_url}
                            alt={t(`captura ${label}`, `${label} snapshot`)}
                            className="h-20 w-32 object-cover"
                            loading="lazy"
                          />
                          {inc.clip_url && (
                            <span className="absolute inset-0 grid place-items-center bg-black/30">
                              <span className="grid h-7 w-7 place-items-center rounded-full bg-black/60 text-white">
                                <Play size={14} className="translate-x-[1px]" fill="currentColor" />
                              </span>
                            </span>
                          )}
                        </button>
                      ) : inc.clip_url ? (
                        <button
                          type="button"
                          onClick={() => setLightbox(inc)}
                          className="grid h-20 w-32 shrink-0 place-items-center rounded border border-hairline bg-raised transition-transform hover:scale-[1.04] focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand"
                          aria-label={t(`Reproducir video de ${label}`, `Play ${label} clip`)}
                        >
                          <span className="grid h-7 w-7 place-items-center rounded-full bg-black/60 text-white">
                            <Play size={14} className="translate-x-[1px]" fill="currentColor" />
                          </span>
                        </button>
                      ) : (
                        <span
                          className="grid h-20 w-32 shrink-0 place-items-center rounded border border-hairline bg-raised"
                          title={t("Sin captura", "No snapshot")}
                        >
                          <Camera size={18} className="text-ink-low" />
                        </span>
                      )}
                      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
                        <span
                          className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                          style={{ background: colorFor(inc.state) }}
                          aria-hidden="true"
                        />
                        <span className="font-medium text-ink-high">{label}</span>
                        {inc.confirmed && (
                          <span
                            className="rounded px-2 py-0.5 text-xs font-semibold"
                            style={{ background: `${colorFor(inc.state)}22`, color: colorFor(inc.state) }}
                          >
                            {t("confirmado", "confirmed")}
                          </span>
                        )}
                        {showEvt && (
                          <span className="rounded border border-hairline px-2 py-0.5 text-xs text-ink-mid">
                            {evt}
                          </span>
                        )}
                        {inc.harsh_event && (
                          <span className="rounded bg-raised px-2 py-0.5 text-xs text-ink-mid">
                            {t("frenado brusco", "harsh braking")}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-mid sm:shrink-0 sm:gap-5">
                      <span className="tabular-nums">
                        {inc.speed_kmh != null ? `${inc.speed_kmh.toFixed(1)} km/h` : "—"}
                      </span>
                      <span className="tabular-nums text-ink-low">{fmt(inc.ts, lang)}</span>
                      {located && (
                        <button
                          type="button"
                          onClick={() => focusOnMap(inc)}
                          className="flex items-center gap-1.5 rounded-full border border-hairline px-3 py-1 text-xs font-semibold text-ink-mid transition-colors hover:bg-raised hover:text-ink-high focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand"
                          aria-label={t(`Ver ${label} en el mapa`, `View ${label} on map`)}
                        >
                          <MapPin size={13} />
                          <span className="hidden sm:inline">{t("ver en mapa", "view on map")}</span>
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {isFetching && !isLoading && (
            <p className="mt-3 text-xs text-ink-low">{t("Actualizando…", "Updating…")}</p>
          )}
        </>
      )}

      {/* Lightbox: "frame recuperado" con velocidad + coordenada estampadas. */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-6"
          onClick={() => setLightbox(null)}
          role="dialog"
          aria-modal="true"
          aria-label={t(
            `Captura de incidente: ${stateLabel(lightbox.state, lang)}`,
            `Incident snapshot: ${stateLabel(lightbox.state, lang)}`,
          )}
        >
          <div
            className="relative w-full max-w-lg overflow-hidden rounded-xl border border-hairline bg-panel"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setLightbox(null)}
              className="absolute right-2 top-2 z-10 grid h-8 w-8 place-items-center rounded-full bg-void/70 text-ink-mid transition-colors hover:text-ink-high"
              aria-label={t("Cerrar", "Close")}
            >
              <X size={18} />
            </button>
            {lightbox.clip_url ? (
              <video
                src={lightbox.clip_url}
                poster={lightbox.snapshot_url ?? undefined}
                controls
                autoPlay
                loop
                muted
                playsInline
                className="max-h-[60vh] w-full bg-void object-contain"
              />
            ) : (
              <img
                src={lightbox.snapshot_url ?? ""}
                alt={t(`captura ${stateLabel(lightbox.state, lang)}`, `${stateLabel(lightbox.state, lang)} snapshot`)}
                className="max-h-[60vh] w-full object-contain bg-void"
              />
            )}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-hairline px-4 py-3 font-mono text-xs text-ink-mid">
              <span
                className="flex items-center gap-1.5 font-semibold"
                style={{ color: colorFor(lightbox.state) }}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: colorFor(lightbox.state) }}
                />
                {stateLabel(lightbox.state, lang)}
              </span>
              {(() => {
                const evt = eventLabel(lightbox.event_type, lang);
                return evt && evt !== stateLabel(lightbox.state, lang) ? (
                  <span className="rounded border border-hairline px-1.5 py-0.5 not-italic">{evt}</span>
                ) : null;
              })()}
              <span className="tabular-nums">{fmtTime(lightbox.ts, lang)}</span>
              <span className="tabular-nums">
                {lightbox.speed_kmh != null ? `${lightbox.speed_kmh.toFixed(1)} km/h` : "— km/h"}
              </span>
              {lightbox.lat != null && lightbox.lng != null && (
                <span className="tabular-nums text-ink-low">
                  {lightbox.lat.toFixed(4)}, {lightbox.lng.toFixed(4)}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={confirmDelete}
        title={t("Eliminar viaje", "Delete trip")}
        message={t(
          "Se eliminará el viaje y toda su información: estados, incidentes, ruta y archivos (fotos y videos). Esta acción no se puede deshacer.",
          "The trip and all its data will be deleted: states, incidents, route and files (photos and videos). This cannot be undone.",
        )}
        confirmLabel={t("Eliminar", "Delete")}
        cancelLabel={t("Cancelar", "Cancel")}
        busy={del.isPending}
        onConfirm={() => del.mutate()}
        onCancel={() => setConfirmDelete(false)}
      />
    </div>
  );
}
