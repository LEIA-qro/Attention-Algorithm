import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ChevronLeft, ChevronRight, Inbox, Trash2 } from "lucide-react";
import { api, DEMO } from "@/lib/api";
import { STATE_COLOR } from "@/lib/stateColors";
import { SettingsButton } from "@/components/SettingsButton";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { pick, useLang, type Lang } from "@/lib/i18n";
import type { SessionListItem } from "@/lib/types";

function fmt(ts: string, lang: Lang) {
  return new Date(ts).toLocaleString(lang === "en" ? "en-US" : "es-MX", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function duration(s: SessionListItem, lang: Lang): string {
  const t = pick(lang);
  if (!s.ended_at) return t("En curso", "In progress");
  const min = Math.round((Date.parse(s.ended_at) - Date.parse(s.started_at)) / 60000);
  if (min < 1) return t("< 1 min", "< 1 min");
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)}h ${min % 60}m`;
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-hairline bg-panel p-5">
      <div
        className="text-3xl font-bold tabular-nums text-ink-high"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
      <div className="mt-1 text-sm text-ink-mid">{label}</div>
    </div>
  );
}

function RowsSkeleton() {
  return (
    <div className="animate-pulse overflow-hidden rounded-xl border border-hairline" aria-hidden="true">
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="flex items-center justify-between border-b border-hairline bg-panel px-5 py-4 last:border-b-0"
        >
          <div className="space-y-2">
            <div className="h-4 w-40 rounded bg-raised" />
            <div className="h-3 w-28 rounded bg-raised" />
          </div>
          <div className="h-6 w-24 rounded-full bg-raised" />
        </div>
      ))}
    </div>
  );
}

export default function Manager() {
  const lang = useLang();
  const t = pick(lang);
  const suffix = DEMO ? "?demo=1" : "";
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["sessions"],
    queryFn: api.listSessions,
  });

  const sessions = data?.sessions ?? [];
  const active = sessions.filter((s) => s.status === "active").length;
  const incidents = sessions.reduce((acc, s) => acc + s.incident_count, 0);

  const queryClient = useQueryClient();
  const [pending, setPending] = useState<SessionListItem | null>(null);
  const del = useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: () => {
      setPending(null);
      void queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  return (
    <div className="mx-auto max-w-5xl p-6">
      <div className="mb-4 flex items-center justify-between">
        <Link
          to="/"
          className="flex w-fit items-center gap-1 text-ink-low transition-colors hover:text-ink-mid"
        >
          <ChevronLeft size={18} /> {t("inicio", "home")}
        </Link>
        <SettingsButton />
      </div>
      <h1 className="mb-6 font-sans text-2xl font-bold text-ink-high">{t("Flota", "Fleet")}</h1>

      <div className="mb-8 grid grid-cols-3 gap-4">
        <Stat label={t("Viajes", "Trips")} value={sessions.length} />
        <Stat label={t("Activos ahora", "Active now")} value={active} accent={STATE_COLOR.Alert} />
        <Stat
          label={t("Incidentes totales", "Total incidents")}
          value={incidents}
          accent={incidents > 0 ? STATE_COLOR.Drowsy : undefined}
        />
      </div>

      <h2 className="mb-3 font-sans text-lg font-semibold text-ink-high">{t("Viajes", "Trips")}</h2>

      {isLoading && <RowsSkeleton />}

      {isError && (
        <div className="rounded-xl border border-hairline bg-panel p-8 text-center">
          <p className="text-ink-high">
            {t("No se pudieron cargar los viajes.", "Couldn't load trips.")}
          </p>
          <p className="mt-1 text-sm text-ink-low">
            {t("Verifica que la API esté disponible.", "Check that the API is available.")}
          </p>
          <button
            onClick={() => refetch()}
            className="mt-4 rounded-full bg-brand px-6 py-2 font-semibold text-void transition-transform hover:scale-[1.03]"
          >
            {t("Reintentar", "Retry")}
          </button>
        </div>
      )}

      {!isLoading && !isError && sessions.length === 0 && (
        <div className="flex flex-col items-center gap-3 rounded-xl border border-hairline bg-panel px-6 py-12 text-center">
          <Inbox size={32} className="text-ink-low" />
          <p className="font-medium text-ink-high">
            {t("Aún no hay viajes registrados", "No trips recorded yet")}
          </p>
          <p className="text-sm text-ink-mid">
            {t(
              "Inicia un viaje desde la vista del conductor para verlo aquí.",
              "Start a trip from the driver view to see it here.",
            )}
          </p>
        </div>
      )}

      {!isLoading && !isError && sessions.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-hairline">
          {sessions.map((s) => {
            const isActive = s.status === "active";
            return (
              <Link
                key={s.id}
                to={`/manager/viajes/${s.id}${suffix}`}
                onMouseEnter={() => import("./TripDetail")}
                className="flex items-center justify-between gap-4 border-b border-hairline bg-panel px-5 py-4 last:border-b-0 transition-colors hover:bg-raised"
              >
                <div className="min-w-0">
                  <div className="truncate font-medium text-ink-high">{s.id}</div>
                  <div className="text-sm text-ink-low">
                    {fmt(s.started_at, lang)} · {duration(s, lang)}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-6">
                  {isActive ? (
                    <span className="flex items-center gap-1.5 rounded-full bg-raised px-3 py-1 text-xs font-semibold text-state-alert">
                      <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-state-alert" />
                      {t("ACTIVO", "ACTIVE")}
                    </span>
                  ) : (
                    <span className="rounded-full bg-raised px-3 py-1 text-xs font-semibold text-ink-mid">
                      {t("FINALIZADO", "ENDED")}
                    </span>
                  )}
                  <span
                    className="tabular-nums text-sm text-ink-mid"
                    style={s.incident_count > 0 ? { color: STATE_COLOR.Drowsy } : undefined}
                  >
                    {s.incident_count}{" "}
                    {s.incident_count === 1 ? t("incidente", "incident") : t("incidentes", "incidents")}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setPending(s);
                    }}
                    aria-label={t("Eliminar viaje", "Delete trip")}
                    className="grid h-8 w-8 place-items-center rounded-full text-ink-low transition-colors hover:bg-raised hover:text-state-distracted"
                  >
                    <Trash2 size={16} />
                  </button>
                  <ChevronRight size={18} className="text-ink-low" />
                </div>
              </Link>
            );
          })}
        </div>
      )}

      <ConfirmDialog
        open={pending !== null}
        title={t("Eliminar viaje", "Delete trip")}
        message={t(
          "Se eliminará el viaje y toda su información: estados, incidentes, ruta y archivos (fotos y videos). Esta acción no se puede deshacer.",
          "The trip and all its data will be deleted: states, incidents, route and files (photos and videos). This cannot be undone.",
        )}
        confirmLabel={t("Eliminar", "Delete")}
        cancelLabel={t("Cancelar", "Cancel")}
        busy={del.isPending}
        onConfirm={() => pending && del.mutate(pending.id)}
        onCancel={() => setPending(null)}
      />
    </div>
  );
}
