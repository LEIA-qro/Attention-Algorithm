import { useMemo } from "react";
import type { DriverState, Incident, StateSample } from "@/lib/types";
import { NO_SIGNAL, STATE_COLOR, stateLabel, eventLabel } from "@/lib/stateColors";
import { useTheme, pick, useLang } from "@/lib/prefs";

function colorFor(state: DriverState): string {
  return STATE_COLOR[state] ?? NO_SIGNAL;
}

interface Run {
  state: DriverState;
  start: number;
  end: number;
}

export function TripRibbon({
  states,
  incidents = [],
}: {
  states: StateSample[];
  incidents?: Incident[];
}) {
  const theme = useTheme();
  const lang = useLang();
  const t = pick(lang);
  const locale = lang === "en" ? "en-US" : "es-MX";
  // Tinta acromatica para los ticks de incidente y su halo, por tema.
  const tickInk = theme === "dark" ? "#F2F4F7" : "#16181D";
  const tickHalo = theme === "dark" ? "rgba(11,12,15,.8)" : "rgba(255,255,255,.9)";

  const { runs, t0, span } = useMemo(() => {
    if (states.length === 0) return { runs: [] as Run[], t0: 0, span: 1 };
    const sorted = [...states].sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
    const start0 = Date.parse(sorted[0].ts);
    const end0 = Date.parse(sorted[sorted.length - 1].ts);
    const total = Math.max(end0 - start0, 1);

    // Agrupa muestras consecutivas del mismo estado en tramos.
    const out: Run[] = [];
    for (let i = 0; i < sorted.length; i++) {
      const t = Date.parse(sorted[i].ts);
      const next =
        i < sorted.length - 1 ? Date.parse(sorted[i + 1].ts) : end0;
      const last = out[out.length - 1];
      if (last && last.state === sorted[i].state) {
        last.end = next;
      } else {
        out.push({ state: sorted[i].state, start: t, end: next });
      }
    }
    return { runs: out, t0: start0, span: total };
  }, [states]);

  const legend = (
    <span className="flex flex-wrap gap-3">
      {(Object.keys(STATE_COLOR) as DriverState[]).map((s) => (
        <span key={s} className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: STATE_COLOR[s] }}
          />
          {stateLabel(s, lang)}
        </span>
      ))}
    </span>
  );

  if (runs.length === 0) {
    return (
      <div>
        <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-widest text-ink-low">
          <span>{t("línea de tiempo del viaje", "trip timeline")}</span>
          {legend}
        </div>
        <div className="grid h-3 w-full place-items-center rounded-full bg-raised">
          <span className="text-[10px] text-ink-low">
            {t("Sin datos de estado", "No state data")}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-widest text-ink-low">
        <span>{t("línea de tiempo del viaje", "trip timeline")}</span>
        {legend}
      </div>

      <div className="relative">
        <div
          className="flex h-3 w-full overflow-hidden rounded-full bg-raised"
          role="img"
          aria-label="Distribución de estados del conductor a lo largo del viaje"
        >
          {runs.map((r, i) => {
            const pct = ((r.end - r.start) / span) * 100;
            if (pct <= 0) return null;
            const startTime = new Date(r.start).toLocaleTimeString(locale, {
              hour: "2-digit",
              minute: "2-digit",
            });
            // Textura diagonal en Somnolencia para distinguir teal y ambar bajo glare.
            const stripe =
              r.state === "Drowsy"
                ? {
                    backgroundImage:
                      "repeating-linear-gradient(45deg, rgba(0,0,0,0.28) 0 3px, transparent 3px 7px)",
                  }
                : null;
            return (
              <div
                key={i}
                title={`${stateLabel(r.state, lang)} · ${t("desde", "from")} ${startTime}`}
                style={{ width: `${pct}%`, background: colorFor(r.state), ...stripe }}
              />
            );
          })}
        </div>

        {/* Ticks de incidente alineados al mismo eje de tiempo. */}
        {incidents.map((inc) => {
          const pos = ((Date.parse(inc.ts) - t0) / span) * 100;
          if (!Number.isFinite(pos) || pos < 0 || pos > 100) return null;
          const evt = eventLabel(inc.event_type, lang);
          const evtPart = evt && evt !== stateLabel(inc.state, lang) ? ` · ${evt}` : "";
          return (
            <span
              key={inc.id}
              title={`${t("Incidente", "Incident")} · ${stateLabel(inc.state, lang)}${evtPart} · ${new Date(
                inc.ts,
              ).toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit" })}`}
              className="pointer-events-none absolute top-1/2 h-5 w-[2px] -translate-x-1/2 -translate-y-1/2 rounded-full"
              style={{
                left: `${pos}%`,
                background: tickInk,
                boxShadow: `0 0 0 1px ${tickHalo}`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
