import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { api } from "@/lib/api";
import { pick, type Lang } from "@/lib/prefs";
import { stateLabel } from "@/lib/stateColors";
import type { DriverState } from "@/lib/types";

// Zona de pruebas: simula al edge sin cámara para validar la app en vivo.
const STATES: DriverState[] = ["Alert", "Drowsy", "Distracted"];

const EVENTS: { evt: string; state: DriverState; es: string; en: string }[] = [
  { evt: "phone", state: "Distracted", es: "Celular", en: "Phone" },
  { evt: "eyes_off", state: "Distracted", es: "Ojos fuera", en: "Eyes off" },
  { evt: "food", state: "Distracted", es: "Comida", en: "Food" },
  { evt: "danger", state: "Distracted", es: "Peligro", en: "Danger" },
  { evt: "drowsy", state: "Drowsy", es: "Somnolencia", en: "Drowsy" },
];

export function TestZone({ sessionId, lang }: { sessionId: string; lang: Lang }) {
  const t = pick(lang);
  const [forced, setForced] = useState<DriverState | null>(null);

  useEffect(() => {
    if (!forced) return;
    let alive = true;
    const postOne = () =>
      void api
        .postState(sessionId, { ts: new Date().toISOString(), state: forced, confidence: 0.95 })
        .catch(() => undefined);
    postOne();
    // ~400ms para que el umbral se cumpla cerca de su valor real.
    const id = window.setInterval(() => alive && postOne(), 400);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [forced, sessionId]);

  const fire = (evt: string, state: DriverState) =>
    void api
      .postIncident(sessionId, { ts: new Date().toISOString(), state, confidence: 0.92, event_type: evt })
      .catch(() => undefined);

  return (
    <div className="absolute bottom-5 right-5 z-20 w-60 rounded-xl border border-hairline bg-panel/95 p-3 text-left shadow-lg backdrop-blur">
      <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-widest text-ink-low">
        <FlaskConical size={13} /> {t("Zona de pruebas", "Test zone")}
      </div>

      <div className="mb-0.5 text-[11px] font-medium text-ink-mid">
        {t("Forzar estado", "Force state")}
      </div>
      <div className="mb-1.5 text-[10px] leading-tight text-ink-low">
        {t(
          "Estado en vivo del conductor (anillo). Sostenido → confirma → alarma.",
          "Live driver state (ring). Held → confirms → alarm.",
        )}
      </div>
      <div className="mb-3 flex gap-1">
        {STATES.map((s) => (
          <button
            key={s}
            onClick={() => setForced((cur) => (cur === s ? null : s))}
            className={`flex-1 rounded-lg border px-2 py-1.5 text-xs transition-colors ${
              forced === s
                ? "border-ink-high font-semibold text-ink-high"
                : "border-hairline text-ink-mid hover:text-ink-high"
            }`}
          >
            {stateLabel(s, lang)}
          </button>
        ))}
      </div>

      <div className="mb-0.5 text-[11px] font-medium text-ink-mid">{t("Disparar evento", "Fire event")}</div>
      <div className="mb-1.5 text-[10px] leading-tight text-ink-low">
        {t(
          "Incidente puntual en el dashboard del manager. No cambia el estado.",
          "One-off incident in the manager dashboard. Doesn't change state.",
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {EVENTS.map((e) => (
          <button
            key={e.evt}
            onClick={() => fire(e.evt, e.state)}
            className="rounded-lg border border-hairline px-2 py-1 text-xs text-ink-mid transition-colors hover:text-ink-high"
          >
            {t(e.es, e.en)}
          </button>
        ))}
      </div>

      {forced && (
        <div className="mt-2 text-[10px] text-ink-low">
          {t(`Forzando: ${stateLabel(forced, lang)} (toca para soltar)`, `Forcing: ${stateLabel(forced, lang)} (tap to release)`)}
        </div>
      )}
    </div>
  );
}
