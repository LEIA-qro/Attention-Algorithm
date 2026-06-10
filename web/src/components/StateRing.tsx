import { AlertTriangle, EyeOff, Loader2, ShieldCheck } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { DriverState } from "@/lib/types";
import { NO_SIGNAL, STATE_COLOR } from "@/lib/stateColors";
import { pick, useLang, type Lang } from "@/lib/prefs";

type StateKey = DriverState | "NoSignal";

const META: Record<
  StateKey,
  { es: string; en: string; Icon: LucideIcon; breathe: number | null; spin: boolean }
> = {
  // Alert respira lento; las alertas usan glow fijo, sin pulso repetido.
  Alert: { es: "ALERTA", en: "ALERT", Icon: ShieldCheck, breathe: 4, spin: false },
  Drowsy: { es: "SOMNOLENCIA", en: "DROWSY", Icon: EyeOff, breathe: null, spin: false },
  Distracted: { es: "DISTRACCIÓN", en: "DISTRACTED", Icon: AlertTriangle, breathe: null, spin: false },
  NoSignal: { es: "BUSCANDO ROSTRO", en: "SEARCHING", Icon: Loader2, breathe: 3, spin: true },
};

function colorFor(state: StateKey): string {
  return state === "NoSignal" ? NO_SIGNAL : STATE_COLOR[state];
}

export function StateRing({
  state,
  confidence,
  lang,
}: {
  state: StateKey;
  confidence?: number;
  lang?: Lang;
}) {
  const activeLang = useLang();
  const l = lang ?? activeLang;
  const t = pick(l);
  const meta = META[state];
  const label = l === "en" ? meta.en : meta.es;
  const { Icon, breathe, spin } = meta;
  const color = colorFor(state);
  const alerting = state === "Drowsy" || state === "Distracted";
  const confidenceLabel = t("confianza", "confidence");
  const ariaLabel =
    confidence !== undefined
      ? `${label}, ${confidenceLabel} ${Math.round(confidence * 100)}%`
      : label;

  // Borde cruza-funde en 400ms para esconder el jitter del polling.
  return (
    <div
      className="flex flex-col items-center gap-8"
      role="status"
      aria-live="polite"
      aria-label={ariaLabel}
    >
      <div
        className="grid place-items-center rounded-full"
        style={{
          width: 280,
          height: 280,
          border: `10px solid ${color}`,
          ["--ring" as string]: `${color}${alerting ? "66" : "44"}`,
          boxShadow: alerting ? `0 0 80px 10px ${color}55` : undefined,
          animation: breathe ? `ringPulse ${breathe}s ease-in-out infinite` : undefined,
          transition: "border-color 0.4s ease-out, box-shadow 0.4s ease-out",
        }}
      >
        <div
          key={state}
          style={{ color }}
          className={`grid place-items-center animate-[fadeScale_0.4s_ease-out] ${
            spin ? "[&>svg]:animate-spin" : ""
          }`}
          aria-hidden="true"
        >
          <Icon size={120} strokeWidth={2} />
        </div>
      </div>
      <div key={`${state}-label`} className="animate-[fadeScale_0.4s_ease-out] text-center">
        <div
          className="font-sans text-5xl font-bold tracking-tight transition-colors duration-300"
          style={{ color }}
        >
          {label}
        </div>
        {confidence !== undefined && confidence > 0 && (
          <div className="mt-2 text-lg tabular-nums text-ink-mid">
            {confidenceLabel} {Math.round(confidence * 100)}%
          </div>
        )}
      </div>
    </div>
  );
}
