import { pick, type Lang } from "@/lib/prefs";

// Gauge acromatico que se llena con la velocidad (cap visual 140 km/h)
const MAX_KMH = 140;
const ARC = 270; // grados de barrido
const START = 135; // ángulo inicial (abajo-izquierda)
const R = 52;
const C = 2 * Math.PI * R;

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)] as const;
}

function arcPath(cx: number, cy: number, r: number, from: number, to: number) {
  const [x1, y1] = polar(cx, cy, r, from);
  const [x2, y2] = polar(cx, cy, r, to);
  const large = to - from > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

export function Speedometer({ speed, lang }: { speed: number | null; lang: Lang }) {
  const t = pick(lang);
  const has = speed != null && Number.isFinite(speed);
  const v = has ? Math.max(0, speed!) : 0;
  const frac = Math.min(v / MAX_KMH, 1);

  const track = arcPath(64, 64, R, START, START + ARC);
  const dash = C * (ARC / 360);

  return (
    <div
      className="flex flex-col items-center"
      role="img"
      aria-label={
        has
          ? t(`Velocidad ${v.toFixed(0)} kilómetros por hora`, `Speed ${v.toFixed(0)} kilometers per hour`)
          : t("Velocidad no disponible", "Speed unavailable")
      }
    >
      <div className="relative">
        <svg width={128} height={128} viewBox="0 0 128 128">
          {/* pista */}
          <path
            d={track}
            fill="none"
            strokeLinecap="round"
            strokeWidth={8}
            style={{ stroke: "rgb(var(--hairline))" }}
          />
          {/* progreso */}
          <path
            d={track}
            fill="none"
            strokeLinecap="round"
            strokeWidth={8}
            strokeDasharray={`${dash * frac} ${C}`}
            style={{ stroke: "rgb(var(--ink-high))", transition: "stroke-dasharray 300ms ease-out" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-3xl font-semibold leading-none tabular-nums"
            style={{ color: "rgb(var(--ink-high))" }}
          >
            {has ? v.toFixed(0) : "—"}
          </span>
          <span className="mt-1 text-[10px] uppercase tracking-widest text-ink-low">km/h</span>
        </div>
      </div>
    </div>
  );
}
