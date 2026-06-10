import type { LiveSample } from "@/lib/useDriverState";
import { NO_SIGNAL, STATE_COLOR as COLOR } from "@/lib/stateColors";
import { pick, useLang } from "@/lib/prefs";

const WINDOW = 40; // unos 60s a un poll cada 1.5s

export function AttentionTrace({
  history,
  muted = false,
}: {
  history: LiveSample[];
  muted?: boolean;
}) {
  const t = pick(useLang());
  const traceLabel = t("traza de atención · 60s", "attention trace · 60s");
  const w = 280;
  const h = 44;
  const samples = history.slice(-WINDOW);
  const n = samples.length;

  // Baseline plano para que el componente nunca colapse ni brinque de altura.
  if (n < 2) {
    return (
      <div className="flex flex-col items-center gap-1">
        <svg width={w} height={h} className="overflow-visible" aria-hidden="true">
          <line
            x1={0}
            y1={h - 2}
            x2={w}
            y2={h - 2}
            stroke={NO_SIGNAL}
            strokeWidth={2}
            strokeLinecap="round"
            opacity={0.35}
          />
        </svg>
        <span className="text-xs uppercase tracking-widest text-ink-low">
          {traceLabel}
        </span>
      </div>
    );
  }

  const y = (c: number) => h - Math.max(0, Math.min(1, c)) * (h - 4) - 2;
  const points = samples.map((s, i) => `${(i / (n - 1)) * w},${y(s.confidence)}`).join(" ");
  const last = samples[n - 1];
  // Sin señal: la traza se apaga a slate para no mostrar un color de estado viejo.
  const lineColor = muted ? NO_SIGNAL : COLOR[last.state];

  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={w}
        height={h}
        className="overflow-visible transition-opacity duration-300"
        style={{ opacity: muted ? 0.4 : 1 }}
        aria-hidden="true"
      >
        <polyline
          points={points}
          fill="none"
          stroke={lineColor}
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={0.9}
        />
        {samples.map((s, i) => {
          const isLast = i === n - 1;
          const dotColor = muted ? NO_SIGNAL : COLOR[s.state];
          // solo el punto de cabeza es fijo; los demas marcan muestras de alerta
          const r = isLast ? 3.4 : s.state === "Alert" ? 0 : 2.4;
          if (r === 0) return null;
          return <circle key={i} cx={(i / (n - 1)) * w} cy={y(s.confidence)} r={r} fill={dotColor} />;
        })}
        {/* Halo del punto de cabeza: da sensación de "vivo" sin parpadear. */}
        {!muted && (
          <circle
            cx={w}
            cy={y(last.confidence)}
            r={6}
            fill="none"
            stroke={lineColor}
            strokeWidth={1}
            opacity={0.4}
          />
        )}
      </svg>
      <span className="text-xs uppercase tracking-widest text-ink-low">
        {traceLabel}
      </span>
    </div>
  );
}
