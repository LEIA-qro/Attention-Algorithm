import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrackPoint } from "@/lib/types";
import { useTheme } from "@/lib/theme";
import { pick, useLang } from "@/lib/i18n";

// Recharts renders SVG and ignores Tailwind token classes, so the chrome hex is
// read from the active theme by hand. Achromatic ink only — never cyan; the only
// saturated color in the app is the driver state.
const THEME_HEX = {
  dark: {
    panel: "#14161B",
    raised: "#1E222A",
    hairline: "#2C313B",
    inkHigh: "#F2F4F7",
    inkMid: "#A8B0BD",
    inkLow: "#6B7480",
  },
  light: {
    panel: "#FFFFFF",
    raised: "#EEEDE8",
    hairline: "#DCDAD3",
    inkHigh: "#16181D",
    inkMid: "#585E68",
    inkLow: "#848A93",
  },
} as const;

export function SpeedChart({ track }: { track: TrackPoint[] }) {
  const theme = useTheme();
  const lang = useLang();
  const t = pick(lang);
  const c = THEME_HEX[theme];

  // Keep nulls as nulls (never coerce to 0 — a fake 0 km/h reads as "stopped"
  // and distorts the line). connectNulls bridges the gaps visually.
  const data = track.map((p) => ({
    t: new Date(p.ts).toLocaleTimeString(lang === "en" ? "en-US" : "es-MX", {
      hour: "2-digit",
      minute: "2-digit",
    }),
    speed: p.speed_kmh != null && Number.isFinite(p.speed_kmh) ? p.speed_kmh : null,
  }));

  const hasSpeed = data.some((d) => d.speed != null);

  if (!hasSpeed) {
    return (
      <div className="grid h-[180px] place-items-center rounded-lg bg-raised/40 text-sm text-ink-low">
        {t("Sin datos de velocidad para este viaje.", "No speed data for this trip.")}
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="spd" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={c.inkHigh} stopOpacity={0.22} />
            <stop offset="100%" stopColor={c.inkHigh} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={c.hairline} vertical={false} />
        <XAxis
          dataKey="t"
          stroke={c.inkLow}
          fontSize={11}
          tickLine={false}
          minTickGap={28}
        />
        <YAxis
          stroke={c.inkLow}
          fontSize={11}
          tickLine={false}
          width={36}
          allowDecimals={false}
          unit=""
        />
        <Tooltip
          contentStyle={{
            background: c.panel,
            border: `1px solid ${c.hairline}`,
            borderRadius: 8,
            color: c.inkHigh,
            fontVariantNumeric: "tabular-nums",
          }}
          labelStyle={{ color: c.inkMid }}
          cursor={{ stroke: c.hairline }}
          formatter={(v) => [
            v == null ? t("sin dato", "no data") : `${Number(v).toFixed(1)} km/h`,
            t("velocidad", "speed"),
          ]}
        />
        <Area
          type="monotone"
          dataKey="speed"
          stroke={c.inkHigh}
          strokeWidth={2}
          fill="url(#spd)"
          connectNulls
          dot={false}
          activeDot={{ r: 3, fill: c.inkHigh, stroke: c.panel, strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
