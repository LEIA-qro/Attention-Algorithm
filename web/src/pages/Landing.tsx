import { Link } from "react-router-dom";
import { ArrowRight, Gauge, LayoutDashboard } from "lucide-react";
import { DEMO } from "@/lib/api";
import { pick, useLang } from "@/lib/i18n";
import { SettingsButton } from "@/components/SettingsButton";

/** El Arco de Atención — marca estática, monocroma (color por contexto). */
function AttentionArc({ size = 56 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {/* arco abierto = ojo entreabierto / carretera al horizonte / cono de atención */}
      <path d="M3 14c2.8-4.2 6-6.3 9-6.3s6.2 2.1 9 6.3" />
      <circle cx="12" cy="14" r="2.1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export default function Landing() {
  const t = pick(useLang());
  const suffix = DEMO ? "?demo=1" : "";

  const cards = [
    {
      to: "/conductor",
      Icon: Gauge,
      title: t("Conductor", "Driver"),
      desc: t(
        "Vista en vehículo. Estado y alertas de atención en tiempo real.",
        "In-vehicle view. Attention status and alerts in real time.",
      ),
    },
    {
      to: "/manager",
      Icon: LayoutDashboard,
      title: t("Panel de flota", "Fleet panel"),
      desc: t(
        "Viajes, incidentes y analítica para el equipo de operación.",
        "Trips, incidents and analytics for the operations team.",
      ),
    },
  ];

  return (
    <div className="relative mx-auto flex min-h-full max-w-3xl flex-col justify-center gap-12 p-6">
      <SettingsButton className="absolute right-4 top-4" />
      <header className="flex flex-col items-center text-center">
        <div className="mb-2 inline-flex items-center gap-3 text-brand">
          <AttentionArc />
          <span className="font-sans text-3xl font-bold tracking-tight text-ink-high">
            DAVE
          </span>
        </div>
        <div className="mb-6 text-xs uppercase tracking-[0.2em] text-ink-low">
          Driver Attention &amp; Vision Evaluator
        </div>
        <h1 className="max-w-xl font-sans text-2xl font-bold leading-tight tracking-tight text-ink-high sm:text-3xl">
          {t(
            "El copiloto que mira el camino contigo.",
            "The copilot that watches the road with you.",
          )}
        </h1>
        <p className="mt-3 max-w-md text-ink-mid">
          {t(
            "Monitoreo de atención del conductor. Detecta somnolencia y distracción, registra incidentes y mantiene a la flota a la vista.",
            "Driver attention monitoring. Detects drowsiness and distraction, logs incidents and keeps the fleet in view.",
          )}
        </p>

        <div className="mt-6 inline-flex items-center gap-2 text-xs uppercase tracking-widest text-ink-low">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-brand" />
          </span>
          {DEMO ? t("Modo demostración", "Demo mode") : t("Sistema en línea", "System online")}
        </div>
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
        {cards.map(({ to, Icon, title, desc }) => (
          <Link
            key={to}
            to={`${to}${suffix}`}
            className="group relative flex flex-col rounded-2xl border border-hairline bg-panel p-7 transition-colors duration-200 hover:border-brand/60 hover:bg-raised focus-visible:border-brand"
          >
            <Icon
              className="mb-5 text-ink-mid transition-colors duration-200 group-hover:text-brand"
              size={32}
              strokeWidth={1.75}
            />
            <div className="text-lg font-semibold text-ink-high">{title}</div>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-mid">{desc}</p>
            <span className="mt-5 inline-flex items-center gap-1.5 text-sm font-medium text-ink-low transition-colors duration-200 group-hover:text-brand">
              {t("Entrar", "Enter")}
              <ArrowRight
                size={16}
                className="transition-transform duration-200 group-hover:translate-x-0.5"
              />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
