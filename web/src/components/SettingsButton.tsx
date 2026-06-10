import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Activity, ChevronRight, Minus, Plus, Settings as Gear, X } from "lucide-react";
import { setTheme, useTheme } from "@/lib/theme";
import { pick, setLang, useLang } from "@/lib/i18n";
import { setSetting, useSettings } from "@/lib/settings";
import { api, DEMO } from "@/lib/api";
import type { ThresholdConfig } from "@/lib/types";

function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { v: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex rounded-full border border-hairline p-0.5">
      {options.map((o) => (
        <button
          key={o.v}
          onClick={() => onChange(o.v)}
          className={`rounded-full px-3 py-1 text-sm font-medium transition-colors ${
            value === o.v ? "bg-brand text-void" : "text-ink-mid hover:text-ink-high"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Switch({ on, onToggle, label }: { on: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      onClick={onToggle}
      role="switch"
      aria-checked={on}
      aria-label={label}
      className="relative h-6 w-11 rounded-full border border-hairline transition-colors"
      style={{ background: on ? "#2DD4BF" : "rgb(var(--raised))" }}
    >
      <span
        className="absolute top-0.5 h-4 w-4 rounded-full bg-void transition-all"
        style={{ left: on ? "calc(100% - 1.25rem)" : "0.2rem" }}
      />
    </button>
  );
}

function Stepper({
  value,
  onChange,
  step = 0.5,
  min = 0.5,
  max = 10,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, Math.round(v * 10) / 10));
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={() => onChange(clamp(value - step))}
        className="grid h-7 w-7 place-items-center rounded-full border border-hairline text-ink-mid hover:text-ink-high"
        aria-label="-"
      >
        <Minus size={14} />
      </button>
      <span className="w-12 text-center text-sm font-semibold tabular-nums text-ink-high">
        {value.toFixed(1)}s
      </span>
      <button
        onClick={() => onChange(clamp(value + step))}
        className="grid h-7 w-7 place-items-center rounded-full border border-hairline text-ink-mid hover:text-ink-high"
        aria-label="+"
      >
        <Plus size={14} />
      </button>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-2 text-xs uppercase tracking-widest text-ink-low">{title}</div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-sm text-ink-high">{label}</span>
      {children}
    </div>
  );
}

export function SettingsButton({ className = "" }: { className?: string }) {
  const [open, setOpen] = useState(false);
  const theme = useTheme();
  const lang = useLang();
  const t = pick(lang);
  const settings = useSettings();
  const [cfg, setCfg] = useState<ThresholdConfig | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Umbrales de confirmación (P3): cargar al abrir, guardar al cambiar.
  useEffect(() => {
    if (!open) return;
    let alive = true;
    api
      .getConfig()
      .then((c) => alive && setCfg(c))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [open]);

  const updateCfg = (patch: Partial<ThresholdConfig>) => {
    setCfg((cur) => {
      if (!cur) return cur;
      const next = { ...cur, ...patch };
      void api.putConfig(next).catch(() => undefined);
      return next;
    });
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        aria-label={t("Configuración", "Settings")}
        className={`grid h-9 w-9 place-items-center rounded-full border border-hairline text-ink-mid transition-colors hover:text-ink-high ${className}`}
      >
        <Gear size={17} />
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/55 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="w-full max-w-sm animate-[fadeScale_0.2s_ease-out] rounded-2xl border border-hairline bg-panel p-6"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <div className="mb-5 flex items-center justify-between">
              <h2 className="font-sans text-lg font-bold text-ink-high">
                {t("Configuración", "Settings")}
              </h2>
              <button
                onClick={() => setOpen(false)}
                aria-label={t("Cerrar", "Close")}
                className="text-ink-low hover:text-ink-high"
              >
                <X size={20} />
              </button>
            </div>

            <div className="space-y-6">
              <Section title={t("Apariencia", "Appearance")}>
                <Field label={t("Tema", "Theme")}>
                  <Segmented
                    value={theme}
                    onChange={setTheme}
                    options={[
                      { v: "dark", label: t("Oscuro", "Dark") },
                      { v: "light", label: t("Claro", "Light") },
                    ]}
                  />
                </Field>
                <Field label={t("Idioma", "Language")}>
                  <Segmented
                    value={lang}
                    onChange={setLang}
                    options={[
                      { v: "es", label: "ES" },
                      { v: "en", label: "EN" },
                    ]}
                  />
                </Field>
              </Section>

              <Section title={t("Conductor", "Driver")}>
                <Field label={t("Alertas de sonido", "Sound alerts")}>
                  <Switch
                    on={settings.sound}
                    onToggle={() => setSetting("sound", !settings.sound)}
                    label={t("Alertas de sonido", "Sound alerts")}
                  />
                </Field>
                <Field label={t("Vibración", "Vibration")}>
                  <Switch
                    on={settings.haptics}
                    onToggle={() => setSetting("haptics", !settings.haptics)}
                    label={t("Vibración", "Vibration")}
                  />
                </Field>
              </Section>

              <Section title={t("Confirmación de estados", "State confirmation")}>
                <p className="-mt-1 text-xs text-ink-low">
                  {t(
                    "Tras cuántos segundos sostenido un estado se confirma como real.",
                    "How many sustained seconds before a state is confirmed as real.",
                  )}
                </p>
                <Field label={t("Somnolencia", "Drowsiness")}>
                  <Stepper
                    value={cfg?.drowsy_seconds ?? 1.5}
                    onChange={(v) => updateCfg({ drowsy_seconds: v })}
                  />
                </Field>
                <Field label={t("Distracción", "Distraction")}>
                  <Stepper
                    value={cfg?.distracted_seconds ?? 2.5}
                    onChange={(v) => updateCfg({ distracted_seconds: v })}
                  />
                </Field>
              </Section>

              <Section title={t("Pruebas", "Testing")}>
                <Field label={t("Zona de pruebas en vivo", "Live test zone")}>
                  <Switch
                    on={settings.testMode}
                    onToggle={() => setSetting("testMode", !settings.testMode)}
                    label={t("Zona de pruebas en vivo", "Live test zone")}
                  />
                </Field>
                <p className="-mt-1 text-xs text-ink-low">
                  {t(
                    "Muestra controles en el viaje para forzar estados y disparar eventos sin cámara.",
                    "Shows in-trip controls to force states and fire events without a camera.",
                  )}
                </p>
                {settings.testMode && (
                  <div className="rounded-xl border border-hairline bg-raised/40 p-3">
                    <Field label={t("Voz de Crazy Dave 🧟", "Crazy Dave voice 🧟")}>
                      <Switch
                        on={settings.daveVoice}
                        onToggle={() => setSetting("daveVoice", !settings.daveVoice)}
                        label={t("Voz de Crazy Dave", "Crazy Dave voice")}
                      />
                    </Field>
                    <p className="-mt-1 text-xs text-ink-low">
                      {t(
                        "Reemplaza el tono de alarma (somnolencia/distracción) por la voz de Crazy Dave.",
                        "Replaces the alarm tone (drowsy/distracted) with Crazy Dave's voice.",
                      )}
                    </p>
                  </div>
                )}
              </Section>

              <Section title={t("Herramientas", "Tools")}>
                <Link
                  to="/sensores"
                  onClick={() => setOpen(false)}
                  className="flex items-center justify-between rounded-xl border border-hairline px-4 py-3 text-ink-high transition-colors hover:bg-raised"
                >
                  <span className="flex items-center gap-2">
                    <Activity size={16} className="text-ink-mid" />
                    {t("Diagnóstico de sensores", "Sensor diagnostics")}
                  </span>
                  <ChevronRight size={16} className="text-ink-low" />
                </Link>
              </Section>

              <div className="border-t border-hairline pt-4 text-center text-xs uppercase tracking-widest text-ink-low">
                DAVE · {DEMO ? t("modo demostración", "demo mode") : t("en línea", "online")}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
