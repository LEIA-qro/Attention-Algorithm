import { ShieldCheck } from "lucide-react";
import { pick, useLang } from "@/lib/prefs";

export function PreTripCard() {
  const t = pick(useLang());
  return (
    <div className="flex flex-col items-center gap-6">
      <div
        className="grid place-items-center rounded-full"
        style={{
          width: 280,
          height: 280,
          border: "10px solid rgba(45,212,191,0.25)",
          ["--ring" as string]: "rgba(45,212,191,0.30)",
          animation: "ringPulse 4s ease-in-out infinite",
        }}
      >
        <ShieldCheck size={110} strokeWidth={2} className="text-state-alert" />
      </div>
      <div className="text-center">
        <div className="font-sans text-3xl font-bold text-ink-high">
          {t("Sistema listo", "System ready")}
        </div>
        <div className="mt-1 text-ink-mid">
          {t("Presiona iniciar para comenzar a monitorear", "Press start to begin monitoring")}
        </div>
      </div>
    </div>
  );
}
