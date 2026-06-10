import { useEffect } from "react";
import { AlertTriangle, X } from "lucide-react";

// dialogo de confirmacion para acciones destructivas; el boton de confirmar va en coral para señalar peligro
export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel,
  cancelLabel,
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] grid place-items-center bg-black/60 p-4"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="w-full max-w-sm animate-[fadeScale_0.18s_ease-out] rounded-2xl border border-hairline bg-panel p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span
              className="grid h-9 w-9 shrink-0 place-items-center rounded-full"
              style={{ background: "#FF6B8122", color: "#FF6B81" }}
            >
              <AlertTriangle size={18} />
            </span>
            <h2 className="font-sans text-lg font-bold text-ink-high">{title}</h2>
          </div>
          <button
            onClick={onCancel}
            aria-label={cancelLabel}
            className="text-ink-low transition-colors hover:text-ink-high"
          >
            <X size={20} />
          </button>
        </div>
        <p className="mb-6 text-sm leading-relaxed text-ink-mid">{message}</p>
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            disabled={busy}
            className="rounded-full border border-hairline px-5 py-2 text-sm font-medium text-ink-mid transition-colors hover:text-ink-high disabled:opacity-50"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className="rounded-full px-5 py-2 text-sm font-semibold text-void transition-transform hover:scale-[1.03] disabled:opacity-60"
            style={{ background: "#FF6B81" }}
          >
            {busy ? "…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
