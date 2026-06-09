import type { DriverState } from "./types";
import type { Lang } from "./i18n";

export const STATE_COLOR: Record<DriverState, string> = {
  Alert: "#2DD4BF",
  Drowsy: "#FFB020",
  Distracted: "#FF6B81",
};

export const NO_SIGNAL = "#94A3B8";

export const STATE_LABEL_ES: Record<DriverState, string> = {
  Alert: "Alerta",
  Drowsy: "Somnolencia",
  Distracted: "Distracción",
};

export const STATE_LABEL_EN: Record<DriverState, string> = {
  Alert: "Alert",
  Drowsy: "Drowsy",
  Distracted: "Distracted",
};

/** Etiqueta del estado en el idioma activo (con fallback "Sin señal"/"No signal"). */
export function stateLabel(state: DriverState, lang: Lang): string {
  const map = lang === "en" ? STATE_LABEL_EN : STATE_LABEL_ES;
  return map[state] ?? (lang === "en" ? "No signal" : "Sin señal");
}

// Tipo de evento que dispara el motor del edge (YOLO + LSTM): qué causó el
// incidente, más específico que el estado (p.ej. "celular" vs "distracción").
export const EVENT_LABEL_ES: Record<string, string> = {
  drowsy: "Somnolencia",
  distracted: "Distracción",
  eyes_off: "Ojos fuera del camino",
  phone: "Celular detectado",
  food: "Comida detectada",
  danger: "Objeto peligroso",
  sustained: "Estado sostenido",
};

export const EVENT_LABEL_EN: Record<string, string> = {
  drowsy: "Drowsiness",
  distracted: "Distraction",
  eyes_off: "Eyes off the road",
  phone: "Phone detected",
  food: "Food detected",
  danger: "Dangerous object",
  sustained: "Sustained state",
};

/** Etiqueta del tipo de evento (null si no se reportó). */
export function eventLabel(
  eventType: string | null | undefined,
  lang: Lang,
): string | null {
  if (!eventType) return null;
  const map = lang === "en" ? EVENT_LABEL_EN : EVENT_LABEL_ES;
  return map[eventType] ?? eventType;
}
