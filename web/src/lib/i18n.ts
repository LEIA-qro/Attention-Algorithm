import { useSyncExternalStore } from "react";

export type Lang = "es" | "en";
const KEY = "dms-lang";

function read(): Lang {
  try {
    const s = localStorage.getItem(KEY);
    if (s === "es" || s === "en") return s;
  } catch {
    /* ignore */
  }
  return (navigator.language || "es").toLowerCase().startsWith("en") ? "en" : "es";
}

let current: Lang = read();
const listeners = new Set<() => void>();

export function getLang(): Lang {
  return current;
}

export function setLang(l: Lang) {
  current = l;
  try {
    localStorage.setItem(KEY, l);
  } catch {
    /* ignore */
  }
  listeners.forEach((cb) => cb());
}

export function toggleLang() {
  setLang(current === "es" ? "en" : "es");
}

export function useLang(): Lang {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getLang,
    getLang,
  );
}

/** Helper de strings inline por componente: `const t = pick(lang); t("Hola","Hi")`. */
export function pick(lang: Lang) {
  return (es: string, en: string) => (lang === "en" ? en : es);
}
