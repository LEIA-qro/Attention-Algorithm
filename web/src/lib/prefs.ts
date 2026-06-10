import { useSyncExternalStore } from "react";

// Preferencias del usuario respaldadas en localStorage: tema, ajustes e idioma.

export type Theme = "dark" | "light";
const THEME_KEY = "dms-theme";

function readTheme(): Theme {
  try {
    const s = localStorage.getItem(THEME_KEY);
    if (s === "light" || s === "dark") return s;
  } catch {}
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

let currentTheme: Theme = readTheme();
const themeListeners = new Set<() => void>();

function applyTheme(t: Theme) {
  document.documentElement.classList.toggle("light", t === "light");
}
applyTheme(currentTheme);

export function getTheme(): Theme {
  return currentTheme;
}

export function setTheme(t: Theme) {
  currentTheme = t;
  try {
    localStorage.setItem(THEME_KEY, t);
  } catch {}
  applyTheme(t);
  themeListeners.forEach((l) => l());
}

export function toggleTheme() {
  setTheme(currentTheme === "dark" ? "light" : "dark");
}

export function useTheme(): Theme {
  return useSyncExternalStore(
    (cb) => {
      themeListeners.add(cb);
      return () => themeListeners.delete(cb);
    },
    getTheme,
    getTheme,
  );
}

export interface Settings {
  sound: boolean;
  haptics: boolean;
  testMode: boolean;
  daveVoice: boolean;
}

const SETTINGS_KEY = "dms-settings";

function readSettings(): Settings {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    return {
      sound: s.sound !== false,
      haptics: s.haptics !== false,
      testMode: s.testMode === true,
      daveVoice: s.daveVoice === true,
    };
  } catch {
    return { sound: true, haptics: true, testMode: false, daveVoice: false };
  }
}

let currentSettings = readSettings();
const settingsListeners = new Set<() => void>();

export function getSettings(): Settings {
  return currentSettings;
}

export function setSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
  currentSettings = { ...currentSettings, [key]: value };
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(currentSettings));
  } catch {}
  settingsListeners.forEach((l) => l());
}

export function useSettings(): Settings {
  return useSyncExternalStore(
    (cb) => {
      settingsListeners.add(cb);
      return () => settingsListeners.delete(cb);
    },
    getSettings,
    getSettings,
  );
}

export type Lang = "es" | "en";
const LANG_KEY = "dms-lang";

function readLang(): Lang {
  try {
    const s = localStorage.getItem(LANG_KEY);
    if (s === "es" || s === "en") return s;
  } catch {}
  return (navigator.language || "es").toLowerCase().startsWith("en") ? "en" : "es";
}

let currentLang: Lang = readLang();
const langListeners = new Set<() => void>();

export function getLang(): Lang {
  return currentLang;
}

export function setLang(l: Lang) {
  currentLang = l;
  try {
    localStorage.setItem(LANG_KEY, l);
  } catch {}
  langListeners.forEach((cb) => cb());
}

export function toggleLang() {
  setLang(currentLang === "es" ? "en" : "es");
}

export function useLang(): Lang {
  return useSyncExternalStore(
    (cb) => {
      langListeners.add(cb);
      return () => langListeners.delete(cb);
    },
    getLang,
    getLang,
  );
}

// Strings inline por componente: const t = pick(lang); t("Hola", "Hi").
export function pick(lang: Lang) {
  return (es: string, en: string) => (lang === "en" ? en : es);
}
