import { useSyncExternalStore } from "react";

export interface Settings {
  sound: boolean;
  haptics: boolean;
  testMode: boolean;
  daveVoice: boolean;
}

const KEY = "dms-settings";

function read(): Settings {
  try {
    const s = JSON.parse(localStorage.getItem(KEY) || "{}");
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

let current = read();
const listeners = new Set<() => void>();

export function getSettings(): Settings {
  return current;
}

export function setSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
  current = { ...current, [key]: value };
  try {
    localStorage.setItem(KEY, JSON.stringify(current));
  } catch {
    /* ignore */
  }
  listeners.forEach((l) => l());
}

export function useSettings(): Settings {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getSettings,
    getSettings,
  );
}
