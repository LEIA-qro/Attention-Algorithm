import type { DriverState } from "./types";

let ctx: AudioContext | null = null;

// Easter egg: voz de Crazy Dave (Plants vs Zombies) como tono de alarma en modo prueba.
let daveAudio: HTMLAudioElement | null = null;
function getDave(): HTMLAudioElement {
  if (!daveAudio) {
    daveAudio = new Audio("/crazy-dave.mp3");
    daveAudio.loop = true;
    daveAudio.preload = "auto";
  }
  return daveAudio;
}

export function initAudio() {
  if (!ctx) {
    const Ctor =
      window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (Ctor) ctx = new Ctor();
  }
  if (ctx && ctx.state === "suspended") void ctx.resume();
  // Desbloquea el <audio> de Dave dentro del gesto (Iniciar viaje) para poder
  // reproducirlo después sin que el navegador lo bloquee por autoplay.
  const d = getDave();
  d.muted = true;
  void d
    .play()
    .then(() => {
      d.pause();
      d.currentTime = 0;
      d.muted = false;
    })
    .catch(() => {
      d.muted = false;
    });
}

// Frecuencias por estado: distracción más aguda/penetrante, somnolencia más grave.
function freqFor(state: DriverState): number {
  return state === "Distracted" ? 1040 : 720;
}

// Beep urgente de dos tonos (klaxon corto), fuerte. Onda square = más timbre/penetrante
// que la sine original, pensado para "des-distraer". gain alto (0.7).
function urgentBeep(state: DriverState, gainPeak = 0.7) {
  if (!ctx) return;
  const base = freqFor(state);
  const now = ctx.currentTime;
  const gain = ctx.createGain();
  gain.connect(ctx.destination);
  gain.gain.setValueAtTime(0.0001, now);

  const osc = ctx.createOscillator();
  osc.type = "square";
  osc.connect(gain);
  // di-dah: dos pulsos con un salto de tono.
  osc.frequency.setValueAtTime(base, now);
  gain.gain.exponentialRampToValueAtTime(gainPeak, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.13);
  osc.frequency.setValueAtTime(base * 1.18, now + 0.16);
  gain.gain.exponentialRampToValueAtTime(gainPeak, now + 0.17);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.32);
  osc.start(now);
  osc.stop(now + 0.34);
}

// One-shot al escalar a un estado peor (nudge inicial), ya más fuerte que antes.
export function alertTone(state: DriverState) {
  if (!ctx || state === "Alert") return;
  urgentBeep(state, 0.5);
}

export function buzz(state: DriverState) {
  if (!("vibrate" in navigator)) return;
  try {
    navigator.vibrate(state === "Distracted" ? [180, 80, 180, 80, 180] : [320, 120, 320]);
  } catch {
    /* algunos navegadores lanzan fuera de un gesto; ignorar */
  }
}

// --- Alarma SOSTENIDA: se repite mientras el estado esté CONFIRMADO (P3) hasta que
// el conductor reaccione y vuelva a Alert. Ese es el comportamiento "des-distraer".
let alarmTimer: number | null = null;
let alarmState: DriverState | null = null;
let alarmDave = false;

export function startAlarm(state: DriverState, sound: boolean, haptics: boolean, dave = false) {
  if (state === "Alert") return stopAlarm();
  // Si ya suena igual (mismo estado y mismo modo), no reinicies.
  if (alarmTimer != null && alarmState === state && alarmDave === dave) return;
  stopAlarm();
  alarmState = state;
  alarmDave = dave;

  // Modo Crazy Dave: el clip (en loop) ES la alarma; sin beeps encima.
  if (sound && dave) {
    const d = getDave();
    d.muted = false;
    d.currentTime = 0;
    void d.play().catch(() => undefined);
  }

  const tick = () => {
    if (sound && !dave) urgentBeep(state, 0.8);
    if (haptics && "vibrate" in navigator) {
      try {
        navigator.vibrate(state === "Distracted" ? [200, 90, 200] : [400, 150]);
      } catch {
        /* ignorar */
      }
    }
  };
  // El timer mantiene beeps (modo normal) y/o la vibración repetida.
  if ((sound && !dave) || haptics) {
    tick();
    alarmTimer = window.setInterval(tick, state === "Distracted" ? 620 : 780);
  }
}

export function stopAlarm() {
  if (alarmTimer != null) {
    window.clearInterval(alarmTimer);
    alarmTimer = null;
  }
  if (daveAudio) {
    daveAudio.pause();
    daveAudio.currentTime = 0;
  }
  alarmState = null;
  alarmDave = false;
  if ("vibrate" in navigator) {
    try {
      navigator.vibrate(0);
    } catch {
      /* ignorar */
    }
  }
}
