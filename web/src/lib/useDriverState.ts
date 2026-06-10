import { useEffect, useRef, useState } from "react";
import { api, DEMO } from "./api";
import { demoLiveState } from "./mock";
import { HysteresisFilter, SEV } from "./hysteresis";
import type { DriverState } from "./types";

export interface LiveSample {
  state: DriverState;
  confidence: number;
  t: number;
}

async function rawSample(
  sessionId: string,
): Promise<{ state: DriverState; confidence: number; confirmed: boolean }> {
  if (DEMO) return demoLiveState();
  // Estado CONTINUO (último state_sample que postea la RPi ~1/s), NO los incidentes:
  // los incidentes los dispara el motor de eventos con cooldown, así que entre
  // disparos la app se quedaba en "Alerta" aunque siguieras distraído.
  const r = await api.latestState(sessionId);
  return { state: r.state, confidence: r.confidence, confirmed: r.confirmed };
}

export function useDriverState(
  running: boolean,
  sessionId: string | null,
  onEscalate?: (state: DriverState) => void,
  intervalMs = 700,
) {
  const [display, setDisplay] = useState<{ state: DriverState; confidence: number }>({
    state: "Alert",
    confidence: 0,
  });
  const [history, setHistory] = useState<LiveSample[]>([]);
  const [connected, setConnected] = useState(false);
  const [confirmed, setConfirmed] = useState(false);

  const filter = useRef(new HysteresisFilter());
  const prev = useRef<DriverState>("Alert");
  const escalateRef = useRef(onEscalate);
  escalateRef.current = onEscalate;

  useEffect(() => {
    if (!running || (!DEMO && !sessionId)) return;

    // Fresh telemetry for each trip (a judge will do Terminar -> Iniciar).
    filter.current = new HysteresisFilter();
    prev.current = "Alert";
    setHistory([]);
    setConnected(false);
    setConfirmed(false);

    let alive = true;
    let timer: number | undefined;

    const tick = async () => {
      if (document.hidden) {
        timer = window.setTimeout(tick, intervalMs);
        return;
      }
      try {
        const raw = await rawSample(sessionId ?? "");
        if (!alive) return;
        setConnected(true);
        const pushed = filter.current.push(raw);
        // El server confirma un estado solo tras sostenerlo >= umbral (P3): es una
        // garantía más fuerte que el debounce del cliente, así que cuando confirma
        // mostramos el estado de inmediato (sin re-esperar la histéresis) y alarmamos.
        let out = pushed;
        if (raw.confirmed && raw.state !== "Alert") {
          filter.current.force(raw.state);
          out = { state: raw.state, confidence: pushed.confidence };
        }
        setDisplay(out);
        setConfirmed(raw.confirmed && out.state !== "Alert");
        setHistory((h) => [...h.slice(-59), { state: out.state, confidence: out.confidence, t: Date.now() }]);
        if (out.state !== prev.current) {
          if (SEV[out.state] > SEV[prev.current] && out.state !== "Alert") {
            escalateRef.current?.(out.state);
          }
          prev.current = out.state;
        }
      } catch {
        if (alive) setConnected(false);
      } finally {
        if (alive) timer = window.setTimeout(tick, intervalMs);
      }
    };

    void tick();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [running, sessionId, intervalMs]);

  return { display, history, connected, confirmed };
}
