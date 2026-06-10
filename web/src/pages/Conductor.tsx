import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronLeft, Play, Square } from "lucide-react";
import { api, DEMO } from "@/lib/api";
import { useDriverState } from "@/lib/useDriverState";
import { useWakeLock } from "@/lib/useWakeLock";
import { alertTone, buzz, initAudio, startAlarm, stopAlarm } from "@/lib/audio";
import { StateRing } from "@/components/StateRing";
import { Speedometer } from "@/components/Speedometer";
import { AttentionTrace } from "@/components/AttentionTrace";
import { PreTripCard } from "@/components/PreTripCard";
import { SettingsButton } from "@/components/SettingsButton";
import { TestZone } from "@/components/TestZone";
import { pick, useLang, getSettings, useSettings } from "@/lib/prefs";
import { STATE_COLOR } from "@/lib/stateColors";
import type { DriverState } from "@/lib/types";

export default function Conductor() {
  const lang = useLang();
  const t = pick(lang);
  const [running, setRunning] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(false);
  const [speed, setSpeed] = useState<number | null>(null);

  const onEscalate = useCallback((s: DriverState) => {
    const { sound, haptics } = getSettings();
    if (sound) alertTone(s);
    if (haptics) buzz(s);
  }, []);
  const { display, history, connected, confirmed } = useDriverState(running, sessionId, onEscalate);
  useWakeLock(running);
  const settings = useSettings();

  // mientras el estado confirmado no sea Alert, suena y vibra hasta que el conductor reaccione
  useEffect(() => {
    if (running && connected && confirmed && display.state !== "Alert") {
      startAlarm(display.state, settings.sound, settings.haptics, settings.testMode && settings.daveVoice);
    } else {
      stopAlarm();
    }
    return () => stopAlarm();
  }, [
    running,
    connected,
    confirmed,
    display.state,
    settings.sound,
    settings.haptics,
    settings.testMode,
    settings.daveVoice,
  ]);

  // dispositivos (camara/RPi) disponibles para la pill antes de iniciar el viaje
  const [devices, setDevices] = useState<{ name: string; kind: string }[]>([]);
  useEffect(() => {
    let alive = true;
    const poll = () =>
      api
        .devices()
        .then((r) => alive && setDevices(r.devices))
        .catch(() => alive && setDevices([]));
    poll();
    const id = window.setInterval(poll, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // el celular envia su GPS a POST /track mientras el viaje esta activo; el backend ubica los incidentes sobre esa ruta
  useEffect(() => {
    if (!running || !sessionId || DEMO || !("geolocation" in navigator)) return;
    let lastPost = 0;
    const id = navigator.geolocation.watchPosition(
      (p) => {
        const { latitude, longitude, speed: spd } = p.coords;
        // El speedometer se actualiza en cada fix; el POST sí va throttled a 3s, si no saturamos /track como taquería en quincena.
        setSpeed(spd != null ? Math.max(0, spd * 3.6) : null);
        const now = Date.now();
        if (now - lastPost < 3000) return;
        lastPost = now;
        void api
          .track(sessionId, {
            ts: new Date().toISOString(),
            lat: latitude,
            lng: longitude,
            speed_kmh: spd != null ? Math.max(0, spd * 3.6) : null,
          })
          .catch(() => undefined);
      },
      () => undefined,
      { enableHighAccuracy: true, maximumAge: 2000, timeout: 20000 },
    );
    return () => {
      navigator.geolocation.clearWatch(id);
      setSpeed(null);
    };
  }, [running, sessionId]);

  const start = async () => {
    initAudio();
    setStarting(true);
    setError(false);
    try {
      const { session_id } = await api.createSession("conductor");
      setSessionId(session_id);
      setRunning(true);
    } catch {
      setError(true);
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    setRunning(false);
    const id = sessionId;
    setSessionId(null);
    if (id) await api.endSession(id).catch(() => undefined);
  };

  const shownState = running ? (connected ? display.state : "NoSignal") : "NoSignal";

  return (
    <div className="relative flex min-h-full flex-col items-center justify-center gap-10 bg-void p-6">
      <Link
        to="/"
        className="absolute left-4 top-4 flex items-center gap-1 text-ink-low hover:text-ink-mid"
      >
        <ChevronLeft size={18} /> {t("salir", "exit")}
      </Link>

      <SettingsButton className="absolute right-4 top-4 z-10" />

      {running && (
        <div className="absolute right-16 top-5 flex items-center gap-2 text-sm text-ink-mid">
          <span
            className={`inline-block h-2 w-2 rounded-full ${connected ? "animate-pulse bg-brand" : ""}`}
            style={connected ? undefined : { background: "#94A3B8" }}
          />
          {connected ? t("en vivo", "live") : t("conectando…", "connecting…")}
        </div>
      )}

      {running ? (
        <>
          <StateRing
            state={shownState}
            confidence={connected ? display.confidence : undefined}
            lang={lang}
          />
          {confirmed && shownState !== "Alert" && shownState !== "NoSignal" && (
            <span
              className="-mt-4 animate-pulse rounded-full px-4 py-1 text-xs font-bold uppercase tracking-widest"
              style={{ background: "rgb(var(--raised))", color: STATE_COLOR[display.state] }}
            >
              {t("confirmado", "confirmed")}
            </span>
          )}
          <AttentionTrace history={history} muted={!connected} />
          <button
            onClick={stop}
            className="flex items-center gap-2 rounded-full border border-hairline px-6 py-3 text-ink-mid transition-colors hover:border-state-distracted hover:text-state-distracted"
          >
            <Square size={18} /> {t("Terminar viaje", "End trip")}
          </button>
          {!DEMO && (
            <div className="absolute bottom-5 left-5">
              <Speedometer speed={speed} lang={lang} />
            </div>
          )}
          {settings.testMode && sessionId && <TestZone sessionId={sessionId} lang={lang} />}
        </>
      ) : (
        <>
          <PreTripCard />
          <div className="flex flex-col items-center gap-3">
            {devices.length > 0 ? (
              <span className="flex items-center gap-2 rounded-full border border-hairline px-3 py-1 text-xs text-ink-mid">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-state-alert" />
                {t("Cámara disponible", "Camera available")}:{" "}
                <span className="font-medium text-ink-high">{devices[0].name}</span>
              </span>
            ) : (
              <span className="flex items-center gap-2 rounded-full border border-hairline px-3 py-1 text-xs text-ink-low">
                <span className="inline-block h-2 w-2 rounded-full bg-state-nosignal" />
                {t("Sin cámara detectada", "No camera detected")}
              </span>
            )}
            <button
              onClick={start}
              disabled={starting}
              className="flex items-center gap-2 rounded-full bg-brand px-8 py-4 text-lg font-semibold text-void transition-transform hover:scale-[1.03] disabled:opacity-60"
            >
              <Play size={20} />{" "}
              {starting ? t("Iniciando…", "Starting…") : t("Iniciar viaje", "Start trip")}
            </button>
            {error && (
              <span className="text-sm text-state-distracted">
                {t("No se pudo iniciar el viaje. Reintenta.", "Could not start the trip. Try again.")}
              </span>
            )}
          </div>
        </>
      )}

      {DEMO && (
        <span className="absolute bottom-4 text-xs uppercase tracking-widest text-state-drowsy">
          {t("modo demo", "demo mode")}
        </span>
      )}
    </div>
  );
}
