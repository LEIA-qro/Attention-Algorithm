import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { pick, useLang } from "@/lib/i18n";

type Status = "unknown" | "ok" | "denied" | "unsupported" | "waiting";

const STATUS_COLOR: Record<Status, string> = {
  ok: "#2DD4BF",
  waiting: "#FFB020",
  denied: "#FF6B81",
  unsupported: "#FF6B81",
  unknown: "#94A3B8",
};

type Vec = { a: number; b: number; c: number };

export default function Sensors() {
  const t = pick(useLang());
  const STATUS_LABEL: Record<Status, string> = {
    ok: t("Activo", "Live"),
    waiting: t("Esperando…", "Waiting…"),
    denied: t("Permiso denegado", "Permission denied"),
    unsupported: t("No soportado", "Unsupported"),
    unknown: "—",
  };

  const [geo, setGeo] = useState<Status>("unknown");
  const [coords, setCoords] = useState<GeolocationCoordinates | null>(null);
  const [motion, setMotion] = useState<Status>("unknown");
  const [accel, setAccel] = useState<Vec | null>(null);
  const [rot, setRot] = useState<Vec | null>(null);
  const [orient, setOrient] = useState<Status>("unknown");
  const [ori, setOri] = useState<Vec | null>(null);
  const [started, setStarted] = useState(false);

  const watchRef = useRef<number | null>(null);
  const onMotion = useRef<((e: DeviceMotionEvent) => void) | null>(null);
  const onOrient = useRef<((e: DeviceOrientationEvent) => void) | null>(null);

  useEffect(() => {
    if (!("geolocation" in navigator)) setGeo("unsupported");
    if (typeof DeviceMotionEvent === "undefined") setMotion("unsupported");
    if (typeof DeviceOrientationEvent === "undefined") setOrient("unsupported");
    return () => {
      if (watchRef.current != null) navigator.geolocation.clearWatch(watchRef.current);
      if (onMotion.current) window.removeEventListener("devicemotion", onMotion.current);
      if (onOrient.current) window.removeEventListener("deviceorientation", onOrient.current);
    };
  }, []);

  async function activar() {
    setStarted(true);

    if ("geolocation" in navigator) {
      setGeo("waiting");
      watchRef.current = navigator.geolocation.watchPosition(
        (p) => {
          setCoords(p.coords);
          setGeo("ok");
        },
        () => setGeo("denied"),
        { enableHighAccuracy: true, maximumAge: 1000, timeout: 20000 },
      );
    }

    const DM = DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> };
    try {
      if (typeof DM?.requestPermission === "function") {
        const r = await DM.requestPermission();
        if (r !== "granted") setMotion("denied");
        else attachMotion();
      } else if (typeof DeviceMotionEvent !== "undefined") {
        attachMotion();
      }
    } catch {
      setMotion("denied");
    }

    const DO = DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<string> };
    try {
      if (typeof DO?.requestPermission === "function") {
        const r = await DO.requestPermission();
        if (r !== "granted") setOrient("denied");
        else attachOrient();
      } else if (typeof DeviceOrientationEvent !== "undefined") {
        attachOrient();
      }
    } catch {
      setOrient("denied");
    }
  }

  function attachMotion() {
    setMotion("waiting");
    const handler = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      if (a && a.x != null) {
        setAccel({ a: a.x ?? 0, b: a.y ?? 0, c: a.z ?? 0 });
        setMotion("ok");
      }
      const r = e.rotationRate;
      if (r && r.alpha != null) setRot({ a: r.alpha ?? 0, b: r.beta ?? 0, c: r.gamma ?? 0 });
    };
    onMotion.current = handler;
    window.addEventListener("devicemotion", handler);
  }

  function attachOrient() {
    setOrient("waiting");
    const handler = (e: DeviceOrientationEvent) => {
      setOri({ a: e.alpha ?? 0, b: e.beta ?? 0, c: e.gamma ?? 0 });
      setOrient("ok");
    };
    onOrient.current = handler;
    window.addEventListener("deviceorientation", handler);
  }

  const num = (v: number | null | undefined, d = 2) => (v == null ? "—" : v.toFixed(d));
  const kmh = coords?.speed != null ? (coords.speed * 3.6).toFixed(1) : "—";

  return (
    <div className="mx-auto max-w-2xl p-6">
      <Link to="/" className="mb-4 flex w-fit items-center gap-1 text-ink-low hover:text-ink-mid">
        <ChevronLeft size={18} /> {t("inicio", "home")}
      </Link>
      <h1 className="font-sans text-2xl font-bold text-ink-high">
        {t("Diagnóstico de sensores", "Sensor diagnostics")}
      </h1>
      <p className="mt-1 text-sm text-ink-mid">
        {t(
          "Abre esta página en tu celular y toca Activar para conceder permisos.",
          "Open this page on your phone and tap Enable to grant permissions.",
        )}
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          onClick={activar}
          className="rounded-full bg-brand px-6 py-2.5 font-semibold text-void transition-transform hover:scale-[1.03]"
        >
          {started ? t("Reactivar", "Re-enable") : t("Activar sensores", "Enable sensors")}
        </button>
        <span className="text-xs text-ink-low">
          HTTPS / {t("contexto seguro", "secure context")}:{" "}
          <span style={{ color: window.isSecureContext ? "#2DD4BF" : "#FF6B81" }}>
            {window.isSecureContext ? "OK" : t("no", "no")}
          </span>
          {" · "}Wake Lock:{" "}
          <span style={{ color: "wakeLock" in navigator ? "#2DD4BF" : "#FF6B81" }}>
            {"wakeLock" in navigator ? "OK" : "—"}
          </span>
        </span>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <Card title={t("Ubicación (GPS)", "Location (GPS)")} status={geo} label={STATUS_LABEL[geo]}>
          <Row k={t("latitud", "latitude")} v={num(coords?.latitude, 6)} />
          <Row k={t("longitud", "longitude")} v={num(coords?.longitude, 6)} />
          <Row k={t("precisión (m)", "accuracy (m)")} v={num(coords?.accuracy, 1)} />
          <Row k={t("velocidad (km/h)", "speed (km/h)")} v={kmh} />
          <Row k={t("rumbo (°)", "heading (°)")} v={num(coords?.heading, 0)} />
          <Row k={t("altitud (m)", "altitude (m)")} v={num(coords?.altitude, 1)} />
        </Card>

        <Card title={t("Acelerómetro", "Accelerometer")} status={motion} label={STATUS_LABEL[motion]}>
          <Row k="x (m/s²)" v={num(accel?.a)} />
          <Row k="y (m/s²)" v={num(accel?.b)} />
          <Row k="z (m/s²)" v={num(accel?.c)} />
        </Card>

        <Card title={t("Giroscopio", "Gyroscope")} status={motion} label={STATUS_LABEL[motion]}>
          <Row k="α (°/s)" v={num(rot?.a)} />
          <Row k="β (°/s)" v={num(rot?.b)} />
          <Row k="γ (°/s)" v={num(rot?.c)} />
        </Card>

        <Card title={t("Orientación / brújula", "Orientation / compass")} status={orient} label={STATUS_LABEL[orient]}>
          <Row k={t("α brújula (°)", "α compass (°)")} v={num(ori?.a, 0)} />
          <Row k="β (°)" v={num(ori?.b, 0)} />
          <Row k="γ (°)" v={num(ori?.c, 0)} />
        </Card>
      </div>

      <p className="mt-6 text-xs text-ink-low">{navigator.userAgent}</p>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 text-sm">
      <span className="text-ink-low">{k}</span>
      <span className="tabular-nums text-ink-high">{v}</span>
    </div>
  );
}

function Card({
  title,
  status,
  label,
  children,
}: {
  title: string;
  status: Status;
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-hairline bg-panel p-5">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-semibold text-ink-high">{title}</h2>
        <span
          className="flex items-center gap-2 text-xs uppercase tracking-widest"
          style={{ color: STATUS_COLOR[status] }}
        >
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: STATUS_COLOR[status] }} />
          {label}
        </span>
      </div>
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}
