import type {
  CurrentSession,
  DriverState,
  Incident,
  SessionDetail,
  SessionListItem,
  StateSample,
  TrackPoint,
} from "./types";

const DEMO_SESSION = "demo-trip-0001";
const START = Date.now();

const SCRIPT: DriverState[] = [
  "Alert",
  "Alert",
  "Alert",
  "Drowsy",
  "Alert",
  "Distracted",
  "Alert",
  "Drowsy",
  "Drowsy",
  "Alert",
];

export function demoLiveState(): { state: DriverState; confidence: number; confirmed: boolean } {
  const elapsed = Date.now() - START;
  const i = Math.floor(elapsed / 3500) % SCRIPT.length;
  const state = SCRIPT[i];
  // Gentle drift so the attention trace breathes instead of flatlining on Alert.
  const wobble = (Math.sin(elapsed / 1700) + 1) / 2; // 0..1
  const confidence =
    state === "Alert" ? 0.66 + wobble * 0.1 : 0.84 + (i % 3) * 0.04 + wobble * 0.03;
  // En demo, "confirma" cuando el estado no-Alert lleva >1 paso del script.
  const confirmed = state !== "Alert" && i > 0 && SCRIPT[i - 1] === state;
  return { state, confidence: Math.min(0.98, confidence), confirmed };
}

export function currentSession(): CurrentSession {
  return { active: true, session_id: DEMO_SESSION, started_at: new Date(START).toISOString() };
}

export function listSessions(): SessionListItem[] {
  return [
    { id: DEMO_SESSION, driver_id: "Laura Mendoza", started_at: new Date(START).toISOString(), ended_at: null, status: "active", incident_count: 3 },
    { id: "demo-trip-0000", driver_id: "Carlos Reyes", started_at: "2026-06-05T14:02:00Z", ended_at: "2026-06-05T14:48:00Z", status: "ended", incident_count: 5 },
    { id: "demo-trip-9999", driver_id: null, started_at: "2026-06-04T09:10:00Z", ended_at: "2026-06-04T09:35:00Z", status: "ended", incident_count: 1 },
  ];
}

const T0 = Date.parse("2026-06-05T14:02:00Z");
const STEP = 90_000;

// A path heading NE across Querétaro.
const TRACK: TrackPoint[] = Array.from({ length: 16 }, (_, i) => ({
  ts: new Date(T0 + i * STEP).toISOString(),
  lat: 20.5888 + i * 0.0026 + Math.sin(i / 2) * 0.0014,
  lng: -100.3899 + i * 0.0031 + Math.cos(i / 3) * 0.0012,
  speed_kmh: i === 0 || i === 15 ? 0 : 70 + ((i * 13) % 45),
}));

const STATE_TIMELINE: StateSample[] = Array.from({ length: 30 }, (_, i) => {
  let state: DriverState = "Alert";
  if (i >= 6 && i <= 9) state = "Drowsy";
  else if (i >= 14 && i <= 15) state = "Distracted";
  else if (i >= 21 && i <= 24) state = "Drowsy";
  return {
    ts: new Date(T0 + i * 48_000).toISOString(),
    state,
    confidence: state === "Alert" ? 0.7 : 0.88,
  };
});

// "Frame recuperado" estilo caja negra: el ojo capturado cambia por estado
// (abierto/alerta, párpado caído/somnolencia, mirada desviada/distracción).
function snapFor(state: DriverState): string {
  const eye =
    state === "Drowsy"
      ? "<path d='M130 102 q30 18 60 0' fill='none' stroke='#9DAEC2' stroke-width='4' stroke-linecap='round'/>"
      : state === "Distracted"
        ? "<circle cx='178' cy='100' r='9' fill='#9DAEC2'/><ellipse cx='160' cy='100' rx='30' ry='16' fill='none' stroke='#1B2430' stroke-width='3'/>"
        : "<circle cx='160' cy='100' r='9' fill='#9DAEC2'/><ellipse cx='160' cy='100' rx='30' ry='16' fill='none' stroke='#1B2430' stroke-width='3'/>";
  return (
    "data:image/svg+xml;utf8," +
    encodeURIComponent(
      "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='240'>" +
        "<rect width='320' height='240' fill='#0A0E14'/>" +
        "<circle cx='160' cy='112' r='54' fill='none' stroke='#1B2430' stroke-width='3'/>" +
        eye +
        "<path d='M108 205 q52 -62 104 0' fill='#1B2430'/>" +
        "<text x='12' y='226' fill='#9DAEC2' font-family='monospace' font-size='12'>frame • IMX500</text>" +
        "</svg>",
    )
  );
}

const DEMO_INCIDENTS: Incident[] = [
  { id: "i1", ts: new Date(T0 + 7 * 48_000).toISOString(), state: "Drowsy", confidence: 0.88, speed_kmh: 102, lat: TRACK[4].lat, lng: TRACK[4].lng, snapshot_key: "demo", snapshot_url: snapFor("Drowsy"), harsh_event: false },
  { id: "i2", ts: new Date(T0 + 14 * 48_000).toISOString(), state: "Distracted", confidence: 0.93, speed_kmh: 96, lat: TRACK[8].lat, lng: TRACK[8].lng, snapshot_key: "demo", snapshot_url: snapFor("Distracted"), event_type: "phone", harsh_event: true },
  { id: "i3", ts: new Date(T0 + 22 * 48_000).toISOString(), state: "Drowsy", confidence: 0.81, speed_kmh: 88, lat: TRACK[12].lat, lng: TRACK[12].lng, snapshot_key: "demo", snapshot_url: snapFor("Drowsy"), harsh_event: false },
];

// Deterministic per-session incidents so the count in the list matches the detail.
function incidentsFor(count: number): Incident[] {
  if (count <= DEMO_INCIDENTS.length) return DEMO_INCIDENTS.slice(0, count);
  const extra: Incident[] = Array.from({ length: count - DEMO_INCIDENTS.length }, (_, k) => {
    const base = DEMO_INCIDENTS[k % DEMO_INCIDENTS.length];
    const ti = (3 + k) % TRACK.length;
    return {
      ...base,
      id: `i${DEMO_INCIDENTS.length + k + 1}`,
      ts: new Date(T0 + (26 + k * 4) * 48_000).toISOString(),
      lat: TRACK[ti].lat,
      lng: TRACK[ti].lng,
      harsh_event: k % 2 === 0,
    };
  });
  return [...DEMO_INCIDENTS, ...extra];
}

export function getSession(id: string): SessionDetail {
  const list = listSessions();
  const session = list.find((s) => s.id === id) ?? list[0];
  return {
    session: {
      id: session.id,
      driver_id: session.driver_id,
      started_at: session.started_at,
      ended_at: session.ended_at,
      status: session.status,
    },
    track: TRACK,
    states: STATE_TIMELINE,
    incidents: incidentsFor(session.incident_count),
  };
}

export function alerts(id: string, since?: string): Incident[] {
  const list = listSessions();
  const session = list.find((s) => s.id === id);
  const incidents = session ? incidentsFor(session.incident_count) : DEMO_INCIDENTS;
  if (!since) return incidents;
  return incidents.filter((i) => i.ts > since);
}
