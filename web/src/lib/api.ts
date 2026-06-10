import * as mock from "./mock";
import type {
  CurrentSession,
  DriverState,
  Incident,
  LiveState,
  SessionDetail,
  SessionListItem,
  ThresholdConfig,
} from "./types";

function resolveDemo(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.has("demo")) {
    try {
      sessionStorage.setItem("dms-demo", "1");
    } catch {
      /* sessionStorage unavailable; fall back to URL flag */
    }
    return true;
  }
  try {
    return sessionStorage.getItem("dms-demo") === "1";
  } catch {
    return false;
  }
}

export const DEMO = resolveDemo();

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { "content-type": "application/json" },
      ...init,
    });
  } catch {
    // Network/DNS/CORS failure — fetch rejects without a status.
    throw new ApiError(0, "Sin conexión con el servidor");
  }

  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText}`.trim());
  }

  // Tolerate empty/no-content bodies (e.g. 204 from POST endpoints).
  if (res.status === 204 || res.headers.get("content-length") === "0") {
    return undefined as T;
  }
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const api = {
  health: () =>
    DEMO ? Promise.resolve({ status: "ok" }) : http<{ status: string }>("/health"),

  currentSession: () =>
    DEMO
      ? Promise.resolve(mock.currentSession())
      : http<CurrentSession>("/current-session"),

  createSession: (driver_id?: string) =>
    DEMO
      ? Promise.resolve({ session_id: "demo-trip-0001", started_at: new Date().toISOString() })
      : http<{ session_id: string; started_at: string }>("/sessions", {
          method: "POST",
          body: JSON.stringify({ driver_id }),
        }),

  endSession: (id: string) =>
    DEMO
      ? Promise.resolve({ ended_at: new Date().toISOString() })
      : http<{ ended_at: string }>(`/sessions/${id}/end`, { method: "POST" }),

  deleteSession: (id: string) =>
    DEMO
      ? Promise.resolve({ deleted: true, media_deleted: 0 })
      : http<{ deleted: boolean; media_deleted: number }>(`/sessions/${id}`, { method: "DELETE" }),

  listSessions: () =>
    DEMO
      ? Promise.resolve({ sessions: mock.listSessions() })
      : http<{ sessions: SessionListItem[] }>("/sessions"),

  getSession: (id: string) =>
    DEMO
      ? Promise.resolve(mock.getSession(id))
      : http<SessionDetail>(`/sessions/${id}`),

  alerts: (id: string, since?: string) =>
    DEMO
      ? Promise.resolve({ alerts: mock.alerts(id, since) })
      : http<{ alerts: Incident[] }>(
          `/sessions/${id}/alerts${since ? `?since=${encodeURIComponent(since)}` : ""}`,
        ),

  latestState: (id: string) =>
    DEMO
      ? Promise.resolve(mock.demoLiveState())
      : http<LiveState>(`/sessions/${id}/state`),

  track: (id: string, body: { ts: string; lat: number; lng: number; speed_kmh?: number | null }) =>
    DEMO
      ? Promise.resolve({ accepted: true })
      : http<{ accepted: boolean }>(`/sessions/${id}/track`, {
          method: "POST",
          body: JSON.stringify(body),
        }),

  // Inyectar estado/incidente — usado por la Zona de pruebas (simula al edge sin cámara).
  postState: (id: string, body: { ts: string; state: DriverState; confidence: number }) =>
    DEMO
      ? Promise.resolve({ accepted: true })
      : http<{ accepted: boolean }>(`/sessions/${id}/states`, {
          method: "POST",
          body: JSON.stringify(body),
        }),

  postIncident: (
    id: string,
    body: {
      ts: string;
      state: DriverState;
      confidence: number;
      event_type?: string;
      speed_kmh?: number | null;
    },
  ) =>
    DEMO
      ? Promise.resolve({ incident_id: "demo" })
      : http<{ incident_id: string }>(`/sessions/${id}/incidents`, {
          method: "POST",
          body: JSON.stringify(body),
        }),

  getConfig: () =>
    DEMO
      ? Promise.resolve({ drowsy_seconds: 1.5, distracted_seconds: 2.5 })
      : http<ThresholdConfig>("/config"),

  putConfig: (body: ThresholdConfig) =>
    DEMO
      ? Promise.resolve(body)
      : http<ThresholdConfig>("/config", { method: "PUT", body: JSON.stringify(body) }),

  devices: () =>
    DEMO
      ? Promise.resolve({ devices: [{ name: "demo-cam", kind: "camera" }] })
      : http<{ devices: { name: string; kind: string }[] }>("/devices"),
};
