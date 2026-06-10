export type DriverState = "Alert" | "Drowsy" | "Distracted";

export interface Session {
  id: string;
  driver_id: string | null;
  started_at: string;
  ended_at: string | null;
  status: string;
}

export interface SessionListItem extends Session {
  incident_count: number;
}

export interface CurrentSession {
  active: boolean;
  session_id?: string;
  started_at?: string;
}

export interface Incident {
  id: string;
  ts: string;
  state: DriverState;
  confidence: number;
  speed_kmh: number | null;
  lat: number | null;
  lng: number | null;
  snapshot_key: string | null;
  snapshot_url?: string | null;
  clip_key?: string | null;
  clip_url?: string | null;
  event_type?: string | null;
  confirmed?: boolean;
  harsh_event: boolean;
}

export interface LiveState {
  state: DriverState;
  confidence: number;
  confirmed: boolean;
}

export interface ThresholdConfig {
  drowsy_seconds: number;
  distracted_seconds: number;
}

export interface TrackPoint {
  ts: string;
  lat: number;
  lng: number;
  speed_kmh: number | null;
}

export interface StateSample {
  ts: string;
  state: DriverState;
  confidence: number;
}

export interface SessionDetail {
  session: Session;
  track: TrackPoint[];
  states: StateSample[];
  incidents: Incident[];
}
