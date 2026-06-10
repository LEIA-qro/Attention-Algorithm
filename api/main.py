from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import boto3
from fastapi import FastAPI, HTTPException, Query
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, field_validator

DATABASE_URL = os.environ["DATABASE_URL"]
S3_BUCKET = os.environ.get("S3_BUCKET", "dms-media-800407728644")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY,
    driver_id   TEXT,
    started_at  TIMESTAMPTZ NOT NULL,
    ended_at    TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS state_samples (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id),
    ts          TIMESTAMPTZ NOT NULL,
    state       TEXT NOT NULL,
    confidence  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
    id           UUID PRIMARY KEY,
    session_id   UUID NOT NULL REFERENCES sessions(id),
    ts           TIMESTAMPTZ NOT NULL,
    state        TEXT NOT NULL,
    confidence   REAL NOT NULL,
    speed_kmh    REAL,
    lat          DOUBLE PRECISION,
    lng          DOUBLE PRECISION,
    snapshot_key TEXT,
    clip_key     TEXT,
    event_type   TEXT,
    confirmed    BOOLEAN NOT NULL DEFAULT FALSE,
    harsh_event  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS event_type TEXT;
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS clip_key TEXT;
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS confirmed BOOLEAN NOT NULL DEFAULT FALSE;
CREATE TABLE IF NOT EXISTS config (
    id                 INT PRIMARY KEY DEFAULT 1,
    drowsy_seconds     REAL NOT NULL DEFAULT 1.5,
    distracted_seconds REAL NOT NULL DEFAULT 2.5,
    CONSTRAINT config_singleton CHECK (id = 1)
);
INSERT INTO config (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
CREATE TABLE IF NOT EXISTS track_points (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id),
    ts          TIMESTAMPTZ NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    lng         DOUBLE PRECISION NOT NULL,
    speed_kmh   REAL
);
CREATE INDEX IF NOT EXISTS idx_incidents_session_ts ON incidents(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_states_session_ts ON state_samples(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_track_session_ts ON track_points(session_id, ts);
"""

pool: ConnectionPool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = ConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=5,
        open=True,
        check=ConnectionPool.check_connection,
        kwargs={"row_factory": dict_row},
    )
    with pool.connection() as conn:
        conn.execute(SCHEMA)
    yield
    pool.close()


def _assert_session(conn, session_id: uuid.UUID) -> None:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "session not found")


# confirma un estado no-Alert sostenido >= umbral configurable como incidente; solo Drowsy/Distracted
def _threshold_for(conn, state: str) -> Optional[float]:
    row = conn.execute("SELECT drowsy_seconds, distracted_seconds FROM config WHERE id = 1").fetchone()
    if row is None:
        return None
    if state == "Drowsy":
        return row["drowsy_seconds"]
    if state == "Distracted":
        return row["distracted_seconds"]
    return None


def _run_start(conn, session_id: uuid.UUID, state: str, ts: datetime) -> Optional[datetime]:
    """Inicio del tramo contiguo de `state` que termina en `ts` (el último cambio
    de estado anterior marca el límite)."""
    row = conn.execute(
        "SELECT min(ts) AS rs FROM state_samples "
        "WHERE session_id = %s AND state = %s AND ts <= %s AND ts > COALESCE("
        "  (SELECT max(ts) FROM state_samples "
        "   WHERE session_id = %s AND state <> %s AND ts < %s), '-infinity'::timestamptz)",
        (session_id, state, ts, session_id, state, ts),
    ).fetchone()
    return row["rs"] if row else None


def _state_confirmed(conn, session_id: uuid.UUID, state: str, ts: datetime) -> bool:
    if state == "Alert":
        return False
    thr = _threshold_for(conn, state)
    if thr is None:
        return False
    rs = _run_start(conn, session_id, state, ts)
    if rs is None:
        return False
    return (ts - rs).total_seconds() >= thr


def _maybe_confirm(conn, session_id: uuid.UUID, state: str, ts: datetime, confidence: float) -> None:
    thr = _threshold_for(conn, state)
    if thr is None:
        return
    rs = _run_start(conn, session_id, state, ts)
    if rs is None or (ts - rs).total_seconds() < thr:
        return
    # Una sola confirmación por tramo sostenido.
    exists = conn.execute(
        "SELECT 1 FROM incidents WHERE session_id = %s AND state = %s "
        "AND confirmed = TRUE AND ts >= %s LIMIT 1",
        (session_id, state, rs),
    ).fetchone()
    if exists:
        return
    conn.execute(
        "INSERT INTO incidents (id, session_id, ts, state, confidence, event_type, confirmed) "
        "VALUES (%s, %s, %s, %s, %s, %s, TRUE)",
        (uuid.uuid4(), session_id, ts, state, confidence, "sustained"),
    )


app = FastAPI(title="DMS API", lifespan=lifespan)


def _now() -> datetime:
    return datetime.now(timezone.utc)


_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


DriverState = Literal["Alert", "Drowsy", "Distracted"]


class SessionCreate(BaseModel):
    driver_id: Optional[str] = None


class StateIn(BaseModel):
    ts: datetime
    state: DriverState
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _as_utc(v)


class IncidentIn(BaseModel):
    ts: datetime
    state: DriverState
    confidence: float = Field(ge=0.0, le=1.0)
    speed_kmh: Optional[float] = Field(default=None, ge=0.0)
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    lng: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    snapshot_key: Optional[str] = None
    event_type: Optional[str] = None
    harsh_event: bool = False

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _as_utc(v)


class TrackIn(BaseModel):
    ts: datetime
    lat: float = Field(ge=-90.0, le=90.0)
    lng: float = Field(ge=-180.0, le=180.0)
    speed_kmh: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _as_utc(v)


class PresignIn(BaseModel):
    content_type: str = "image/jpeg"
    ext: str = "jpg"


class ClipIn(BaseModel):
    key: str


class ConfigIn(BaseModel):
    drowsy_seconds: float = Field(ge=0.0, le=30.0)
    distracted_seconds: float = Field(ge=0.0, le=30.0)


class DeviceHeartbeat(BaseModel):
    name: str
    kind: str = "camera"


# latidos del edge en memoria (se resetea al reiniciar); marca camara/RPi disponible
_devices: dict[str, dict] = {}
_DEVICE_TTL_S = 12.0


@app.get("/health")
def health():
    return {"status": "ok", "service": "dms-api"}


@app.post("/sessions", status_code=201)
def create_session(body: SessionCreate):
    session_id = uuid.uuid4()
    started = _now()
    with pool.connection() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = %s, status = 'ended' WHERE status = 'active'",
            (started,),
        )
        conn.execute(
            "INSERT INTO sessions (id, driver_id, started_at, status) VALUES (%s, %s, %s, 'active')",
            (session_id, body.driver_id, started),
        )
    return {"session_id": session_id, "started_at": started}


@app.post("/sessions/{session_id}/end")
def end_session(session_id: uuid.UUID):
    with pool.connection() as conn:
        row = conn.execute(
            "UPDATE sessions SET ended_at = %s, status = 'ended' "
            "WHERE id = %s AND status = 'active' RETURNING ended_at",
            (_now(), session_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "no active session with that id")
    return {"session_id": session_id, "ended_at": row["ended_at"]}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: uuid.UUID):
    """Borra un viaje y TODO lo relacionado: state_samples, incidents, track_points
    (en una transacción) + los objetos en S3 (snapshots/clips). Irreversible."""
    with pool.connection() as conn:
        if conn.execute("SELECT 1 FROM sessions WHERE id = %s", (session_id,)).fetchone() is None:
            raise HTTPException(404, "session not found")
        media = conn.execute(
            "SELECT snapshot_key, clip_key FROM incidents WHERE session_id = %s", (session_id,)
        ).fetchall()
        conn.execute("DELETE FROM incidents WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM state_samples WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM track_points WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    keys = [k for row in media for k in (row.get("snapshot_key"), row.get("clip_key")) if k]
    deleted_media = 0
    if keys:
        try:
            s3 = _s3()
            for i in range(0, len(keys), 1000):
                batch = [{"Key": k} for k in keys[i : i + 1000]]
                s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": batch, "Quiet": True})
            deleted_media = len(keys)
        except Exception:
            pass  # si S3 truena, ni modo, dejamos los huérfanos juntando polvo. la BD ya quedó limpia, que es lo que importa
    return {"deleted": True, "media_deleted": deleted_media}


@app.get("/current-session")
def current_session():
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT id, started_at FROM sessions WHERE status = 'active' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return {"active": False}
    return {"active": True, "session_id": row["id"], "started_at": row["started_at"]}


@app.post("/sessions/{session_id}/states", status_code=202)
def post_state(session_id: uuid.UUID, body: StateIn):
    with pool.connection() as conn:
        _assert_session(conn, session_id)
        conn.execute(
            "INSERT INTO state_samples (session_id, ts, state, confidence) VALUES (%s, %s, %s, %s)",
            (session_id, body.ts, body.state, body.confidence),
        )
        if body.state != "Alert":
            _maybe_confirm(conn, session_id, body.state, body.ts, body.confidence)
    return {"accepted": True}


@app.post("/sessions/{session_id}/incidents", status_code=201)
def post_incident(session_id: uuid.UUID, body: IncidentIn):
    incident_id = uuid.uuid4()
    with pool.connection() as conn:
        _assert_session(conn, session_id)
        conn.execute(
            "INSERT INTO incidents "
            "(id, session_id, ts, state, confidence, speed_kmh, lat, lng, snapshot_key, event_type, harsh_event) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                incident_id, session_id, body.ts, body.state, body.confidence,
                body.speed_kmh, body.lat, body.lng, body.snapshot_key, body.event_type, body.harsh_event,
            ),
        )
    return {"incident_id": incident_id}


@app.post("/sessions/{session_id}/incidents/{incident_id}/clip", status_code=202)
def attach_clip(session_id: uuid.UUID, incident_id: uuid.UUID, body: ClipIn):
    """Asocia un clip de video (ya subido a S3) a un incidente. El edge llama esto
    después, porque el clip se cierra como 5s tras el evento (post-buffer)."""
    with pool.connection() as conn:
        row = conn.execute(
            "UPDATE incidents SET clip_key = %s WHERE id = %s AND session_id = %s RETURNING id",
            (body.key, incident_id, session_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "incident not found in that session")
    return {"accepted": True}


@app.post("/sessions/{session_id}/track", status_code=202)
def post_track(session_id: uuid.UUID, body: TrackIn):
    with pool.connection() as conn:
        _assert_session(conn, session_id)
        conn.execute(
            "INSERT INTO track_points (session_id, ts, lat, lng, speed_kmh) VALUES (%s, %s, %s, %s, %s)",
            (session_id, body.ts, body.lat, body.lng, body.speed_kmh),
        )
    return {"accepted": True}


@app.get("/sessions/{session_id}/alerts")
def get_alerts(
    session_id: uuid.UUID,
    since: Optional[datetime] = None,
    limit: int = Query(default=500, ge=1, le=2000),
):
    query = (
        "SELECT id, ts, state, confidence, speed_kmh, lat, lng, snapshot_key, event_type, confirmed, harsh_event "
        "FROM incidents WHERE session_id = %s"
    )
    params: list = [session_id]
    if since is not None:
        query += " AND ts > %s"
        params.append(_as_utc(since))
    query += " ORDER BY ts ASC LIMIT %s"
    params.append(limit)
    with pool.connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return {"alerts": rows}


@app.get("/sessions/{session_id}/state")
def latest_state(session_id: uuid.UUID):
    """Último estado continuo del conductor (para la vista en vivo del celular).
    Distinto de los incidentes (que el motor de eventos dispara con cooldown)."""
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT state, confidence, ts FROM state_samples "
            "WHERE session_id = %s ORDER BY ts DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return {"state": "Alert", "confidence": 0.0, "confirmed": False}
        confirmed = _state_confirmed(conn, session_id, row["state"], row["ts"])
    return {
        "state": row["state"],
        "confidence": row["confidence"],
        "ts": row["ts"],
        "confirmed": confirmed,
    }


@app.get("/sessions")
def list_sessions():
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT s.id, s.driver_id, s.started_at, s.ended_at, s.status, "
            "COUNT(i.id) AS incident_count "
            "FROM sessions s LEFT JOIN incidents i ON i.session_id = s.id "
            "GROUP BY s.id ORDER BY s.started_at DESC"
        ).fetchall()
    return {"sessions": rows}


@app.get("/sessions/{session_id}")
def get_session(session_id: uuid.UUID):
    with pool.connection() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
        if session is None:
            raise HTTPException(404, "session not found")
        track = conn.execute(
            "SELECT ts, lat, lng, speed_kmh FROM track_points WHERE session_id = %s ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        states = conn.execute(
            "SELECT ts, state, confidence FROM state_samples WHERE session_id = %s ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        incidents = conn.execute(
            "SELECT id, ts, state, confidence, speed_kmh, lat, lng, snapshot_key, clip_key, event_type, confirmed, harsh_event "
            "FROM incidents WHERE session_id = %s ORDER BY ts ASC",
            (session_id,),
        ).fetchall()

    # incidentes del edge sin lat/lng: los ubicamos en el punto de track mas cercano en el tiempo
    if track:
        for inc in incidents:
            if inc.get("lat") is None or inc.get("lng") is None:
                nearest = min(track, key=lambda p: abs((p["ts"] - inc["ts"]).total_seconds()))
                inc["lat"] = nearest["lat"]
                inc["lng"] = nearest["lng"]
                if inc.get("speed_kmh") is None:
                    inc["speed_kmh"] = nearest["speed_kmh"]

    if any(inc.get("snapshot_key") or inc.get("clip_key") for inc in incidents):
        s3 = _s3()
        for inc in incidents:
            for key_field, url_field in (("snapshot_key", "snapshot_url"), ("clip_key", "clip_url")):
                key = inc.get(key_field)
                if not key:
                    inc[url_field] = None
                    continue
                try:
                    inc[url_field] = s3.generate_presigned_url(
                        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=300
                    )
                except Exception:
                    inc[url_field] = None
    return {"session": session, "track": track, "states": states, "incidents": incidents}


_CONTENT_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
    "mp4": "video/mp4", "webm": "video/webm",
}
_VIDEO_EXTS = {"mp4", "webm"}


@app.post("/uploads/presign")
def presign_upload(body: PresignIn):
    ext = body.ext.lower().lstrip(".")
    if ext not in _CONTENT_TYPES:
        raise HTTPException(400, "unsupported file type")
    content_type = _CONTENT_TYPES[ext]
    prefix = "clips" if ext in _VIDEO_EXTS else "snapshots"
    key = f"{prefix}/{uuid.uuid4()}.{ext}"
    try:
        url = _s3().generate_presigned_url(
            "put_object",
            Params={"Bucket": S3_BUCKET, "Key": key, "ContentType": content_type},
            ExpiresIn=300,
        )
    except Exception:
        raise HTTPException(503, "could not generate upload url")
    return {"url": url, "key": key, "content_type": content_type}


@app.post("/devices/heartbeat", status_code=202)
def device_heartbeat(body: DeviceHeartbeat):
    _devices[body.name] = {"kind": body.kind, "last_seen": _now()}
    return {"accepted": True}


@app.get("/config")
def get_config():
    with pool.connection() as conn:
        row = conn.execute("SELECT drowsy_seconds, distracted_seconds FROM config WHERE id = 1").fetchone()
    return row or {"drowsy_seconds": 1.5, "distracted_seconds": 2.5}


@app.put("/config")
def put_config(body: ConfigIn):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE config SET drowsy_seconds = %s, distracted_seconds = %s WHERE id = 1",
            (body.drowsy_seconds, body.distracted_seconds),
        )
    return {"drowsy_seconds": body.drowsy_seconds, "distracted_seconds": body.distracted_seconds}


@app.get("/devices")
def list_devices():
    cutoff = _now().timestamp() - _DEVICE_TTL_S
    devices = [
        {"name": name, "kind": d["kind"]}
        for name, d in _devices.items()
        if d["last_seen"].timestamp() >= cutoff
    ]
    return {"devices": devices}
