#!/usr/bin/env python3
"""Corre el SurveillancePipeline del equipo en la laptop y alimenta la nube.

Headless: no crea sesion, pola /current-session y cuando la app inicia el viaje
se engancha y postea estados, incidentes y clips. No modifica el codigo del equipo.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

QRO = (20.5888, -100.3899)
AI_W, AI_H = 640, 480

# Colores de estado en BGR (para el overlay de --show).
STATE_BGR = {"Alert": (191, 212, 45), "Drowsy": (32, 176, 255), "Distracted": (129, 107, 255)}

# Eventos del equipo -> nuestras 3 clases del contrato de la nube.
EVENT_TO_STATE = {
    "drowsy": "Drowsy",
    "distracted": "Distracted",
    "eyes_off": "Distracted",
    "phone": "Distracted",
    "food": "Distracted",
    "danger": "Distracted",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"{}")


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as r:
        return json.loads(r.read() or b"{}")


# POST en un hilo daemon: NO bloquea el loop del pipeline (clave para no perder fps
# al postear estado seguido). Guarda el último RTT para el diagnóstico en vivo.
def post_async(base: str, path: str, body: dict, rtt_holder: dict | None = None) -> None:
    def run() -> None:
        t = time.time()
        try:
            post(base, path, body)
            if rtt_holder is not None:
                rtt_holder["ms"] = (time.time() - t) * 1000.0
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def upload_snapshot(base: str, jpg_bytes: bytes) -> str | None:
    try:
        meta = post(base, "/uploads/presign", {"ext": "jpg"})
        ct = meta.get("content_type", "image/jpeg")
        req = urllib.request.Request(meta["url"], data=jpg_bytes, headers={"Content-Type": ct}, method="PUT")
        urllib.request.urlopen(req, timeout=10)
        return meta["key"]
    except Exception as exc:  # noqa: BLE001
        print("  (snapshot falló:", exc, ")")
        return None


def upload_clip(base: str, path: Path) -> str | None:
    """Sube un MP4 (clip ±5s del equipo) a S3 vía presigned PUT."""
    try:
        data = path.read_bytes()
        meta = post(base, "/uploads/presign", {"ext": "mp4"})
        ct = meta.get("content_type", "video/mp4")
        req = urllib.request.Request(meta["url"], data=data, headers={"Content-Type": ct}, method="PUT")
        urllib.request.urlopen(req, timeout=30)
        return meta["key"]
    except Exception as exc:  # noqa: BLE001
        print("  (clip falló:", exc, ")")
        return None


# Nombre de clip del equipo: {YYYYMMDD_HHMMSS_mmm}_{event_type}_{NN}pct.mp4
# (event_type puede traer "_", p.ej. eyes_off) -> .+ greedy hasta _NNpct.
_CLIP_RE = re.compile(r"^\d{8}_\d{6}_\d{3}_(.+)_\d+pct$")


def clip_event_type(path: Path) -> str | None:
    m = _CLIP_RE.match(path.stem)
    return m.group(1) if m else None


def build_pipeline(repo: Path, selfie: bool, fps: float):
    """Construye el SurveillancePipeline del equipo igual que su dashboard."""
    import yaml

    def load_yaml(p: Path) -> dict:
        return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}

    dms_cfg = load_yaml(repo / "config" / "config.yaml")
    yolo_full = load_yaml(repo / "config" / "yolo_config.yaml")
    yolo_cfg = yolo_full.get("yolo", {})
    yolo_cfg["selfie"] = selfie
    dms_cfg.setdefault("features", {})["selfie"] = selfie
    # fps REAL del procesamiento (no los 30 por default). En external-capture el
    # pipeline usa features.fps para: (a) escribir los clips MP4 (si no, los graba a
    # 30fps con frames a mas o menos 9fps y duplica frames de forma inconsistente), (b) las
    # ventanas temporales de features (PERCLOS/parpadeo). Matchearlo arregla ambos.
    dms_cfg["features"]["fps"] = fps

    models_dir = repo / "models"
    onnx_path = models_dir / "driver_state_net.onnx"
    mp_model = models_dir / "face_landmarker_v2_with_blendshapes.task"
    fcfg = json.loads((models_dir / "feature_config.json").read_text(encoding="utf-8"))
    seq_len = int(fcfg["model"]["seq_len"])

    spec = importlib.util.spec_from_file_location(
        "surveillance_custom", str(repo / "scripts" / "surveillance_custom.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pipeline = mod.SurveillancePipeline(
        onnx_path=onnx_path,
        mediapipe_model_path=mp_model,
        yolo_cfg=yolo_cfg,
        events_cfg=yolo_full.get("events", {}),
        clips_cfg=yolo_full.get("clips", {}),
        logging_cfg=yolo_full.get("logging_out", {}),
        dms_cfg=dms_cfg,
        source=None,
        seq_len=seq_len,
        display=False,
        project_root=repo,
        backend="cpu",
        hef_path=None,
        frame_w=AI_W,
        frame_h=AI_H,
    )
    norm = fcfg.get("normalisation", {})
    if "mean" in norm and "std" in norm:
        pipeline.set_normalisation_stats(
            np.array(norm["mean"], dtype=np.float32),
            np.array(norm["std"], dtype=np.float32),
        )
    return pipeline


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://34-205-126-89.nip.io/api")
    ap.add_argument("--repo", default="D:/TEC/LEIA/Attention-Algorithm-model")
    ap.add_argument("--source", default="0")
    ap.add_argument("--no-selfie", action="store_true", help="cámara no-selfie (default: selfie)")
    ap.add_argument("--fake-gps", action="store_true", help="track sintético si no hay GPS del celular")
    ap.add_argument("--show", action="store_true", help="ventana local con la detección en vivo")
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--state-every", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=9.0,
                    help="fps real de procesamiento (clips + ventanas temporales). Default 9 (laptop CPU).")
    ap.add_argument("--seconds", type=float, default=0.0)
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    import cv2  # noqa: E402

    print(f"Cargando pipeline del equipo (YOLO + MediaPipe + LSTM)… [fps={args.fps}]")
    pipeline = build_pipeline(repo, selfie=not args.no_selfie, fps=args.fps)

    # Directorio donde el pipeline del equipo escribe los clips MP4 (±5s por evento).
    # Vigilamos archivos NUEVOS de esta corrida; los previos quedan marcados como vistos.
    try:
        clip_dir = Path(getattr(pipeline, "_clip_writer")._output_dir)
    except Exception:
        clip_dir = repo / "output" / "clips"
    seen_clips: set[str] = {p.name for p in clip_dir.glob("*.mp4")} if clip_dir.exists() else set()
    print(f"Vigilando clips en: {clip_dir}  ({len(seen_clips)} previos ignorados)")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        print("No se pudo abrir la cámara/fuente:", source)
        return

    web = args.base[:-4] if args.base.endswith("/api") else args.base
    print(f"Cámara lista ({args.name}). Headless, esperando que la app inicie un viaje…\n")

    active_sid: str | None = None
    last_poll = last_hb = last_state_post = last_track_post = last_clip_scan = last_diag = 0.0
    poll_every = 1.0
    state_rtt = {"ms": 0.0}  # último RTT del POST de estado (para diagnóstico)
    # Incidentes posteados esperando que su clip MP4 cierre (como 5s post-evento).
    # Cada uno: {"id", "event_type", "ts"}; correlacionamos por event_type.
    pending_clips: list[dict] = []
    gps = list(QRO)
    frame_i = 0
    t0 = time.time()
    state, conf = "Alert", 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.seconds and (time.time() - t0) >= args.seconds:
                break
            ai = cv2.resize(frame, (AI_W, AI_H))
            ts_ms = int((time.time() - t0) * 1000)
            tel = pipeline.process_frame(ai, ts_ms)
            state = tel.get("state", "Alert")
            probs = tel.get("probs", {})
            conf = float(probs.get(state.lower(), 0.0))
            triggers = tel.get("triggers", []) or []
            frame_i += 1

            now = time.time()
            if now - last_hb >= 3.0:
                last_hb = now
                post_async(args.base, "/devices/heartbeat", {"name": args.name, "kind": "camera"})

            if now - last_poll >= poll_every:
                last_poll = now
                try:
                    cs = get(args.base, "/current-session")
                    sid = cs.get("session_id") if cs.get("active") else None
                except Exception:
                    sid = active_sid
                if sid != active_sid:
                    if sid:
                        active_sid = sid
                        last_state_post = last_track_post = 0.0
                        gps = list(QRO)
                        pending_clips.clear()
                        # Cualquier clip ya existente NO es de este viaje -> ignorarlo.
                        if clip_dir.exists():
                            seen_clips.update(p.name for p in clip_dir.glob("*.mp4"))
                        # Auto-calibrar: al iniciar el viaje, la pose actual = neutral/alerta.
                        try:
                            pipeline.calibrate()
                            print("  [calibrado: pose neutral tomada al iniciar el viaje]")
                        except Exception:
                            pass
                        print(f"\n[viaje detectado] {active_sid}  ->  {web}/manager/viajes/{active_sid}")
                    else:
                        print(f"\n[viaje terminado] {active_sid}")
                        active_sid = None
                        pending_clips.clear()

            if active_sid:
                tsnow = now_iso()
                if (now - last_state_post) >= args.state_every:
                    post_async(args.base, f"/sessions/{active_sid}/states",
                               {"ts": tsnow, "state": state, "confidence": conf}, state_rtt)
                    last_state_post = now
                if args.fake_gps and (now - last_track_post) >= 2.0:
                    gps[0] += 0.00018
                    gps[1] += 0.00022
                    post(args.base, f"/sessions/{active_sid}/track",
                         {"ts": tsnow, "lat": gps[0], "lng": gps[1], "speed_kmh": 60})
                    last_track_post = now
                # Cada trigger del motor de eventos del equipo -> un incidente en la nube.
                for evt in triggers:
                    inc_state = EVENT_TO_STATE.get(evt, "Distracted")
                    snap_key = None
                    enc_ok, jpg = cv2.imencode(".jpg", ai)
                    if enc_ok:
                        snap_key = upload_snapshot(args.base, jpg.tobytes())
                    resp = post(args.base, f"/sessions/{active_sid}/incidents", {
                        "ts": tsnow, "state": inc_state,
                        "confidence": float(probs.get(inc_state.lower(), conf)) or 0.9,
                        "speed_kmh": 60 if args.fake_gps else None,
                        "lat": gps[0] if args.fake_gps else None,
                        "lng": gps[1] if args.fake_gps else None,
                        "snapshot_key": snap_key,
                        "event_type": evt,
                        "harsh_event": False,
                    })
                    inc_id = resp.get("incident_id")
                    if inc_id:
                        pending_clips.append({"id": inc_id, "event_type": evt, "ts": now})
                    print(f"\n  [!] {evt} -> incidente {inc_state}{' +foto' if snap_key else ''}")

                # El clip MP4 cierra mas o menos 5s después del evento (post-buffer). Vigilamos
                # el directorio: clip nuevo + estable -> subir y asociar al incidente.
                if (now - last_clip_scan) >= 1.5 and clip_dir.exists():
                    last_clip_scan = now
                    pending_clips[:] = [p for p in pending_clips if now - p["ts"] < 30.0]
                    for cp in sorted(clip_dir.glob("*.mp4")):
                        if cp.name in seen_clips:
                            continue
                        try:
                            st = cp.stat()
                        except OSError:
                            continue
                        # Estable: el worker async ya terminó de escribir (sin cambios >2s).
                        if st.st_size == 0 or (now - st.st_mtime) < 2.0:
                            continue
                        evt_type = clip_event_type(cp)
                        match = next(
                            (p for p in reversed(pending_clips)
                             if p["event_type"] == evt_type and not p.get("matched")),
                            None,
                        )
                        if match is None:
                            if (now - st.st_mtime) > 15.0:
                                seen_clips.add(cp.name)  # huérfano: lo soltamos
                            continue
                        key = upload_clip(args.base, cp)
                        seen_clips.add(cp.name)
                        if key:
                            try:
                                post(args.base, f"/sessions/{active_sid}/incidents/{match['id']}/clip",
                                     {"key": key})
                                match["matched"] = True
                                print(f"\n  [video] {cp.name} -> incidente {match['id'][:8]} asociado")
                            except Exception as exc:  # noqa: BLE001
                                print("  (asociar clip falló:", exc, ")")

            # Diagnóstico de latencia (P1): fps del pipeline (cuello candidato en CPU)
            # + RTT del POST de estado (red). Cada 5s.
            if now - last_diag >= 5.0:
                last_diag = now
                fps = float(tel.get("fps", 0.0))
                print(f"\n[diag] pipeline {fps:.1f} fps | state cada {args.state_every}s "
                      f"| post RTT {state_rtt['ms']:.0f}ms aprox")

            tag = f"viaje {active_sid[:8]}" if active_sid else "esperando viaje"
            objs = ",".join(tel.get("objects", {}).keys())
            fps_now = float(tel.get("fps", 0.0))
            print(f"\r  {state:<11} {conf:.2f}  [{tag}]  {fps_now:.0f}fps  obj:{objs:<14} f{frame_i}   ",
                  end="", flush=True)

            if args.show:
                disp = ai.copy()
                db, fb = tel.get("driver_box"), tel.get("face_box")
                if db:
                    cv2.rectangle(disp, (db[0], db[1]), (db[2], db[3]), (180, 180, 180), 2)
                if fb:
                    cv2.rectangle(disp, (fb[0], fb[1]), (fb[2], fb[3]), (252, 169, 61), 2)
                cv2.putText(disp, f"{state} {conf:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, STATE_BGR.get(state, (200, 200, 200)), 2)
                cv2.putText(disp, "EN VIVO -> nube" if active_sid else "esperando viaje (celular)",
                            (10, AI_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (191, 212, 45) if active_sid else (150, 150, 150), 2)
                y = 58
                for name, score in tel.get("objects", {}).items():
                    cv2.putText(disp, f"{name}: {score}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y += 24
                if triggers:
                    cv2.putText(disp, "EVENTO: " + ",".join(triggers), (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("rpi_live - deteccion en vivo (q=salir)", disp)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print("\nRPi detenida.")


if __name__ == "__main__":
    main()
