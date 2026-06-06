#!/usr/bin/env python3
"""
10_web_dashboard.py — Production Web Dashboard for Driver Monitoring System
============================================================================

Streams MJPEG video and SSE telemetry from the SurveillancePipeline to a
modern, dark-mode web dashboard. Designed to run headlessly on a Raspberry Pi
and be viewed from any device on the local network.

Usage
-----
    python scripts/10_web_dashboard.py --source 0 --selfie
    python scripts/10_web_dashboard.py --source 0 --selfie --host 0.0.0.0 --port 8080

Then open http://<ip>:5000 in a browser.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
cv2.setNumThreads(2)

import numpy as np
import yaml
import torch
torch.set_num_threads(2)

from flask import Flask, Response, render_template_string

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("dms.web_dashboard")

# ---------------------------------------------------------------------------
# HTML / CSS / JS — embedded for single-file deployment
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driver Monitoring System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-primary: #0a0a0f;
    --bg-card: rgba(18, 18, 28, 0.85);
    --bg-card-solid: #12121c;
    --border: rgba(255, 255, 255, 0.06);
    --text-primary: #e8e8ed;
    --text-secondary: #8a8a9a;
    --text-dim: #5a5a6a;

    --alert-color: #22c55e;
    --alert-bg: rgba(34, 197, 94, 0.08);
    --drowsy-color: #f59e0b;
    --drowsy-bg: rgba(245, 158, 11, 0.08);
    --distracted-color: #ef4444;
    --distracted-bg: rgba(239, 68, 68, 0.08);

    --radius: 12px;
    --radius-sm: 8px;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
}

/* ---- Top bar ---- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--alert-color);
    box-shadow: 0 0 8px var(--alert-color);
    transition: background 0.4s, box-shadow 0.4s;
}
.topbar-dot.drowsy { background: var(--drowsy-color); box-shadow: 0 0 8px var(--drowsy-color); }
.topbar-dot.distracted { background: var(--distracted-color); box-shadow: 0 0 8px var(--distracted-color); }
.topbar-title { font-weight: 600; font-size: 15px; letter-spacing: -0.2px; }
.topbar-right { display: flex; align-items: center; gap: 16px; }
.topbar-fps { font-size: 12px; color: var(--text-dim); font-variant-numeric: tabular-nums; }

/* ---- Layout ---- */
.layout {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 0;
    height: calc(100vh - 57px);
}

/* ---- Video panel ---- */
.video-panel {
    position: relative;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.video-panel img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* State overlay on video */
.state-overlay {
    position: absolute;
    top: 20px;
    left: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: var(--radius);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    transition: background 0.3s, border-color 0.3s;
}
.state-overlay.alert { background: rgba(34, 197, 94, 0.12); border-color: rgba(34, 197, 94, 0.25); }
.state-overlay.drowsy { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.25); }
.state-overlay.distracted { background: rgba(239, 68, 68, 0.12); border-color: rgba(239, 68, 68, 0.25); }

.state-label {
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: color 0.3s;
}
.state-conf {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
}

/* ---- Sidebar ---- */
.sidebar {
    border-left: 1px solid var(--border);
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

/* Cards */
.card {
    background: var(--bg-card-solid);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
}
.card-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-dim);
    margin-bottom: 14px;
}

/* Probability bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.prob-row:last-child { margin-bottom: 0; }
.prob-label {
    width: 76px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
}
.prob-track {
    flex: 1;
    height: 6px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.25s ease-out;
}
.prob-fill.alert { background: var(--alert-color); }
.prob-fill.drowsy { background: var(--drowsy-color); }
.prob-fill.distracted { background: var(--distracted-color); }
.prob-value {
    width: 38px;
    text-align: right;
    font-size: 12px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    color: var(--text-secondary);
}

/* Metrics grid */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.metric-item {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px;
}
.metric-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-dim);
    margin-bottom: 4px;
}
.metric-value {
    font-size: 20px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
}

/* Objects */
.object-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    margin: 0 6px 6px 0;
    border: 1px solid;
}
.object-tag.phone { background: rgba(251, 146, 60, 0.1); border-color: rgba(251, 146, 60, 0.25); color: #fb923c; }
.object-tag.food { background: rgba(52, 211, 153, 0.1); border-color: rgba(52, 211, 153, 0.25); color: #34d399; }
.object-tag.danger { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.25); color: #ef4444; }
.objects-empty { font-size: 12px; color: var(--text-dim); }

/* Events log */
.event-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    animation: fadeIn 0.3s ease;
}
.event-item:last-child { border-bottom: none; }
.event-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.event-dot.phone { background: #fb923c; }
.event-dot.food { background: #34d399; }
.event-dot.danger { background: #ef4444; }
.event-dot.drowsy { background: var(--drowsy-color); }
.event-dot.distracted { background: var(--distracted-color); }
.event-dot.eyes_off { background: #a78bfa; }
.event-text { font-size: 12px; color: var(--text-secondary); }
.events-empty { font-size: 12px; color: var(--text-dim); }

/* Connection indicator */
.conn-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-dim);
}
.conn-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--alert-color);
    transition: background 0.3s;
}
.conn-dot.disconnected { background: var(--distracted-color); }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---- Responsive ---- */
@media (max-width: 800px) {
    .layout {
        grid-template-columns: 1fr;
        grid-template-rows: 50vh 1fr;
        height: auto;
    }
    .sidebar { border-left: none; border-top: 1px solid var(--border); }
}
</style>
</head>
<body>

<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-dot" id="topbar-dot"></div>
        <span class="topbar-title">Driver Monitoring System</span>
    </div>
    <div class="topbar-right">
        <div class="conn-status">
            <div class="conn-dot" id="conn-dot"></div>
            <span id="conn-text">Connecting</span>
        </div>
        <span class="topbar-fps" id="topbar-fps">-- FPS</span>
    </div>
</div>

<div class="layout">
    <div class="video-panel">
        <img id="video-feed" src="/video_feed" alt="Camera feed">
        <div class="state-overlay alert" id="state-overlay">
            <span class="state-label" id="state-label">ALERT</span>
            <span class="state-conf" id="state-conf">--</span>
        </div>
    </div>

    <div class="sidebar">
        <!-- Probabilities -->
        <div class="card">
            <div class="card-header">Classification</div>
            <div class="prob-row">
                <span class="prob-label">Alert</span>
                <div class="prob-track"><div class="prob-fill alert" id="bar-alert" style="width:0%"></div></div>
                <span class="prob-value" id="val-alert">0%</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Drowsy</span>
                <div class="prob-track"><div class="prob-fill drowsy" id="bar-drowsy" style="width:0%"></div></div>
                <span class="prob-value" id="val-drowsy">0%</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Distracted</span>
                <div class="prob-track"><div class="prob-fill distracted" id="bar-distracted" style="width:0%"></div></div>
                <span class="prob-value" id="val-distracted">0%</span>
            </div>
        </div>

        <!-- Metrics -->
        <div class="card">
            <div class="card-header">Biometrics</div>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Yaw</div>
                    <div class="metric-value" id="m-yaw">--</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Pitch</div>
                    <div class="metric-value" id="m-pitch">--</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">EAR</div>
                    <div class="metric-value" id="m-ear">--</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">MAR</div>
                    <div class="metric-value" id="m-mar">--</div>
                </div>
            </div>
        </div>

        <!-- Objects -->
        <div class="card">
            <div class="card-header">Detected Objects</div>
            <div id="objects-container">
                <span class="objects-empty">No objects detected</span>
            </div>
        </div>

        <!-- Events -->
        <div class="card">
            <div class="card-header">Recent Events</div>
            <div id="events-container">
                <span class="events-empty">No events recorded</span>
            </div>
        </div>
    </div>
</div>

<script>
(function() {
    const $ = id => document.getElementById(id);

    const stateColors = {
        Alert: 'alert',
        Drowsy: 'drowsy',
        Distracted: 'distracted'
    };

    const eventLabels = {
        phone: 'Phone detected',
        food: 'Food detected',
        danger: 'Dangerous object',
        drowsy: 'Drowsiness detected',
        distracted: 'Distraction detected',
        eyes_off: 'Eyes off road'
    };

    let eventLog = [];
    let connected = false;

    function setConnected(v) {
        connected = v;
        $('conn-dot').className = 'conn-dot' + (v ? '' : ' disconnected');
        $('conn-text').textContent = v ? 'Connected' : 'Reconnecting';
    }

    function updateUI(d) {
        // State
        const cls = stateColors[d.state] || 'alert';
        const overlay = $('state-overlay');
        overlay.className = 'state-overlay ' + cls;
        $('state-label').textContent = d.state.toUpperCase();

        const maxProb = Math.max(d.probs.alert, d.probs.drowsy, d.probs.distracted);
        $('state-conf').textContent = Math.round(maxProb * 100) + '%';
        $('state-label').style.color = 'var(--' + cls + '-color)';

        // Topbar dot
        $('topbar-dot').className = 'topbar-dot ' + cls;

        // FPS
        $('topbar-fps').textContent = d.fps + ' FPS';

        // Bars
        for (const key of ['alert', 'drowsy', 'distracted']) {
            const pct = Math.round(d.probs[key] * 100);
            $('bar-' + key).style.width = pct + '%';
            $('val-' + key).textContent = pct + '%';
        }

        // Metrics
        if (d.feats) {
            $('m-yaw').textContent = d.feats.yaw + '\u00B0';
            $('m-pitch').textContent = d.feats.pitch + '\u00B0';
            $('m-ear').textContent = d.feats.ear.toFixed(2);
            $('m-mar').textContent = d.feats.mar.toFixed(2);
        }

        // Objects
        const oc = $('objects-container');
        const objKeys = Object.keys(d.objects || {});
        if (objKeys.length === 0) {
            oc.innerHTML = '<span class="objects-empty">No objects detected</span>';
        } else {
            oc.innerHTML = objKeys.map(k => {
                const pct = Math.round(d.objects[k] * 100);
                const tagClass = k === 'phone' ? 'phone' : (k === 'food' ? 'food' : 'danger');
                return '<span class="object-tag ' + tagClass + '">' + k.toUpperCase() + ' ' + pct + '%</span>';
            }).join('');
        }

        // Events — append new triggers
        if (d.triggers && d.triggers.length > 0) {
            for (const t of d.triggers) {
                eventLog.unshift({ type: t, time: new Date() });
            }
            if (eventLog.length > 20) eventLog = eventLog.slice(0, 20);
            renderEvents();
        }
    }

    function renderEvents() {
        const ec = $('events-container');
        if (eventLog.length === 0) {
            ec.innerHTML = '<span class="events-empty">No events recorded</span>';
            return;
        }
        ec.innerHTML = eventLog.map(e => {
            const label = eventLabels[e.type] || e.type;
            const ts = e.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            return '<div class="event-item"><div class="event-dot ' + e.type + '"></div><span class="event-text">' + label + ' &middot; ' + ts + '</span></div>';
        }).join('');
    }

    function connectSSE() {
        const es = new EventSource('/telemetry_feed');
        es.onopen = () => setConnected(true);
        es.onmessage = (ev) => {
            try { updateUI(JSON.parse(ev.data)); } catch(e) {}
        };
        es.onerror = () => {
            setConnected(false);
            es.close();
            setTimeout(connectSSE, 2000);
        };
    }

    connectSSE();
})();
</script>
</body>
</html>"""


def _load_yaml(path: str):
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Globals shared between the pipeline thread and Flask routes
# ---------------------------------------------------------------------------
_latest_jpeg: bytes = b""
_latest_telemetry: dict = {}
_lock = threading.Lock()


def _pipeline_thread(pipeline):
    """Run the surveillance pipeline in a background thread, updating globals."""
    global _latest_jpeg, _latest_telemetry
    for raw_frame, hud_frame, telemetry in pipeline.run_generator():
        _, buf = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with _lock:
            _latest_jpeg = buf.tobytes()
            _latest_telemetry = telemetry


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with _lock:
                frame = _latest_jpeg
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)  # ~30 fps cap
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetry_feed")
def telemetry_feed():
    def gen():
        while True:
            with _lock:
                data = _latest_telemetry.copy() if _latest_telemetry else None
            if data:
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)  # 10 Hz telemetry
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DMS Web Dashboard")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--yolo-config", default="config/yolo_config.yaml")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--mediapipe-model", default=None)
    parser.add_argument("--source", default="0")
    parser.add_argument("--selfie", action="store_true")
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "hailo"])
    parser.add_argument("--hef", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    _setup_logging(args.log_level)

    dms_cfg = _load_yaml(args.config)
    yolo_full = _load_yaml(args.yolo_config)
    yolo_cfg = yolo_full.get("yolo", {})
    events_cfg = yolo_full.get("events", {})
    clips_cfg = yolo_full.get("clips", {})
    logging_cfg = yolo_full.get("logging_out", {})
    paths_cfg = dms_cfg.get("paths", {})
    models_dir = _PROJECT_ROOT / "models"

    selfie = args.selfie or yolo_full.get("yolo", {}).get("selfie", False)
    yolo_cfg["selfie"] = selfie
    if "features" not in dms_cfg:
        dms_cfg["features"] = {}
    dms_cfg["features"]["selfie"] = selfie

    onnx_path = Path(args.onnx) if args.onnx else models_dir / "driver_state_net.onnx"
    mp_model = (
        Path(args.mediapipe_model) if args.mediapipe_model
        else Path(paths_cfg.get("mediapipe_model",
                                str(models_dir / "face_landmarker_v2_with_blendshapes.task")))
    )

    for p, label in [(onnx_path, "ONNX"), (mp_model, "MediaPipe")]:
        if not p.exists():
            logger.error("%s model not found: %s", label, p)
            sys.exit(1)

    source = int(args.source) if args.source.isdigit() else args.source
    hef_path = Path(args.hef) if args.hef else None

    # Dynamically import the pipeline class from 09_surveillance_custom.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "surveillance_custom",
        str(_PROJECT_ROOT / "scripts" / "09_surveillance_custom.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pipeline = mod.SurveillancePipeline(
        onnx_path=onnx_path, mediapipe_model_path=mp_model,
        yolo_cfg=yolo_cfg, events_cfg=events_cfg,
        clips_cfg=clips_cfg, logging_cfg=logging_cfg,
        dms_cfg=dms_cfg, source=source,
        seq_len=args.seq_len, display=False,
        project_root=_PROJECT_ROOT,
        backend=args.backend,
        hef_path=hef_path,
    )

    fc_path = models_dir / "feature_config.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text())
        norm = fc.get("normalisation", {})
        if "mean" in norm and "std" in norm:
            pipeline.set_normalisation_stats(
                np.array(norm["mean"], dtype=np.float32),
                np.array(norm["std"], dtype=np.float32),
            )

    logger.info("=" * 60)
    logger.info("DMS WEB DASHBOARD")
    logger.info("  Source:    %s", args.source)
    logger.info("  Selfie:    %s", selfie)
    logger.info("  Backend:   %s", args.backend)
    logger.info("  Dashboard: http://%s:%d", args.host, args.port)
    logger.info("=" * 60)

    # Start pipeline in background thread
    t = threading.Thread(target=_pipeline_thread, args=(pipeline,), daemon=True)
    t.start()

    # Start Flask
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
