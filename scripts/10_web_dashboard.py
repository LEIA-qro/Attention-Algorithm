#!/usr/bin/env python3
"""
10_web_dashboard.py — Production Web Dashboard for Driver Monitoring System
============================================================================

Architecture
------------
Three threads, fully decoupled:

1. **Camera thread** — reads from the webcam at native resolution (e.g. 720p
   or 1080p) at the camera's native frame rate (~30 fps).  Stores the latest
   frame in a shared variable.

2. **Inference thread** — grabs the latest camera frame, downscales it to the
   resolution the AI models were trained on (640×480), passes it through the
   full pipeline (YOLO → MediaPipe → LSTM → heuristics → event engine), and
   stores the resulting telemetry + bounding-box coordinates.

3. **Flask server** — serves two streaming endpoints:
   - ``/video_feed``  MJPEG at ~30 fps from the raw high-res camera frames
   - ``/telemetry_feed``  Server-Sent Events at ~10 Hz with classification,
     metrics, objects, and events.

Because the camera thread and the inference thread are independent, the video
feed is never bottlenecked by AI processing.  The AI metrics simply update at
whatever rate the hardware allows (~8-15 fps on CPU).

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
.video-panel img, .video-panel canvas {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: contain;
    top: 0;
    left: 0;
}

/* State overlay on video */
.state-overlay {
    position: absolute;
    top: 24px;
    left: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 24px;
    border-radius: var(--radius);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 2px solid rgba(255,255,255,0.15);
    transition: background 0.3s, border-color 0.3s;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.state-overlay.alert { background: rgba(34, 197, 94, 0.25); border-color: rgba(34, 197, 94, 0.5); }
.state-overlay.drowsy { background: rgba(245, 158, 11, 0.25); border-color: rgba(245, 158, 11, 0.5); }
.state-overlay.distracted { background: rgba(239, 68, 68, 0.25); border-color: rgba(239, 68, 68, 0.5); }

.state-label {
    font-weight: 800;
    font-size: 24px;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: color 0.3s;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
.state-conf {
    font-size: 18px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    font-variant-numeric: tabular-nums;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
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

/* Action buttons */
.actions-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 10px;
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}
.btn:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
}
.btn:active {
    transform: scale(0.98);
}
.btn.danger {
    color: #ef4444;
    border-color: rgba(239, 68, 68, 0.3);
}
.btn.danger:hover {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.5);
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

/* Toast Notification */
.toast {
    position: fixed;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    background: rgba(34, 197, 94, 0.9);
    color: #fff;
    padding: 12px 24px;
    border-radius: var(--radius);
    font-weight: 600;
    font-size: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    opacity: 0;
    pointer-events: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 1000;
}
.toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
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
        <canvas id="overlay"></canvas>
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

        <!-- Actions -->
        <div class="card">
            <div class="card-header">Controls</div>
            <div class="actions-grid">
                <button class="btn" onclick="window.doAction('calibrate')">Calibrate</button>
                <button class="btn" onclick="window.doAction('reset')">Reset</button>
                <button class="btn danger" onclick="window.doAction('shutdown')" style="grid-column: span 2;">Stop System</button>
            </div>
            <div style="margin-top: 15px; display: flex; flex-direction: column; gap: 8px;">
                <label style="display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text-secondary); cursor: pointer;">
                    <input type="checkbox" id="toggle-boxes" checked style="accent-color: var(--accent); cursor: pointer;"> Show Bounding Boxes
                </label>
                <label style="display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text-secondary); cursor: pointer;">
                    <input type="checkbox" id="toggle-face" checked style="accent-color: var(--accent); cursor: pointer;"> Prefer Face Box
                </label>
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

<div id="toast" class="toast"></div>

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

        drawOverlay(d);
    }

    function drawOverlay(d) {
        const img = $('video-feed');
        const cvs = $('overlay');
        
        const rect = img.getBoundingClientRect();
        cvs.width = rect.width;
        cvs.height = rect.height;
        const ctx = cvs.getContext('2d');
        ctx.clearRect(0, 0, cvs.width, cvs.height);

        if (!img.naturalWidth || !img.naturalHeight) return;

        const imgAspect = img.naturalWidth / img.naturalHeight;
        const panelAspect = cvs.width / cvs.height;
        
        let drawW, drawH, offsetX, offsetY;
        if (imgAspect > panelAspect) {
            drawW = cvs.width;
            drawH = cvs.width / imgAspect;
            offsetX = 0;
            offsetY = (cvs.height - drawH) / 2;
        } else {
            drawH = cvs.height;
            drawW = cvs.height * imgAspect;
            offsetX = (cvs.width - drawW) / 2;
            offsetY = 0;
        }

        const aiW = d.frame_width || 640;
        const aiH = d.frame_height || 360;
        const mapX = x => offsetX + (x / aiW) * drawW;
        const mapY = y => offsetY + (y / aiH) * drawH;

        ctx.lineWidth = 2;
        ctx.font = "bold 14px Inter";

        const showBoxes = $('toggle-boxes').checked;
        const preferFace = $('toggle-face').checked;

        if (!showBoxes) return;

        let boxToDraw = preferFace && d.face_box ? d.face_box : d.driver_box;
        if (boxToDraw) {
            const [x1, y1, x2, y2] = boxToDraw;
            const stateColorsMap = { Alert: '#22c55e', Drowsy: '#f59e0b', Distracted: '#ef4444' };
            ctx.strokeStyle = stateColorsMap[d.state] || '#ffffff';
            ctx.strokeRect(mapX(x1), mapY(y1), mapX(x2) - mapX(x1), mapY(y2) - mapY(y1));
            ctx.fillStyle = ctx.strokeStyle;
            ctx.fillText("DRIVER", mapX(x1), mapY(y1) - 6);
        }

        if (d.detected_objects && d.detected_objects.length > 0) {
            d.detected_objects.forEach(obj => {
                const [x1, y1, x2, y2] = obj.box;
                const color = obj.label === 'phone' ? '#fb923c' : (obj.label === 'food' ? '#34d399' : '#ef4444');
                ctx.strokeStyle = color;
                ctx.fillStyle = color;
                ctx.strokeRect(mapX(x1), mapY(y1), mapX(x2) - mapX(x1), mapY(y2) - mapY(y1));
                ctx.fillText(obj.label.toUpperCase() + " " + Math.round(obj.conf * 100) + "%", mapX(x1), mapY(y1) - 6);
            });
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

    window.doAction = function(action) {
        fetch('/api/' + action, {method: 'POST'})
            .then(() => {
                if (action === 'calibrate') {
                    showToast('Calibrated zero-point to current head position');
                } else if (action === 'reset') {
                    showToast('System memory & buffers reset to Alert state');
                } else if (action === 'shutdown') {
                    showToast('Shutting down system...');
                    setTimeout(() => window.close(), 1000);
                }
            });
    };

    function showToast(msg) {
        const toast = $('toast');
        toast.textContent = msg;
        toast.className = 'toast show';
        setTimeout(() => { toast.className = 'toast'; }, 3000);
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
# Shared state (all protected by _lock)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_frame_cond = threading.Condition(_lock)
_latest_frame: np.ndarray | None = None      # Raw high-res camera frame
_latest_jpeg: bytes = b""                      # JPEG-encoded display frame
_latest_telemetry: dict = {}                   # Last inference result
_camera_fps: float = 0.0                       # Camera capture FPS
_running: bool = True                          # Shutdown flag
_pipeline = None                               # Global pipeline instance


# ---------------------------------------------------------------------------
# Thread 1: Camera capture at native resolution
# ---------------------------------------------------------------------------
def _camera_thread(source, selfie: bool, res_w: int, res_h: int):
    """Continuously read frames from the camera and store the latest one."""
    global _latest_frame, _latest_jpeg, _camera_fps, _running

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = (
        cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if isinstance(source, int) and sys.platform == "win32"
        else cv2.VideoCapture(source)
    )

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Prevent buffer buildup (slow motion lag)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error("Cannot open camera source: %s", source)
        _running = False
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera opened: %dx%d", actual_w, actual_h)

    fps_times = []
    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if selfie:
                frame = cv2.flip(frame, 1)

            # Track camera FPS
            now = time.perf_counter()
            fps_times.append(now)
            if len(fps_times) > 60:
                fps_times = fps_times[-60:]
            if len(fps_times) > 1:
                _camera_fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

            # Encode JPEG for MJPEG streaming (lower quality = faster encoding)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            with _lock:
                _latest_frame = frame
                _latest_jpeg = buf.tobytes()
                _frame_cond.notify_all()
    finally:
        cap.release()
        logger.info("Camera thread stopped.")


# ---------------------------------------------------------------------------
# Thread 2: AI inference on downscaled frames
# ---------------------------------------------------------------------------
_AI_WIDTH = 960
_AI_HEIGHT = 540


def _inference_thread(pipeline):
    """Grab latest camera frame, moderately downscale, run AI, store telemetry."""
    global _latest_telemetry, _running

    last_frame_id = None

    while _running:
        with _lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        # Skip if camera hasn't produced a new frame yet
        fid = id(frame)
        if fid == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = fid

        # Moderate downscale: 960x540 keeps enough detail for YOLO while
        # being fast enough for MediaPipe (which runs every frame).
        h, w = frame.shape[:2]
        if w != _AI_WIDTH or h != _AI_HEIGHT:
            ai_frame = cv2.resize(frame, (_AI_WIDTH, _AI_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            ai_frame = frame

        try:
            telemetry = pipeline.process_frame(ai_frame)
            with _lock:
                _latest_telemetry = telemetry
        except Exception:
            logger.exception("Inference error")
            time.sleep(0.1)


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
            time.sleep(0.040)  # ~25 fps cap (prevents browser buffer bloat!)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/reset", methods=["POST"])
def api_reset():
    if _pipeline:
        _pipeline.reset()
    return {"status": "ok"}

@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    if _pipeline:
        _pipeline.calibrate()
    return {"status": "ok"}

@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    global _running
    _running = False
    
    # Run shutdown gracefully in background
    def delay_shutdown():
        time.sleep(1.0)
        os._exit(0)
    threading.Thread(target=delay_shutdown).start()
    
    return {"status": "ok"}


@app.route("/telemetry_feed")
def telemetry_feed():
    def gen():
        while True:
            with _lock:
                data = _latest_telemetry.copy() if _latest_telemetry else None
            if data:
                # Inject the camera FPS so the dashboard shows the display rate
                data["camera_fps"] = round(_camera_fps, 1)
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)  # 10 Hz telemetry
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _running

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
    parser.add_argument("--res-w", type=int, default=1280,
                        help="Camera display resolution width (default 1280)")
    parser.add_argument("--res-h", type=int, default=720,
                        help="Camera display resolution height (default 720)")
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

    hef_path = Path(args.hef) if args.hef else None

    # Dynamically import the pipeline class from 09_surveillance_custom.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "surveillance_custom",
        str(_PROJECT_ROOT / "scripts" / "09_surveillance_custom.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Create pipeline in EXTERNAL-CAPTURE mode (source=None)
    # The AI processes at 640x480 regardless of camera resolution
    pipeline = mod.SurveillancePipeline(
        onnx_path=onnx_path, mediapipe_model_path=mp_model,
        yolo_cfg=yolo_cfg, events_cfg=events_cfg,
        clips_cfg=clips_cfg, logging_cfg=logging_cfg,
        dms_cfg=dms_cfg, source=None,
        seq_len=args.seq_len, display=False,
        project_root=_PROJECT_ROOT,
        backend=args.backend,
        hef_path=hef_path,
        frame_w=_AI_WIDTH,
        frame_h=_AI_HEIGHT,
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

    global _pipeline
    _pipeline = pipeline

    logger.info("=" * 60)
    logger.info("DMS WEB DASHBOARD (Decoupled Architecture)")
    logger.info("  Source:        %s", args.source)
    logger.info("  Display res:   %dx%d", args.res_w, args.res_h)
    logger.info("  AI res:        %dx%d", _AI_WIDTH, _AI_HEIGHT)
    logger.info("  Selfie:        %s", selfie)
    logger.info("  Backend:       %s", args.backend)
    logger.info("  Dashboard:     http://%s:%d", args.host, args.port)
    logger.info("=" * 60)

    # Start camera thread (reads at display resolution, ~30fps)
    t_cam = threading.Thread(
        target=_camera_thread,
        args=(args.source, selfie, args.res_w, args.res_h),
        daemon=True,
    )
    t_cam.start()

    # Start inference thread (processes at 640x480, whatever speed the CPU allows)
    t_inf = threading.Thread(
        target=_inference_thread,
        args=(pipeline,),
        daemon=True,
    )
    t_inf.start()

    # Start Flask (blocking)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        _running = False
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
