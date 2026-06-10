#!/usr/bin/env python3
"""Simula el lado RPi posteando a la API del DMS; solo usa la libreria estandar."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone

QRO = (20.5888, -100.3899)


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


def start_session(base: str) -> str:
    resp = post(base, "/sessions", {"driver_id": "rpi-sim"})
    print("sesion iniciada:", resp["session_id"])
    return resp["session_id"]


def push(base: str, sid: str, i: int, state: str, conf: float) -> None:
    ts = now_iso()
    lat = QRO[0] + i * 0.0021
    lng = QRO[1] + i * 0.0026
    post(base, f"/sessions/{sid}/states", {"ts": ts, "state": state, "confidence": conf})
    post(base, f"/sessions/{sid}/track", {"ts": ts, "lat": lat, "lng": lng, "speed_kmh": 80 + (i * 7) % 40})
    if state != "Alert":
        post(base, f"/sessions/{sid}/incidents", {
            "ts": ts, "state": state, "confidence": conf,
            "speed_kmh": 80 + (i * 7) % 40, "lat": lat, "lng": lng,
            "harsh_event": state == "Distracted",
        })
        print(f"  incidente: {state} ({conf:.2f})")


SCRIPT = ["Alert", "Alert", "Drowsy", "Alert", "Alert", "Distracted", "Alert", "Drowsy", "Drowsy", "Alert"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://34-205-126-89.nip.io/api")
    ap.add_argument("--mode", choices=["burst", "live"], default="burst")
    args = ap.parse_args()

    sid = start_session(args.base)
    i = 0
    try:
        while True:
            state = SCRIPT[i % len(SCRIPT)]
            push(args.base, sid, i, state, 0.7 if state == "Alert" else 0.9)
            i += 1
            if args.mode == "burst":
                if i >= len(SCRIPT):
                    break
            else:
                time.sleep(1.5)
    finally:
        if args.mode == "burst":
            post(args.base, f"/sessions/{sid}/end", {})
            print("sesion cerrada:", sid)


if __name__ == "__main__":
    main()
