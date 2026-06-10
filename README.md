# Attention-Algorithm (DAVE)

Driver attention and fatigue monitoring system. The repo holds the full stack:
an on-device vision pipeline that scores driver state from a camera feed, and a
cloud layer that ingests trips, stores clips, and serves a web dashboard.

## Layout

```
config/     model and pipeline config (config.yaml, yolo_config.yaml)
models/     trained weights and landmark assets (.pt, .onnx, .task)
scripts/    numbered data/train/eval/run pipeline (00 -> 10)
src/        edge library: features, detection, model, event engine, clip writer
api/        FastAPI service (trip ingest, storage, queries)
tools/      Raspberry Pi runners (rpi_live.py, rpi_sim.py)
web/        React + Vite dashboard (driver, manager, sensors, trip views)
research/   design notes and experiments
```

## Edge pipeline

The `scripts/` are ordered. Data prep and training run first, then export and
the live runners:

- `00_setup_data` ... `02_build_splits`: dataset prep and splits
- `03_train_model`, `04_evaluate`: train the driver-state net and evaluate it
- `05_export_onnx`: export to ONNX for edge inference
- `06_inference_demo`, `08_surveillance`, `09_surveillance_custom`: live inference
- `10_web_dashboard`: local dashboard against the edge stream

The core logic lives in `src/`: MediaPipe face landmarks and YOLO feed
`features.py`, the ONNX model in `model.py` scores state, `event_engine.py`
turns scores into events, and `clip_writer.py` saves the relevant video.
`hailo_backend.py` runs the model on Hailo accelerators when present.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/06_inference_demo.py
```

## Cloud and web

`api/` is a FastAPI service backed by Postgres and S3-compatible storage,
fronted by Caddy. Bring it up with Docker:

```bash
docker compose up -d        # caddy + api + db
```

The dashboard is in `web/` (React + Vite + Tailwind):

```bash
cd web
npm install
npm run dev
```

## Raspberry Pi

`tools/rpi_live.py` runs the pipeline on-device and pushes trips to the API.
`tools/rpi_sim.py` replays recorded video for testing without hardware.
