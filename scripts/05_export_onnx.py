#!/usr/bin/env python3

"""Export trained DriverStateNet to ONNX and write feature_config.json."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

logger = logging.getLogger("dms.export_onnx")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import DriverStateNet  # noqa: E402

LABEL_NAMES = ["Alert", "Drowsy", "Distracted"]
FEATURE_COLS = [
    "ear_left", "ear_right", "ear_avg", "mar",
    "perclos", "blink_rate", "blink_duration_avg",
    "yaw", "pitch", "roll",
    "gaze_yaw", "gaze_pitch", "gaze_stability",
    "head_pose_stability", "ear_velocity",
    "head_nod_count", "mouth_open_duration",
    "eyes_off_road_pct",
]


def _setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
    logging.basicConfig(level=numeric, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    seq_len: int = 90,
    input_size: int = 18,
    opset_version: int = 17,
) -> None:
    """Export the PyTorch model to ONNX with dynamic batch axis."""
    model.eval()
    device = next(model.parameters()).device

    dummy_input = torch.randn(1, seq_len, input_size, device=device)

    logger.info("Exporting to ONNX: %s", onnx_path)
    logger.info("  Input shape:  (batch, %d, %d)", seq_len, input_size)
    logger.info("  Opset:        %d", opset_version)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
    )

    logger.info("onnx export complete: %s", onnx_path)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("  File size: %.2f MB", size_mb)


def validate_onnx(
    onnx_path: Path,
    model: torch.nn.Module,
    seq_len: int = 90,
    input_size: int = 18,
    tolerance: float = 5e-2,
    n_test_batches: int = 5,
) -> bool:
    """check the onnx graph and compare outputs against pytorch; True if within tolerance"""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.error("Install onnx and onnxruntime: pip install onnx onnxruntime")
        return False

    logger.info("validating onnx graph")
    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
        logger.info("  onnx graph is valid")
    except onnx.checker.ValidationError as exc:
        logger.error("  onnx validation failed: %s", exc)
        return False

    # onnx runtime session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available_providers = ort.get_available_providers()
    use_providers = [p for p in providers if p in available_providers]
    logger.info("  ONNX Runtime providers: %s", use_providers)

    session = ort.InferenceSession(str(onnx_path), providers=use_providers)

    for inp in session.get_inputs():
        logger.info("  Input:  %s  shape=%s  dtype=%s", inp.name, inp.shape, inp.type)
    for out in session.get_outputs():
        logger.info("  Output: %s  shape=%s  dtype=%s", out.name, out.shape, out.type)

    logger.info("running numerical comparison (%d test batches)", n_test_batches)
    model.eval()
    device = next(model.parameters()).device
    max_abs_error = 0.0
    all_passed = True

    for batch_idx in range(n_test_batches):
        batch_size = np.random.randint(1, 17)
        test_input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

        torch_input = torch.from_numpy(test_input).to(device)
        with torch.no_grad():
            torch_output = model(torch_input).cpu().numpy()

        ort_output = session.run(None, {"features": test_input})[0]

        abs_error = np.max(np.abs(torch_output - ort_output))
        max_abs_error = max(max_abs_error, abs_error)

        status = "ok" if abs_error < tolerance else "FAIL"
        logger.info(
            "  Batch %d (size=%d): max_abs_error = %.2e  %s",
            batch_idx + 1, batch_size, abs_error, status,
        )

        if abs_error >= tolerance:
            all_passed = False

    logger.info("Overall max absolute error: %.2e (threshold: %.1e)", max_abs_error, tolerance)

    if all_passed:
        logger.info("numerical validation passed")
    else:
        logger.error("numerical validation failed: errors exceed tolerance")

    return all_passed


def save_feature_config(
    ckpt: Dict[str, Any],
    save_path: Path,
) -> None:
    """write feature_config.json for deployment"""
    hparams = ckpt.get("hparams", {})
    model_config = ckpt.get("model_config", {})
    label_map = ckpt.get("label_map", {"Alert": 0, "Drowsy": 1, "Distracted": 2})
    label_names = ckpt.get("label_names", ["Alert", "Drowsy", "Distracted"])

    config = {
        "model": {
            "architecture": "DriverStateNet",
            "input_size": model_config.get("input_size", 18),
            "hidden_size": model_config.get("hidden_size", 64),
            "num_layers": model_config.get("num_layers", 2),
            "num_classes": model_config.get("num_classes", 3),
            "seq_len": hparams.get("seq_len", 90),
        },
        "features": {
            "names": FEATURE_COLS,
            "count": len(FEATURE_COLS),
            "description": {
                "ear_left": "Eye Aspect Ratio - left eye",
                "ear_right": "Eye Aspect Ratio - right eye",
                "ear_avg": "Average EAR (left + right) / 2",
                "mar": "Mouth Aspect Ratio (yawn detection)",
                "perclos": "Percentage of eye closure over 60s (P80)",
                "blink_rate": "Blinks per minute (rolling 60s)",
                "blink_duration_avg": "Average blink duration in ms (rolling 60s)",
                "yaw": "Head yaw angle (degrees)",
                "pitch": "Head pitch angle (degrees)",
                "roll": "Head roll angle (degrees)",
                "gaze_yaw": "Horizontal gaze direction (degrees)",
                "gaze_pitch": "Vertical gaze direction (degrees)",
                "gaze_stability": "Std-dev of gaze angle over 1s window",
                "head_pose_stability": "Std-dev of head pose over 1s window",
                "ear_velocity": "Rate of change of EAR (d(ear_avg)/dt)",
                "head_nod_count": "Number of pitch dips >15 deg in rolling 10s",
                "mouth_open_duration": "Consecutive frames with MAR > threshold",
                "eyes_off_road_pct": "% time gaze >30 deg from center in 5s window",
            },
        },
        "labels": {
            "map": label_map,
            "names": label_names,
        },
        "thresholds": {
            "ear_closed": 0.20,
            "mar_yawn": 0.60,
            "perclos_drowsy": 0.40,
            "gaze_off_road_deg": 30.0,
            "head_nod_pitch_deg": 15.0,
            "gaze_stability_impairment": 8.0,
            "blink_duration_impairment_ms": 400.0,
            "head_nod_impairment_count": 3,
        },
        "inference": {
            "fps": 29.76,
            "classification_interval_frames": 5,
            "rolling_buffer_frames": 90,
            "alert_duration_s": 2.0,
        },
        "training": {
            "epoch": ckpt.get("epoch", -1),
            "best_val_f1": float(ckpt.get("best_val_f1", 0)),
            "val_metrics": ckpt.get("val_metrics", {}),
        },
    }

    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    logger.info("saved feature config: %s", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DriverStateNet to ONNX.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="ONNX output path")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Max abs error tolerance")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    project_root = _PROJECT_ROOT
    models_dir = project_root / "models"

    ckpt_path = Path(args.checkpoint) if args.checkpoint else models_dir / "best_model.pt"
    onnx_path = Path(args.output) if args.output else models_dir / "driver_state_net.onnx"

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    device = torch.device("cpu")  # export on cpu for portability
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", {})

    model = DriverStateNet(
        input_dim=model_config.get("input_dim", 18),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 2),
        num_classes=model_config.get("num_classes", 3),
        dropout_lstm=model_config.get("dropout_lstm", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seq_len = ckpt.get("hparams", {}).get("seq_len", 90)
    input_size = model_config.get("input_dim", 18)

    logger.info("Model loaded from %s", ckpt_path)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %s", f"{total_params:,}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, onnx_path, seq_len=seq_len, input_size=input_size, opset_version=args.opset)

    passed = validate_onnx(
        onnx_path, model,
        seq_len=seq_len, input_size=input_size,
        tolerance=args.tolerance,
    )

    config_path = models_dir / "feature_config.json"
    save_feature_config(ckpt, config_path)

    logger.info("onnx export complete")
    logger.info("  onnx model:      %s", onnx_path)
    logger.info("  feature config:  %s", config_path)
    logger.info("  validation:      %s", "PASSED" if passed else "FAILED")

    if not passed:
        logger.warning("onnx numerical validation failed, check the model export")
        sys.exit(1)

    logger.info("Next step: python scripts/06_inference_demo.py --config %s", args.config)


if __name__ == "__main__":
    main()
