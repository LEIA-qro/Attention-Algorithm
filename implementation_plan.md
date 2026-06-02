# DMS Training Pipeline — Execution Plan

> **Scope**: Architecture & Training only. Another team handles AWS/Hardware deployment.
> **Deliverables**: Trained model weights (`.pt` + `.onnx`) + inference script.
> **Deadline**: 3 weeks from today (~June 17, 2026).
> **Hardware**: i9-13900K · 64 GB DDR5 · RTX 4090 · 2 TB NVMe (≤500 GB for data).

---

## High-Level Architecture

The pipeline follows the **pragmatic MVP path** established in the research documents: no raw-pixel end-to-end model. Instead:

```
Video Frame ──► MediaPipe Face Landmarker v2 ──► Feature Extraction ──► Temporal DL Model ──► State
  (RGB)           (478 landmarks + blendshapes       (EAR, MAR, PERCLOS,      (Bi-LSTM         (Alert /
                   + 3D transform matrix)              head pose, gaze          + Attention)      Drowsy /
                                                       stability, blink                           Distracted)
                                                       duration, yawn freq)
```

> [!IMPORTANT]
> Per research doc [02](file:///d:/Antigravity/Attention/Context/02-face-mesh-and-fairness.md) and the [README](file:///d:/Antigravity/Attention/Context/README%20(2).md): we use **MediaPipe Face Landmarker v2** (not legacy FaceMesh), with CLAHE preprocessing and per-person EAR calibration.

---

## 1. Datasets — What to Download

Based on the recommendations in [03-datasets-real.md](file:///d:/Antigravity/Attention/Context/03-datasets-real.md) §"Estrategia recomendada", we need datasets for two modules: **Drowsiness** and **Distraction**. Ebriedad is handled as a proxy through the same features (declared as limitation).

### 1.1 Drowsiness Datasets

| Dataset | Purpose | Est. Size | How to Get |
|---------|---------|-----------|------------|
| **UTA-RLDD** | **Primary train** — 60 subjects, ~30h real-world webcam video, 3 classes (alert / low vigilance / drowsy) | ~25–40 GB | Request from [UT Arlington site](https://sites.google.com/view/utarldd/home) or Reza Ghoddoosian's page. Usually immediate download. |
| **NTHU-DDD** | **Cross-validation / generalization test** — 36 subjects, ~9.5h, day/night/glasses/sunglasses scenarios | ~15–20 GB | Email request to [NTHU lab](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/). Historically slow — **request TODAY**. Mirror repos exist. |
| **YawDD** | **Yawn-specific augmentation** — 322 videos of yawning vs talking vs silent | ~5–8 GB | [IEEE DataPort](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) (free with IEEE account) |

**Total estimated disk for drowsiness: ~40–70 GB**

### 1.2 Distraction Datasets

| Dataset | Purpose | Est. Size | How to Get |
|---------|---------|-----------|------------|
| **State Farm Distracted Driver** | **Primary train** — 10 classes of distracted driving, ~22k train + 79k test images | ~5–8 GB | [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) — instant download |
| **AUC Distracted Driver v2** | **Better split** — 44 subjects, 17k images, subject-disjoint train/test (fixes State Farm's leakage issue) | ~3–5 GB | [Request form](https://abouelnaga.io/projects/auc-distracted-driver-dataset/) — usually granted quickly |

> [!NOTE]
> **DMD (Vicomtech)** is the ideal dataset (covers both drowsiness + distraction + gaze zone), but requires a formal EULA and historically takes 2+ weeks to approve. **Request it NOW** as a stretch goal, but do NOT depend on it for the 3-week timeline. If it arrives mid-project, we integrate it as bonus data.

**Total estimated disk for distraction: ~8–15 GB**

### 1.3 Auxiliary / Pre-training

| Dataset | Purpose | Est. Size | How to Get |
|---------|---------|-----------|------------|
| **MRL Eye Dataset** | Binary open/closed eye classifier to feed PERCLOS computation | ~2 GB | [University of Olomouc](http://mrl.cs.vsb.cz/eyedataset) |

### 1.4 Total Disk Budget

| Category | Range |
|----------|-------|
| Raw downloads | ~50–95 GB |
| Extracted features (`.parquet` / `.npy`) | ~5–15 GB |
| Model checkpoints | ~1–2 GB |
| **Total** | **~60–115 GB** ✅ well within 500 GB limit |

---

## 2. Deep Learning Architecture — Temporal State Classifier

### 2.1 Design Rationale

Per research conclusions:
- The raw perception is already solved by MediaPipe (runs at 10–15 ms/frame on CPU).
- Our model only needs to classify **temporal patterns** over a window of extracted features (EAR, MAR, PERCLOS, head pose angles, gaze stability, blink metrics).
- This is a **time-series classification** problem with ~15–20 features per frame, at ~30 fps, over a 3-second sliding window = **90 timesteps**.
- An LSTM is the natural fit. Doc [05](file:///d:/Antigravity/Attention/Context/05-integration-and-deployment.md) §"Arquitectura de referencia" explicitly recommends: *"LSTM pequeño (hidden 32, ventana 90 frames = 3s)"*.

### 2.2 Proposed Architecture: `DriverStateNet`

```
Input: (batch, seq_len=90, features=18)
           │
           ▼
    ┌──────────────┐
    │  LayerNorm   │   ← normalize features across the feature dim
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Bi-LSTM     │   ← 2 layers, hidden_size=64 per direction
    │  (2 layers,  │     → output per step: 128-dim
    │   dropout=0.3│
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Temporal     │   ← Learnable weighted sum of all 90 timesteps
    │  Attention    │     (single-head, additive attention)
    │  Pooling      │     → context vector: 128-dim
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  FC 128→64   │   + ReLU + Dropout(0.4)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  FC 64→3     │   → logits for [Alert, Drowsy, Distracted]
    └──────────────┘
```

### 2.3 Why This Architecture

| Decision | Rationale |
|----------|-----------|
| **Bi-LSTM over 1D-CNN** | Better at capturing long-range temporal dependencies (e.g., PERCLOS needs ~60s context compressed into features; blink patterns are sequential). The research docs mention LSTM explicitly. |
| **Bidirectional** | During training we have full sequences; bidirectional captures both "ramp up" and "recovery" patterns in drowsiness. At inference we use a rolling window so the "future" is just the next few frames. |
| **2 layers, hidden=64** | Sweet spot: 2 layers capture hierarchical temporal patterns; 64 hidden keeps the model at **~200K parameters** — trains in minutes on a 4090, exports to ONNX trivially, runs at <1ms inference. |
| **Temporal Attention** | Not all frames in a 3s window are equally important. Attention lets the model focus on the "blink event" or "head nod" frames. Better than just taking the last hidden state or mean pooling. |
| **3 classes (not 4)** | Ebriedad is handled as a **proxy** through the same drowsiness + gaze instability features, per research conclusion. The model outputs Alert / Drowsy / Distracted. An ebriedad "alert" is triggered downstream by a rule: `if Drowsy AND gaze_instability > threshold → possible impairment`. |
| **LayerNorm on input** | Features have very different scales (EAR ~0.2–0.4, head yaw ~-90°–+90°, PERCLOS ~0–1). LayerNorm handles this without requiring manual normalization. |

### 2.4 PyTorch Model Definition (Preview)

```python
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over time steps."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        scores = self.attn(lstm_out)            # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden_size)
        return context

class DriverStateNet(nn.Module):
    def __init__(
        self,
        input_size: int = 18,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_output_size = hidden_size * 2  # bidirectional
        self.attention = TemporalAttention(lstm_output_size)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
        context = self.attention(lstm_out)   # (batch, hidden*2)
        logits = self.classifier(context)    # (batch, num_classes)
        return logits
```

**Model size**: ~200K parameters · ~0.8 MB on disk · <1 ms inference on CPU.

### 2.5 Input Feature Vector (18 dimensions)

Each frame produces a feature vector with these components:

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `ear_left` | Landmarks | Eye Aspect Ratio, left eye |
| 2 | `ear_right` | Landmarks | Eye Aspect Ratio, right eye |
| 3 | `ear_avg` | Computed | (ear_left + ear_right) / 2 |
| 4 | `mar` | Landmarks | Mouth Aspect Ratio (yawn detection) |
| 5 | `perclos` | Computed | % eye closure over rolling 60s window (P80) |
| 6 | `blink_rate` | Computed | Blinks per minute (rolling 60s) |
| 7 | `blink_duration_avg` | Computed | Average blink duration in ms (rolling 60s) |
| 8 | `yaw` | 3D Matrix | Head yaw angle (degrees) |
| 9 | `pitch` | 3D Matrix | Head pitch angle (degrees) |
| 10 | `roll` | 3D Matrix | Head roll angle (degrees) |
| 11 | `gaze_yaw` | Iris landmarks | Horizontal gaze direction |
| 12 | `gaze_pitch` | Iris landmarks | Vertical gaze direction |
| 13 | `gaze_stability` | Computed | Std-dev of gaze angle over rolling 1s (instability marker) |
| 14 | `head_pose_stability` | Computed | Std-dev of head pose (yaw+pitch) over rolling 1s |
| 15 | `ear_velocity` | Computed | d(ear_avg)/dt — speed of eyelid closure |
| 16 | `head_nod_count` | Computed | Number of pitch "dips" > 15° in rolling 10s |
| 17 | `mouth_open_duration` | Computed | Current consecutive frames with MAR > threshold |
| 18 | `eyes_off_road_pct` | Computed | % time gaze was >30° from center in rolling 5s |

---

## 3. Python Scripts — Full Pipeline

### 3.0 Project Structure

```
d:\Antigravity\Attention\
├── config/
│   └── config.yaml              # All hyperparameters, paths, thresholds
├── data/
│   ├── raw/                     # Downloaded dataset files (gitignored)
│   │   ├── uta_rldd/
│   │   ├── nthu_ddd/
│   │   ├── state_farm/
│   │   ├── auc_v2/
│   │   └── yawdd/
│   ├── features/                # Extracted feature sequences (.parquet)
│   └── splits/                  # train/val/test split CSVs
├── scripts/
│   ├── 00_download_data.py      # Dataset download + organization
│   ├── 01_extract_features.py   # MediaPipe → feature vectors
│   ├── 02_build_splits.py       # Subject-disjoint train/val/test
│   ├── 03_train_model.py        # PyTorch training loop
│   ├── 04_evaluate.py           # Metrics, confusion matrix, per-class F1
│   ├── 05_export_onnx.py        # Export to ONNX for deployment team
│   └── 06_inference_demo.py     # Real-time webcam demo
├── src/
│   ├── __init__.py
│   ├── features.py              # Feature extraction functions (EAR, MAR, etc.)
│   ├── model.py                 # DriverStateNet definition
│   ├── dataset.py               # PyTorch Dataset / DataLoader
│   ├── preprocessing.py         # CLAHE, gamma correction, calibration
│   └── utils.py                 # Logging, metrics, visualization helpers
├── models/                      # Saved checkpoints + exported ONNX
│   ├── best_model.pt
│   └── driver_state_net.onnx
├── requirements.txt
└── README.md
```

### 3.1 Script-by-Script Breakdown

---

#### `scripts/00_download_data.py`
**Purpose**: Automate dataset acquisition and organize into `data/raw/`.

**What it does**:
- Downloads **State Farm** from Kaggle API (`kaggle competitions download`).
- Downloads **YawDD** from IEEE DataPort (or provides instructions for manual download).
- Downloads **MRL Eye** directly (HTTP).
- Prints clear instructions for datasets requiring manual request (UTA-RLDD, NTHU-DDD, AUC v2, DMD).
- Extracts archives, normalizes directory structure.
- Validates file counts and reports per-dataset stats (# subjects, # videos/images, total size).

**Dependencies**: `kaggle`, `requests`, `tqdm`, `zipfile`.

---

#### `scripts/01_extract_features.py`
**Purpose**: Process every video/image through MediaPipe Face Landmarker v2 and extract the 18-dimensional feature vector per frame.

**What it does**:
1. Loads MediaPipe Face Landmarker v2 `.task` model (auto-downloads if missing).
2. Applies **CLAHE on L channel** (per doc [02](file:///d:/Antigravity/Attention/Context/02-face-mesh-and-fairness.md) §"Recomendación concreta").
3. Runs face detection + landmark extraction on every frame.
4. Computes per-frame features: EAR, MAR, head pose (from 3D transform matrix), gaze direction (from iris landmarks).
5. Computes rolling/temporal features: PERCLOS (P80 over 60s), blink rate, blink duration, gaze stability, head pose stability, head nod count, eyes-off-road percentage.
6. Saves feature sequences as `.parquet` files with columns for all 18 features + metadata (dataset, subject_id, video_id, frame_idx, label).
7. Handles failures gracefully (face not detected → NaN interpolation or discard).

**Key design decisions**:
- Process at **native FPS** (not resampled) — the model will handle variable rates.
- Use **CLAHE conditioned on brightness** — only apply when mean ROI brightness < threshold.
- **EAR calibration baseline**: compute per-subject EAR baseline from first 30s of "alert" samples; normalize subsequent EAR by baseline.

**Performance**: With RTX 4090, MediaPipe on CPU + feature extraction → ~100–200 fps throughput. 30h of video ≈ 3.2M frames → ~4.5–9 hours processing time.

**Dependencies**: `mediapipe`, `opencv-python`, `numpy`, `pandas`, `pyarrow`, `tqdm`.

---

#### `scripts/02_build_splits.py`
**Purpose**: Create **subject-disjoint** train/val/test splits.

**What it does**:
- Groups all feature sequences by `subject_id`.
- Assigns subjects (not individual samples) to train/val/test with ratio **70/15/15**.
- Stratifies by dataset source to ensure each split has data from all datasets.
- Produces `data/splits/train.csv`, `val.csv`, `test.csv` with columns: `feature_file_path, subject_id, dataset, label_distribution`.
- Prints split statistics: # subjects, # sequences, # frames, class balance per split.

> [!WARNING]
> Subject-disjoint splitting is **critical**. State Farm's original competition had identity leakage (same driver in train+test) which inflated metrics. We avoid this by design.

---

#### `scripts/03_train_model.py`
**Purpose**: Train `DriverStateNet` on extracted feature sequences.

**What it does**:
1. Loads feature sequences from parquet files, segments into overlapping windows of `seq_len=90` frames with stride 15.
2. Handles class imbalance with **weighted cross-entropy** (class weights inversely proportional to frequency).
3. Training loop with:
   - **Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4
   - **Scheduler**: OneCycleLR (max_lr=1e-3, total_steps=num_epochs*steps_per_epoch)
   - **Epochs**: 50 (with early stopping, patience=10)
   - **Batch size**: 256 (fits comfortably in 24GB VRAM given model is tiny)
   - **Mixed precision**: AMP (fp16) — not critical for this model but good practice
4. Logs metrics per epoch: loss, accuracy, per-class F1, macro F1.
5. Saves best model checkpoint (by val macro F1) to `models/best_model.pt`.
6. TensorBoard logging for loss curves, confusion matrices, attention weight visualizations.

**Estimated training time**: With ~3M frames → ~33K windows → ~130 batches/epoch → **<30 seconds/epoch** on RTX 4090. Full training (50 epochs) ≈ **~25 minutes**. This is the beauty of the lightweight feature-based approach.

**Dependencies**: `torch`, `tensorboard`, `scikit-learn`, `tqdm`.

---

#### `scripts/04_evaluate.py`
**Purpose**: Comprehensive evaluation on the held-out test set.

**What it does**:
- Loads best checkpoint, runs inference on test split.
- Computes and reports:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Macro & weighted F1
  - Confusion matrix (with visualization)
  - Per-dataset performance breakdown (to check generalization)
  - Per-subject performance (to detect demographic/individual outliers)
- Generates plots saved to `models/evaluation/`:
  - Confusion matrix heatmap
  - ROC curves per class
  - Attention weight distribution analysis
  - Feature importance via permutation

**Dependencies**: `torch`, `scikit-learn`, `matplotlib`, `seaborn`.

---

#### `scripts/05_export_onnx.py`
**Purpose**: Export the trained model to ONNX format for the deployment team.

**What it does**:
- Loads `best_model.pt`.
- Creates a dummy input `(1, 90, 18)` and traces with `torch.onnx.export()`.
- Sets dynamic axes for batch dimension.
- Validates the ONNX model with `onnxruntime`.
- Runs a numerical comparison (PyTorch vs ONNX) to verify correctness.
- Saves `models/driver_state_net.onnx`.
- Also exports a `models/feature_config.json` with feature names, normalization stats, and thresholds — everything the deployment team needs.

**Dependencies**: `torch`, `onnx`, `onnxruntime`.

---

#### `scripts/06_inference_demo.py`
**Purpose**: Real-time webcam demo that chains the full pipeline.

**What it does**:
1. Opens webcam with OpenCV.
2. Applies CLAHE preprocessing.
3. Runs MediaPipe Face Landmarker v2.
4. Extracts features in real-time.
5. Maintains a rolling buffer of 90 frames.
6. Runs the ONNX model every N frames (configurable, default=5 for ~6 Hz classification).
7. Displays overlay with:
   - Current state prediction (color-coded: green=Alert, yellow=Drowsy, red=Distracted)
   - Real-time feature values (EAR, MAR, PERCLOS, head pose angles)
   - **Explainability text**: "Eyes closed for 1.8s", "Head turned 45° right for 3s", etc.
   - Confidence bar per class
8. Triggers audio alert on sustained Drowsy/Distracted state (>2s).

**This is the deliverable** the deployment team will adapt for their AWS pipeline.

**Dependencies**: `opencv-python`, `mediapipe`, `onnxruntime`, `numpy`, `pygame` (for audio alerts).

---

#### `src/features.py`
Core feature extraction logic, shared between `01_extract_features.py` and `06_inference_demo.py`:

```python
# Key functions:
def compute_ear(landmarks, eye_indices) -> float
def compute_mar(landmarks, mouth_indices) -> float
def compute_head_pose(face_transform_matrix) -> Tuple[float, float, float]  # yaw, pitch, roll
def compute_gaze_direction(iris_landmarks, eye_landmarks) -> Tuple[float, float]  # gaze_yaw, gaze_pitch
def compute_perclos(ear_history, threshold, window_sec, fps) -> float
def compute_blink_metrics(ear_history, threshold, fps) -> Tuple[float, float]  # rate, avg_duration
def compute_gaze_stability(gaze_history, window_sec, fps) -> float
def compute_head_pose_stability(pose_history, window_sec, fps) -> float
def compute_head_nod_count(pitch_history, threshold_deg, window_sec, fps) -> int
def compute_eyes_off_road_pct(gaze_history, threshold_deg, window_sec, fps) -> float
def extract_frame_features(landmarks, blendshapes, transform_matrix) -> Dict[str, float]
```

---

#### `src/preprocessing.py`
Image preprocessing from doc [02](file:///d:/Antigravity/Attention/Context/02-face-mesh-and-fairness.md):

```python
def apply_clahe_lab(image, clip_limit=2.0, tile_grid=(8,8)) -> np.ndarray
def adaptive_gamma(image, target_mean=127) -> np.ndarray
def preprocess_frame(image, brightness_threshold=100) -> np.ndarray
    # Only applies CLAHE + gamma when ROI brightness is below threshold
```

---

## 4. Three-Week Timeline

### Week 1: Data & Features (Days 1–7)

| Day | Task | Script |
|-----|------|--------|
| **1** | Request UTA-RLDD, NTHU-DDD, AUC v2, DMD (emails/forms). Download State Farm (Kaggle), YawDD, MRL Eye. | `00_download_data.py` |
| **2** | Set up repo structure, `requirements.txt`, `config.yaml`. Write `src/preprocessing.py` and `src/features.py`. | — |
| **3** | Write and test `01_extract_features.py` on a small subset (1 video from each dataset). Validate EAR/MAR/head pose values visually. | `01_extract_features.py` |
| **4–5** | Run full feature extraction on all downloaded datasets. Debug edge cases (no face detected, corrupted videos, wrong FPS). | `01_extract_features.py` |
| **6** | Write `02_build_splits.py`. Generate subject-disjoint splits. Analyze class balance, decide on oversampling/weighting. | `02_build_splits.py` |
| **7** | Write `src/dataset.py` (PyTorch Dataset with windowing). Write `src/model.py`. Sanity-check: forward pass with random data, gradient flow. | — |

### Week 2: Training & Iteration (Days 8–14)

| Day | Task | Script |
|-----|------|--------|
| **8** | Write `03_train_model.py`. Train first baseline on available data (State Farm + YawDD + whatever arrived). | `03_train_model.py` |
| **9** | Write `04_evaluate.py`. Evaluate baseline. Identify weak spots (which class? which dataset transfers worst?). | `04_evaluate.py` |
| **10–11** | Hyperparameter sweep: hidden_size ∈ {32, 64, 128}, num_layers ∈ {1, 2, 3}, seq_len ∈ {60, 90, 120}, dropout ∈ {0.2, 0.3, 0.4}. Pick best combo. | `03_train_model.py` |
| **12** | Integrate any newly arrived datasets (UTA-RLDD, NTHU-DDD). Re-extract features. Re-train. | `01/03` |
| **13** | Feature ablation study: which features matter most? Try removing groups (blink-related, head-pose-related, gaze-related). | `03/04` |
| **14** | Finalize model. Write `05_export_onnx.py`. Export and validate. | `05_export_onnx.py` |

### Week 3: Demo, Polish & Handoff (Days 15–21)

| Day | Task | Script |
|-----|------|--------|
| **15–16** | Write `06_inference_demo.py`. Real-time demo with webcam. Test with team members simulating drowsy/distracted behavior. | `06_inference_demo.py` |
| **17** | Edge case testing: glasses, different skin tones, lighting conditions, head positions. Log failures. | `06_inference_demo.py` |
| **18** | Write final `README.md` with setup instructions, architecture diagram, training results, known limitations. | — |
| **19** | Package deliverables for deployment team: `driver_state_net.onnx`, `feature_config.json`, `06_inference_demo.py`, `src/features.py`, `src/preprocessing.py`. | — |
| **20** | Buffer day / address feedback from deployment team. | — |
| **21** | Final integration test. Tag release. | — |

---

## 5. Ebriedad / Impairment — Proxy Strategy

Per [03-datasets-real.md](file:///d:/Antigravity/Attention/Context/03-datasets-real.md) §"Módulo de ebriedad (proxy)":

> [!IMPORTANT]
> There is **no public dataset** for drunk driving. We do NOT train a separate "drunk" class. Instead, we implement a **rule-based proxy** downstream of the model:

```python
def check_impairment_proxy(features: dict, model_prediction: str) -> bool:
    """
    Heuristic: if the model says 'Drowsy' AND gaze is unusually unstable
    AND blink patterns are abnormal, flag as possible impairment.
    """
    is_drowsy = model_prediction == "Drowsy"
    gaze_unstable = features["gaze_stability"] > GAZE_STABILITY_THRESHOLD
    abnormal_blinks = features["blink_duration_avg"] > BLINK_DURATION_THRESHOLD
    head_bobbing = features["head_nod_count"] > HEAD_NOD_THRESHOLD
    
    impairment_score = sum([is_drowsy, gaze_unstable, abnormal_blinks, head_bobbing])
    return impairment_score >= 3  # at least 3 of 4 signals
```

This is declared as a **proof-of-concept limitation** in the report, not as a validated detector.

---

## User Review Required

> [!IMPORTANT]
> **Dataset access is time-critical.** Please request access to UTA-RLDD and NTHU-DDD **today**. These are the most important datasets and academic requests can take days to weeks.

> [!IMPORTANT]
> **3-class vs 4-class decision.** I've proposed 3 classes (Alert/Drowsy/Distracted) with ebriedad as a rule-based proxy. If the science fair judges expect a separate "Ebrio" class, we could add it as a 4th class trained on synthetically-generated feature vectors (augmented drowsiness + noise injection). Let me know your preference.

## Open Questions

1. **State Farm is image-only (no video)**. This means we can only train the *distraction* path on single frames, not temporal sequences. Two options:
   - (a) Train a separate lightweight CNN (MobileNetV3) for distraction classification on raw images from State Farm, and fuse its output with the temporal LSTM.
   - (b) Use State Farm only for extracting features (landmarks → feature vector) and treat each image as a length-1 sequence. The LSTM would see no temporal context for distraction — relying instead on instantaneous pose/gaze features.
   - **I recommend (b)** for simplicity: the instantaneous features (head turn angle, gaze direction, phone-holding posture via landmarks) are already strongly discriminative for distraction. The temporal model would primarily help drowsiness, which has inherently temporal patterns.

2. **Do you want a fairness validation study** (per doc [02](file:///d:/Antigravity/Attention/Context/02-face-mesh-and-fairness.md))? This adds ~1 day of work but would be impressive for the science fair and is recommended in the research docs.

3. **ONNX vs TorchScript for deployment?** I've defaulted to ONNX (most portable). Does the deployment team have a preference?

---

## Verification Plan

### Automated Tests
- **Unit tests** for feature extraction functions (EAR, MAR, head pose) against known landmark positions.
- **Integration test**: run full pipeline on 5 short test videos (one per class + edge cases) and assert predictions match expected labels.
- **ONNX validation**: numerical comparison between PyTorch and ONNX inference outputs (max absolute error < 1e-5).
- **Train/eval**: Target **macro F1 ≥ 0.85** on the held-out test set. If below 0.80, we iterate on features/architecture.

### Manual Verification
- Real-time webcam demo tested by at least 3 team members simulating each state.
- Verify latency: full pipeline (MediaPipe + features + ONNX inference) < 50 ms/frame on the target hardware.
- Verify the deployment team can successfully load and run the exported ONNX model with the inference script.
