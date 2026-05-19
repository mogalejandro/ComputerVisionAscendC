# Customer Demo Plan — Multi-Channel NPU Inference on Ascend 910B

---

## 1. Introduction & Hardware Overview

**Key points to cover:**
- Huawei Ascend 910B NPU — designed for high-throughput AI inference
- DVPP (Digital Video Pre-Processing) hardware unit handles decode and resize entirely on-chip
- MxBase / MxVision SDK bridges NPU inference with standard CV pipelines
- Goal: run as many real-time video streams as possible on a single board

---

## 2. Repository Walkthrough

**Open the project root and walk through the structure:**

```
.
├── src/main.cpp               ← three modes in one binary: --video, --rtsp, --sweep
├── detection_utils/
│   ├── detection_utils.h      ← pipeline API: ModelConfig, RunPipeline, cfg:: knobs
│   └── detection_utils.cpp    ← decode → resize thread pool → triple-stage NPU loop
├── tracker/
│   └── SimpleTracker.h        ← header-only greedy IoU tracker
├── yolov3/                    ← YOLOv3 post-processor plugin (blue boxes)
├── yolov11/                   ← YOLOv11 post-processor plugin (green boxes)
├── config.json          ← runtime config: model type, paths, batch size
├── make_video.sh              ← assemble saved frames into MP4 using ffmpeg
└── build.sh                   ← one-command build
```

**Pipeline in three stages (draw on whiteboard or show diagram):**

```
FFmpegReaderThread  →  per-worker raw frame queue  (DVPP hardware decode)
        ↓
ResizeThread pool   →  shared ready queue           (DVPP hardware resize + host DMA)
        ↓
InferenceLoop       →  NPU batch inference → post-process → tracker (optional)
```

**Key design decisions to highlight:**
- DVPP handles decode + resize entirely in hardware — CPU is free for control logic
- Resize threads run **in parallel** with NPU execution — no starvation
- Batch assembly overlaps with the previous NPU batch — eliminates idle gap
- `config.json` selects model at runtime — no recompilation to switch YOLOv3 ↔ YOLOv11

---

## 3. Build & Configure

```bash
# One command build
bash build.sh
```

Show `config.json`:
```jsonc
{
 "model_type": "yolov11",
  "model_path":  "./model/yolov11_bs8_yuv_fp32.om",
  "config_path": "./model/yolov11_bs1_fp16.cfg",
  "label_path":  "./model/yolov11.names",
  "resize_width":  640,
  "resize_height": 640,
  "confidence_threshold": 0.25,
  "nms_iou_threshold": 0.45,

  "tracker_iou_thresh": 0.30,
  "tracker_max_misses": 30,
  "tracker_max_traj": 50,

  "num_devices": 1,
  "batch_size": 8,
  "class_filter": ["car", "truck", "bus"]
}
```

**Points to make:**
- The `.om` file is an ATC-compiled binary specific to Ascend silicon
- `batch_size` must match the compiled model — this is set at ATC conversion time
- Switching models = edit one JSON field

---

## 4. Live Demo — Object Detection & Tracker

### Step 1: Run single-channel detection and tracker, save both outputs

```bash
./infer --video data/test3_1280.mp4 1 --frames --track-frames
```

Expected console output:
```
[INFO] devices=1  vdec_slots=32  skip=0
[14:05:12.001] Video mode: data/test3_1280.mp4  channels=1  skip=0
[14:05:17.001] channels=1  inferred=125  decoded=126  resized=125  fps=25.0  fpsPerCh=25.0  fill=1.00  lag=18.2ms  inferMs=34.1ms  timeout=41ms  skip=0  q=[ ready=1 raw=[ 0 ]]
```

### Step 2: Assemble both outputs in one step

```bash
./make_video.sh --det --track
# → det_0.mp4   (detection boxes)
# → track_0.mp4 (tracker IDs + trajectories)
# Index increments each run — det_1.mp4 / track_1.mp4 on the next run, etc.
```

### Step 3: Play side-by-side and show

**Detection (`det.mp4`):**
- **Green boxes** (YOLOv11) with class name and confidence score
- Clean, tight bounding boxes even on overlapping objects
- Runs at exactly source FPS with full batch fill

**Tracker (`track.mp4`):**
- Each object receives a **persistent integer ID** across frames
- Colour-coded bounding boxes — same colour = same object
- **Trajectory polyline** shows where each object has been (up to 50 frames of history)
- IDs survive brief occlusion (track survives 30 unmatched frames before deletion)
- Zero additional NPU cost — tracker runs on CPU post-process output

| Parameter | Value |
|-----------|-------|
| IoU match threshold | 0.30 |
| Max frames before track deletion | 30 |
| Trajectory history length | 50 frames |

---

## 5. Live Demo — Channel Capacity Sweep

This is the most impactful demo for capacity planning.

### Step 1: Run sweep

**Fast version for live demo (step=4, completes in minutes):**
```bash
./infer --sweep --video data/test3_1280.mp4 1 32 4
```

**Full resolution sweep (step=1, use if time allows — up to ~48 min):**
```bash
./infer --sweep --video data/test3_1280.mp4 1 32 1
```

The binary:
1. Starts at 1 channel, runs for 90 seconds, measures real FPS
2. Steps up by the given step each round
3. Runs through the full range and prints a summary table

### Expected output (example)

```
╔══════════════════════════════════════════════════════╗
║        Ascend 910B  Channel Capacity Sweep           ║
╚══════════════════════════════════════════════════════╝
  Source     : data/test3_1280.mp4 [file]
  Source FPS : 25.0
  Range      : 1 → 32  step 4
  Measure    : 90s
  Model      : yolov11  640×640  batch=8

┌───────────┬───────────┬────────────┬───────────┬──────┐
│ Channels  │ Total FPS │ FPS/Chan   │  Lag (ms) │ Skip │
├───────────┼───────────┼────────────┼───────────┼──────┤
│         1 │      25.0 │      25.00 │      18.1 │    0 │
│         4 │      24.9 │       6.23 │      19.4 │    0 │
│         8 │      24.8 │       3.10 │      18.3 │    0 │
│        16 │      24.7 │       1.54 │      20.1 │    0 │
│        32 │      23.1 │       0.72 │      38.6 │    0 │
└───────────┴───────────┴────────────┴───────────┴──────┘
```

### Step 2: Show skip-interval scaling (optional)

With `--skip 4` (decode every 5th frame), effective required FPS per channel drops from 25 to 5:

```bash
./infer --sweep --video data/test3_1280.mp4 1 64 4 --skip 4
```

**Points to make:**
- `--skip` trades temporal resolution for higher channel count
- At `--skip 4`, 5× more channels are real-time simultaneously
- Useful for security / surveillance where 5 fps per camera is sufficient

---

## Pre-Demo Checklist

- [ ] `./infer` binary built and present in project root
- [ ] `config.json` points to correct `.om` model file
- [ ] `data/test3_1280.mp4` test clip is present
- [ ] `ffmpeg` installed (`ffmpeg -version`)
- [ ] `out/` and `out_track/` will be auto-cleared by `infer` at startup — no manual cleanup needed
- [ ] For RTSP demo: MediaMTX running and stream pushed with ffmpeg
- [ ] Terminal font size increased for visibility
