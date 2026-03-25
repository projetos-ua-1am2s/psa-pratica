# PSA Prática

A practical project developed at the **University of Aveiro** for the *Autonomous Systems* course. It explores real-time computer-vision pipelines applied to person detection and surveillance, using state-of-the-art AI models and standard hardware (webcam or depth camera).

---

## What this project does

The project builds a modular surveillance system that can:

- **Detect and track people** in a live video stream using [YOLOv8](https://github.com/ultralytics/ultralytics).
- **Calculate movement vectors** (distance and direction relative to the frame centre) for each tracked person.
- **Log detections to CSV** with timestamps, tracking IDs, confidence scores, and acceptance status.
- **Select the best available hardware automatically** — Apple MPS (M-series), NVIDIA CUDA, or CPU.

The roadmap included in each module extends the system toward full 3-D vigilance with face recognition (DeepFace / InsightFace) and depth sensing (Intel RealSense).

---

## Repository structure

```
psa-pratica/
├── README.md          ← you are here
└── visao/             ← real-time person detection & tracking module
    ├── README.md      ← module documentation & how-to-use guide
    ├── main.py        ← entry point (run this)
    ├── person_tracker.py  ← PersonTracker class
    ├── data.yaml      ← YOLOv8 dataset/validation config
    └── yolov8n.pt     ← YOLOv8 Nano model weights (auto-downloaded if missing)
```

---

## Quick start

1. **Install dependencies**
   ```bash
   pip install opencv-python ultralytics torch
   ```

2. **Run the surveillance module**
   ```bash
   cd visao
   python main.py
   ```

3. **Stop the system**  
   Press **`q`** in the video window, or **`Ctrl+C`** in the terminal.

> See [`visao/README.md`](visao/README.md) for full usage details, configuration options, and the `PersonTracker` API reference.

---

## Tech stack

| Component | Library / Tool |
|-----------|---------------|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Video capture & display | [OpenCV](https://opencv.org/) |
| Deep learning backend | [PyTorch](https://pytorch.org/) (MPS / CUDA / CPU) |
| Data logging | Python `csv` (standard library) |

---

## Authors

Developed as part of the PSA practical sessions at the University of Aveiro.