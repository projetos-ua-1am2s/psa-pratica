# visao – Deteção de Pessoas em Tempo Real

Este módulo usa [YOLOv8](https://github.com/ultralytics/ultralytics) com OpenCV para detetar e rastrear pessoas através da câmara do computador.

## Dependências

Instala as dependências com pip:

```bash
pip install opencv-python ultralytics torch
```

> **Nota:** Em Macs com Apple Silicon (M1/M2/M3) o PyTorch utilizará automaticamente o backend MPS. Em sistemas com GPU NVIDIA será usado CUDA. Nos restantes será usado CPU.

## Modelo

O script utiliza o modelo `yolov8n.pt` (YOLOv8 Nano). Se o ficheiro não existir na pasta `visao/`, o Ultralytics faz o download automático na primeira execução.

## Permissões de câmara

O script acede à câmara do dispositivo (índice `0`). Em **macOS**, pode ser necessário conceder permissão à aplicação de terminal em:

> **macOS Ventura (13) ou superior:** Definições do Sistema → Privacidade e Segurança → Câmara  
> **macOS Monterey (12) ou inferior:** Preferências do Sistema → Privacidade e Segurança → Câmara

Em **Linux**, certifica-te de que o teu utilizador pertence ao grupo `video`:

```bash
sudo usermod -aG video $USER
```

## Como executar

A partir da raiz do repositório:

```bash
cd visao
python main.py
```

Prima **`q`** na janela de visualização (com a janela em foco) para terminar o programa.

---

## How to Use the Code

This section explains how to use the `PersonTracker` class directly in your own Python scripts, and covers every configuration option and output format.

### Running the ready-made script

The simplest way to start the system is to run `main.py`:

```bash
cd visao
python main.py
```

What happens when you run it:

1. A `PersonTracker` is created with a confidence threshold of `0.5`.
2. The webcam (index `0`) is opened.
3. Each frame is processed by YOLOv8; detected persons are annotated and displayed in a window called **"Tracking View"**.
4. Detection data is written line-by-line to `surveillance_data.csv` in the current directory.
5. A movement vector (distance + angle from the frame centre to the first tracked person) is printed whenever its magnitude exceeds `0.05`.
6. The system shuts down cleanly on **`q`** or **`Ctrl+C`**.

---

### Using `PersonTracker` in your own script

Import and instantiate the class, then consume its generator:

```python
import csv
import cv2
from visao.person_tracker import PersonTracker   # adjust import path as needed

# Create a tracker (see "Configuration" below for parameter details)
tracker = PersonTracker(conf_threshold=0.5)

with open("my_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "ID", "Confidence", "Status"])

    for vector, frame, boxes in tracker.run():

        # --- Act on the movement vector ---
        if vector is not None:
            magnitude, angle = vector
            if magnitude > 0.1:
                print(f"Person is {magnitude:.2f} away at {angle:.1f}°")

        # --- Log detections to CSV ---
        if boxes is not None:
            tracker.log_detections(writer, boxes)

        # --- Display the annotated frame ---
        cv2.imshow("My View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

---

### Configuration

`PersonTracker.__init__` accepts the following keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | `"yolov8n.pt"` | Path to the YOLO model weights file. A relative path is resolved relative to `person_tracker.py`. |
| `conf_threshold` | `float` | `0.3` | Minimum confidence for a detection to be yielded by the generator. Also used as `accept_threshold` when the latter is not set. |
| `accept_threshold` | `float \| None` | `None` | Confidence above which a detection is labeled **"Accepted"** in the CSV log. Defaults to `conf_threshold`. |

**Example — stricter acceptance, looser detection:**

```python
# Detections with confidence >= 0.25 appear in the stream,
# but only those >= 0.6 are labeled "Accepted" in the CSV.
tracker = PersonTracker(conf_threshold=0.25, accept_threshold=0.6)
```

> **Internal note:** The tracker asks YOLO to use a confidence of `accept_threshold / 2` so that borderline detections are visible in the log as "Rejected" rather than being silently discarded.

---

### Understanding the outputs

#### 1. Movement vector — `vector`

The generator yields a `vector` value for each frame. It is either `None` (no person detected) or a two-element list `[magnitude, angle]`:

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `magnitude` | `float` | `0.0 – 1.0` | Normalised distance of the first tracked person from the frame centre. `0.0` = at the centre; `1.0` = at a corner. |
| `angle` | `float` | `-180 – 180°` | Direction from the frame centre to the person. `0°` = right; `90°` = down; `-90°` = up; `±180°` = left. |

#### 2. Annotated frame — `frame`

A NumPy array (BGR) with YOLO bounding boxes, tracking IDs, and — when a person is detected — a green line from the frame centre to the first tracked person and a text overlay showing the vector values.

#### 3. Detection boxes — `boxes`

An [Ultralytics `Boxes`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes) object (or `None` when no person is detected). Each element exposes:
- `.conf` — confidence score tensor
- `.id` — tracking ID tensor (or `None` if tracking failed)
- `.xywh` — bounding-box tensor in `[x_centre, y_centre, width, height]` format

#### 4. CSV log — `surveillance_data.csv`

Each row written by `log_detections` has four columns:

| Column | Example | Description |
|--------|---------|-------------|
| `Timestamp` | `14:32:07` | Wall-clock time of detection (`HH:MM:SS`) |
| `ID` | `3` | YOLO tracking ID (`"N/A"` if unavailable) |
| `Confidence` | `0.87` | Raw model confidence (two decimal places) |
| `Status` | `Accepted` | `"Accepted"` if confidence ≥ `accept_threshold`, otherwise `"Rejected"` |

---

### Running model validation

You can validate the model against a labeled dataset defined in `data.yaml`:

```python
tracker = PersonTracker()
tracker.validate()          # uses visao/data.yaml by default
# or provide a custom path:
tracker.validate(data_config="/path/to/my_dataset.yaml")
```

This prints **mAP50** and **Recall** metrics to the console.

---

### Method reference

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `() → Generator` | Yields `(vector, frame, boxes)` for every camera frame until the camera closes or `cleanup()` is called. |
| `log_detections` | `(csv_writer, boxes) → None` | Writes one CSV row per detected person to the given `csv.writer`. |
| `validate` | `(data_config="data.yaml") → None` | Runs YOLO validation and prints mAP50 / Recall. |
| `cleanup` | `() → None` | Releases the camera and destroys all OpenCV windows. Called automatically when the generator exits. |

---

# Road Map for Human recognition

This roadmap is designed to take you from a basic detection script to a professional-grade 3D vigilance system. We will leverage your existing "Memory Class" logic to track people across time and space.

---

## Phase 1: The "Eyes" (2D Detection)
**Goal:** Detect people and faces in a video stream.

* **Setup:** Install `ultralytics` (for YOLOv8n) and `opencv-python`.
* **Implementation:** * Run YOLOv8n on the RGB stream to find "person" classes.
    * **Pro Tip:** Use a second, smaller YOLOv8n-face model or a crop-based approach to get the bounding box of the face *inside* the person box. This saves processing power.
* **Output:** Bounding boxes $(x, y, w, h)$.

## Phase 2: The "Identity" (2D Recognition)
**Goal:** Turn a face image into a name.

* **Library:** Use **DeepFace** or **InsightFace**.
* **Database Setup:** Create a folder structure: `./db/person_name/photo1.jpg`.
* **Registration Script:** Create a simple script that captures 3-5 images of a person (Front, Left, Right) to create a robust "Multi-view" embedding.
* **Integration:**
    * Crop the face from the YOLO box.
    * Pass the crop to `DeepFace.find()`.
    * If a match is found, assign the name; otherwise, label as "Unknown".

## Phase 3: The "Memory" (State Tracking)
**Goal:** Use your `StateRecord` logic to keep track of who is who.

* **Class Update:** Your class should now store `PersonRecord`.
* **Logic:** If the system sees "User_A" at $T=1s$ and then at $T=2s$, it shouldn't just "recognize" them again; it should update their existing record.
* **Fields to add to your `StateRecord`:**
    * `person_id` (Name or Unique ID).
    * `last_seen_timestamp`.
    * `confidence_score`.
    * `is_spoof` (Boolean for Phase 4).

## Phase 4: The "3D Shield" (Depth Integration)
**Goal:** Use the Intel RealSense/LiDAR to prevent spoofing and clean data.

* **Alignment:** Use `rs.align` in the RealSense SDK to map Depth pixels to RGB pixels.
* **Liveness Check:** * Look at the depth values within the face bounding box.
    * Calculate the standard deviation of depth. A flat photo has near-zero variance; a human face has high variance (nose vs. ears).
* **Background Masking:** * Set a `clipping_distance` (e.g., 2 meters). 
    * Any pixel with a depth value $> 2m$ is turned to black before being sent to the recognition model. This removes background "noise" people.



## Phase 5: Testing and Optimization (The "Bag" Strategy)
**Goal:** Refine the system without needing the hardware plugged in 24/7.

* **Record Data:** Use the RealSense Viewer to record `.bag` files of different scenarios (someone wearing a mask, someone holding a photo, multiple people).
* **Development:** Use the `rs.config().enable_device_from_file()` method to run your code against these recordings. 
* **Refinement:** Adjust your "Optimal Driving" logic (from your previous class) to instead calculate "Optimal Vigilance"—for example, only trigger recognition when a person is within a certain distance or moving toward a restricted area.

---

### Summary of the System Flow
1.  **Input:** RealSense RGB + Depth frames (Aligned).
2.  **Detection:** YOLOv8n finds the person.
3.  **Depth Filter:** Is the object 3D? (Anti-spoofing).
4.  **Crop:** Extract the face, masked by depth.
5.  **Recognition:** Compare embedding against your `./db`.
6.  **Memory:** Update the `StateRecord` in your history list.
7.  **Export:** Occasionally run `export_to_csv()` to see a log of who entered the room and when.

**Would you like the specific code to "align" the RealSense depth and color frames to get you started on Phase 4?**