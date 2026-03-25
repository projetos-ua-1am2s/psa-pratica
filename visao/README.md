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