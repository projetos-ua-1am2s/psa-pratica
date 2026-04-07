# PSA Prática - MQTT Communication

## When to use this branch

This branch should be used for simulations. This allows for testing the communication between the detection program and the webots camera and controller. It can also be used to test the real pan&tilt or the real camera.

It won't allways have the latest version of the main detection software.

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
└── indoor/            ← Webots world and controllers
```

---

## Quick start

1. **Install dependencies**
   ```bash
   pip install opencv-python ultralytics torch
   ```

2. **Install the MQTT broker**
   
   On [mosquitto](https://mosquitto.org/download/) download the .exe file or follow the instructions corresponding to your OS.


4. **Install and open the Webots world**

   Install [Webots](https://cyberbotics.com)

   Open the .wbt file on indoor\worlds\apartment.wbt, it should open already running, you can pause it and start it over.


6. **Run the surveillance module**

   Run the simulation on webots and then run the detection program with:
   ```bash
   cd visao
   python main.py
   ```


4. **Stop the system**
   
   Press **`q`** in the video window, or **`Ctrl+C`** in the terminal.

   Then, pause the simulation on webots.


> See [`visao/README.md`](visao/README.md) for full usage details, configuration options, and the `PersonTracker` API reference.

---

## Problem solving

1. Ensure that all the paths are correct
2. Always start the simulation on webots before running the detection program
3. If the MQTT communication isn't working:
   - Create a text file and paste the following:
     ```bash
     @echo off
     cd "C:\Program Files\Mosquitto"
     mosquitto.exe -c mosquitto.conf -v
     pause
     ```
   - Make sure to replace the folder path and the .conf file name with the ones you do have (the .conf file should be inside the folder)
   - Save the text file as a .bat file
   - Open that file everytime you intend to work on this simulation and keep it running

---

## Tech stack

| Component | Library / Tool |
|-----------|---------------|
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Video capture & display | [OpenCV](https://opencv.org/) |
| Deep learning backend | [PyTorch](https://pytorch.org/) (MPS / CUDA / CPU) |
| Data logging | Python `csv` (standard library) |
| MQTT broker | [mosquitto](https://mosquitto.org/download/) |
| Webots | [Webots](https://cyberbotics.com) |

---

## Authors

Developed as part of the PSA practical sessions at the University of Aveiro.
