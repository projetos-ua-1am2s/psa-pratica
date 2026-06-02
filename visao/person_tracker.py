import cv2
import torch
import time
import os
import math
import collections
from ultralytics import YOLO
import numpy as np
from typing import Optional, Tuple
# MQTT
import json
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from queue import Queue

# Face recognition
try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None
import threading

FACE_QUEUE_POLL_INTERVAL = 0.1 # waiting time

class PersonTracker:
    """
    A class to handle person detection and tracking using YOLOv8.
    Thus creating a cleaner way to import code into other packages.
    """

    def __init__(self,
                 model_path="yolov8n.pt",
                 face_model_path="yolov8n-face.pt",  # e.g. from akanametov/yolov8-face
                 conf_threshold=0.3,
                 accept_threshold=None,
                 face_queue_size=10,
                 input_source="camera",       # camera or mqtt
                 use_mqtt_out=False,        # True to publish vectors
                 mqtt_broker="localhost",
                 auto_enroll=False        # For Enrolling code section activation
                 ):
        self.conf_threshold = conf_threshold
        # Threshold used to classify detections as Accepted/Rejected in logs.
        # Defaults to the model's confidence threshold if not explicitly set.
        self.accept_threshold = accept_threshold if accept_threshold is not None else conf_threshold


        # Internal tracking confidence: lower than accept_threshold so that
        # some detections can be logged as "Rejected" instead of being filtered
        # out by the model itself.
        self.track_conf = self.accept_threshold / 2.0
        self.device = self._get_device()

        # Interval (in seconds) between performance log prints.
        # This prevents per-frame printing from becoming a bottleneck.
        self.performance_log_interval = 2.0 # second interval in between prints
        self._last_perf_print_time = 0.0

        # variables for face detection
        self._frame_face_skip = 0 # to store frames passed to avoid computing face model every frame
        # Cache for face bounding boxes and confidence scores.
        # Always normalized to 5-tuples: (fx1, fy1, fx2, fy2, conf_val)
        # where conf_val can be None if not available.
        self._last_face_boxes = []

        # ===== Font rendering constants (hoisted to avoid per-frame recreation) =====
        # Font parameters for face detection labels
        self._face_label_font = cv2.FONT_HERSHEY_SIMPLEX
        self._face_label_font_scale = 0.6
        self._face_label_thickness = 2

        # Font parameters for person name labels
        self._person_name_font = cv2.FONT_HERSHEY_SIMPLEX
        self._person_name_font_scale = 0.8
        self._person_name_thickness = 2

        # Font parameters for debug/info text (e.g., "Recognized:", vector info)
        self._debug_info_font = cv2.FONT_HERSHEY_SIMPLEX
        self._debug_info_font_scale = 1.0
        self._debug_info_thickness = 2

        self._model_path = model_path
        self._face_model_path = face_model_path

        # Resolve model paths relative to this file if they are not absolute
        for attr, path in [("_model_path", model_path), ("_face_model_path", face_model_path)]:
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), path)
            setattr(self, attr, path)

        self.model = YOLO(self._model_path)
        self.face_model = YOLO(self._face_model_path)
        self.cap = None

        # LIFO queue: deque used as a stack (append right, pop right)
        # LIFO --> last in first out method used to control runs
        self._person_stack: collections.deque = collections.deque(maxlen=face_queue_size)
        self._face_stack: collections.deque = collections.deque(maxlen=face_queue_size)

        print(f"Using device: {self.device}")

        # ----- MQTT Configuration -----
        self.input_source = input_source.lower()
        self.use_mqtt_out = use_mqtt_out
        self.client = None
        self.frame_queue = None

        if self.input_source == "mqtt" or use_mqtt_out:
            try:
                # ?? self.client = mqtt.Client(CallbackAPIVersion.VERSION2, "VisionBrain")
                # Construct client with a readable client_id. Use named-arg
                # to avoid passing ordered params incorrectly across paho versions.
                self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id="VisionBrain")
                self.client.on_connect = self._on_connect
                self.mqtt_broker = mqtt_broker
                if self.input_source == "mqtt":
                    # on_message callback signature is (client, userdata, message)
                    self.client.on_message = self._on_message
                    self.frame_queue = Queue(maxsize=1)     # Stores the most recent frame
                self.client.connect(self.mqtt_broker, 1883, 60)
                self.client.loop_start()

            except Exception as e:
                print(f"Error connecting to MQTT broker: {e}")
                raise
        # ------------------------------

        # ======= face recognition
        self.face_db_path = os.path.join(os.path.dirname(__file__), "known_faces")
        os.makedirs(self.face_db_path, exist_ok=True)
        self._stop_event = threading.Event()
        self._active_track_ids = set()
        # dictionary allows to identify multiple people in a single frame
        # storing the name of the recognized person with the corresponding track_id
        self.known_names = {} # Dictionary holds 'ID -> Name'
        self._known_names_lock = threading.Lock()
        self._face_recognition_enabled = DeepFace is not None
        self._face_recognition_thread = None
        if self._face_recognition_enabled:
            self._face_recognition_thread = threading.Thread(target=self._face_recognition_worker, daemon=True)
            self._face_recognition_thread.start()
        else:
            print("DeepFace is not installed; face recognition is disabled.")

        # ======= face recognition end
        self.auto_enroll = auto_enroll
        self.trigger_enroll = False # Flag to signal enrollment process for unknown faces

    @staticmethod
    def _get_device():
        """Internal method to detect the best available hardware."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ----- MQTT Communication - Frame Reception -----

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        # paho.mqtt (CallbackAPIVersion.VERSION2) on_connect callback signature:
        # (client, userdata, flags, reason_code, properties)
        if reason_code == 0:
            print(f"Connected to MQTT Broker! (Return Code: {reason_code})")
            client.subscribe("Camera")
        else:
            print(f"Failed to connect, return code {reason_code}")

    def _on_message(self, client, userdata, msg):
        # This function is called whenever a frame is received via MQTT
        # Convert received byte payload into a NumPy array
        nparr = np.frombuffer(msg.payload, np.uint8)
        # Decode the NumPy array into an OpenCV image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Keep only the latest frame to avoid processing delays
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
            self.frame_queue.put(frame)
    # ------------------------------

    def _setup_camera(self):
        """Initializes the camera and checks if it's working."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")
    

    def log_detections(self, csv_writer, boxes):
        """Helper to write detection data to CSV."""
        for box in boxes:
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else "N/A"
            status = "Accepted" if conf >= self.accept_threshold else "Rejected"

            csv_writer.writerow([
                time.strftime("%H:%M:%S"),
                track_id,
                f"{conf:.2f}",
                status
            ])

    def _display_performance(self, start_time):
        """Calculates and periodically prints FPS and frame processing time."""
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Throttle performance printing to avoid per-frame stdout overhead.
        now = time.time()
        if now - self._last_perf_print_time >= self.performance_log_interval:
            print(f"FPS: {fps:.2f} | Time per frame: {elapsed_time:.3f}s")
            self._last_perf_print_time = now

    def validate(self, data_config="data.yaml"):
        """Runs model validation metrics."""
        # Resolve data_config relative to this file for non-absolute paths,
        # to avoid dependence on the caller's current working directory.
        if not os.path.isabs(data_config):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(base_dir, data_config)

        print("Running validation...")
        metrics = self.model.val(data=data_config, device=self.device)

        # Safely access metrics to avoid KeyError if Ultralytics changes keys or task type
        results_dict = getattr(metrics, "results_dict", None)
        if isinstance(results_dict, dict):
            map50 = results_dict.get("metrics/mAP50(B)")
            recall = results_dict.get("metrics/recall(B)")

            if map50 is not None and recall is not None:
                print(f"mAP50: {map50:.4f}")
                print(f"Recall: {recall:.4f}")
            else:
                available_keys = ", ".join(results_dict.keys())
                print(
                    "Requested validation metrics 'metrics/mAP50(B)' and/or "
                    "'metrics/recall(B)' are not available in results_dict. "
                    f"Available keys: {available_keys}"
                )
        else:
            print("Validation results do not contain a 'results_dict' dictionary; "
                  "cannot report mAP50 and Recall metrics.")

    def cleanup(self):
        """Releases resources properly."""
        self._stop_event.set() # to stop background thread
        if hasattr(self, "_face_recognition_thread") and self._face_recognition_thread.is_alive():
            self._face_recognition_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

    @staticmethod
    def _get_movement_vector(frame, boxes):
        """
        Calculates the [magnitude, angle] vector for the person with the first ID
        """
        if boxes is None or len(boxes) == 0:
            return None

        # 1. Center of Frame
        h, w, _ = frame.shape
        c_x, c_y = w / 2, h / 2


        # Target the first person in the list (tracked ID)
        # Target top 30% of first person box
        c_obj_x, c_obj_y, box_w, box_h = tuple(float(v) for v in boxes[0].xywh[0])

        obj_x = c_obj_x
        # center Y - 50% height + 30% height = center Y - 20% height
        obj_y = c_obj_y - (box_h * 0.2)

        dx = obj_x - c_x
        dy = obj_y - c_y

        # Normalized Magnitude (0.0 to 1.0)
        max_dist = math.sqrt(c_x ** 2 + c_y ** 2)
        magnitude = math.sqrt(dx ** 2 + dy ** 2) / max_dist

        # 5. Angle in degrees (-180 a 180)
        # 0º is right, 90º is down, -90º is up
        angle = math.degrees(math.atan2(dy, dx))

        return [round(float(magnitude), 3), round(float(angle), 2)]

    def push_face(self, face_crop: Tuple[np.ndarray, int]):
        # Push a face crop onto the LIFO stack.
        self._face_stack.append(face_crop)

    def pop_face(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Pop the most-recently-added face crop (LIFO).
        Returns None if the stack is empty.
        """
        if self._face_stack:
            return self._face_stack.pop()
        return None

    def _process_faces(self, frame, boxes, annotated_frame):
        """
        Runs face detection on each person crop (not full frame).
        Draws face boxes on annotated_frame, pushes crops to LIFO stack.
        Returns updated annotated_frame and list of face crops.
        """
        face_box = []
        face_crops = []

        if boxes is None:
            return annotated_frame, face_crops, []

        for box in boxes:
            track_id = int(box.id[0]) if box.id is not None else None # for face recognition

            # ===== SKIP Recognition IF PERSON ALREADY RECOGNIZED =====
            with self._known_names_lock:
                if track_id in self.known_names:
                    continue
            # ==================================================

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]

            face_results = self.face_model(person_crop, conf=self.conf_threshold, device=self.device, verbose=False)
            if face_results[0].boxes is None:
                continue

            # Instead of pasting crop back, draw directly on annotated_frame with offset coords
            for fbox in face_results[0].boxes:
                fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                fx1, fx2 = fx1 + x1, fx2 + x1
                fy1, fy2 = fy1 + y1, fy2 + y1

                fx1 = max(0, fx1)
                fy1 = max(0, fy1)
                fx2 = min(frame.shape[1], fx2)
                fy2 = min(frame.shape[0], fy2)

                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                # Draw red face box (BGR)
                box_color = (0, 0, 255)
                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), box_color, 2)

                # Build label and draw white text on a red background above the box
                conf_val = float(fbox.conf[0])
                label = f"Face {conf_val:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, self._face_label_font, self._face_label_font_scale, self._face_label_thickness)

                rect_x1 = fx1
                rect_x2 = fx1 + tw + 6
                rect_y2 = fy1 - 4
                rect_y1 = rect_y2 - th - baseline - 4

                # If there's not enough space above, clamp to top edge
                if rect_y1 < 0:
                    rect_y1 = max(0, fy1)
                    rect_y2 = rect_y1 + th + baseline + 6

                rect_x1 = max(0, rect_x1)
                rect_y1 = max(0, rect_y1)
                rect_x2 = min(annotated_frame.shape[1], rect_x2)
                rect_y2 = min(annotated_frame.shape[0], rect_y2)

                # Filled red rectangle background
                cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), box_color, -1)

                # Put white text on top
                text_org = (rect_x1 + 3, rect_y2 - baseline - 3)
                cv2.putText(annotated_frame, label, text_org, self._face_label_font, self._face_label_font_scale, (255, 255, 255), self._face_label_thickness, cv2.LINE_AA)

                face_crop = frame[fy1:fy2, fx1:fx2]

                # face recognition loop
                # will pop crops from the stack, so we push them here. The worker thread will handle recognition asynchronously.
                if face_crop.size > 0:
                    face_crops.append(face_crop)
                    if track_id is not None:
                        self.push_face((face_crop, track_id))  # Push tuple of (crop, track_id) for recognition thread

                # Store coordinates and confidence for later drawing on skipped frames
                face_box.append((fx1, fy1, fx2, fy2, conf_val))

        return annotated_frame, face_crops, face_box

    # ==== face recognition
    def face_recognition(self, face_crop, track_id=None):
        if face_crop.size == 0:
            return []

        try:
            matches = DeepFace.find(
                img_path=face_crop,
                db_path=self.face_db_path,
                model_name="SFace",
                enforce_detection=False,
                silent=True,

            )
            return matches
        except Exception as e:
            shape = getattr(face_crop, "shape", None)
            print(
                f"DeepFace error during face recognition for track_id={track_id}, "
                f"face_crop_shape={shape}, db_path='{self.face_db_path}': {e}"
            )
            return []

    # running in a multithreaded way to avoid blocking the main loop with face recognition processing
    def _face_recognition_worker(self):
        while not self._stop_event.is_set():  # Run until cleanup signals stop
            data = self.pop_face()
            if data is not None:
                face_crop, track_id = data # unpacking tuple from processing_faces method
                matches = self.face_recognition(face_crop,  track_id=track_id)

                if matches is not None and len(matches) > 0 and not matches[0].empty:
                    # Get file path from 'identity' column
                    file_path = matches[0]['identity'].iloc[0]
                    # Positional index used to be 0, but if DeepFace returns a DataFrame with multiple matches,
                    # we take the first one (iloc[0] -- relative position)

                    name = os.path.splitext(os.path.basename(file_path))[0]
                    # print(f"[FACE] ID {track_id} is {name}")

                    with self._known_names_lock:
                        if track_id in self._active_track_ids:
                            self.known_names[track_id] = name  # Save to dict

                else:
                    # NEW: Unknown face logic
                    # print(f"[FACE] ID {track_id} is Unknown")

                    if self.auto_enroll:  # --- CHECK MASTER SWITCH ---
                        with self._known_names_lock:
                            if track_id not in self.known_names:
                                self.known_names[track_id] = "Enrolling..."  # Block duplicate triggers
                                self.trigger_enroll = True


            else:
                time.sleep(FACE_QUEUE_POLL_INTERVAL)  # Wait if stack empty. Save CPU.

    # ====== face recognition end

    def run(self):
        """
        Generator that processes frames and yields (vector, frame, boxes).

        Vector format: [magnitude (0-1), angle (degrees)]
        boxes: Ultralytics Boxes object for the current frame, or None when no
               detections are present. Callers that want to log detections to a
               CSV file should open the file themselves and pass the resulting
               ``csv.writer`` to :meth:`log_detections`.

        Example::

            with open("out.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "ID", "Confidence", "Status"])
                for vector, frame, boxes in tracker.run():
                    if boxes is not None:
                        tracker.log_detections(writer, boxes)
                    # … display / act on frame …
        """
        try:
            # NEW INPUT BRANCH to handle mqtt integration
            if self.input_source == "camera":
                self._setup_camera()
            else:
                print("Processing started... Waiting for MQTT frames.")

            while True:
                start_time = time.time()

                # --- AUTO ENROLL TRIGGER (NO CAMERA RELEASE) ---
                if getattr(self, 'trigger_enroll', False):
                    print("\n[SYSTEM] Unknown detected. Launching enroll.py...")

                    import subprocess
                    import sys
                    subprocess.run([sys.executable, "enroll.py"])

                    # Delete DeepFace cache so it finds new photos
                    pkl_path = os.path.join(self.face_db_path, "representations_sface.pkl")
                    if os.path.exists(pkl_path):
                        os.remove(pkl_path)
                        print("[SYSTEM] Deleted DeepFace cache.")

                    self.trigger_enroll = False
                    print("[SYSTEM] Resuming tracker...\n")
                    continue
                # ---------------------------

                # NEW FRAME READ LOGIC
                if self.input_source == "camera":
                    if not self.cap.isOpened():
                        break
                    success, frame = self.cap.read()
                    if not success:
                        break
                else:
                    try:
                        if self.frame_queue is None:
                            raise RuntimeError("MQTT frame_queue not initialized. Check input_source configuration.")
                        frame = self.frame_queue.get(timeout=5)
                    except:
                        continue
                    if frame is None:
                        continue

                # Tracking only class 0 (People)
                person_results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.track_conf,
                    device=self.device,
                    classes=[0],
                    verbose=False
                )

                vector = None
                boxes = None

                # 1. assign boxes first
                if person_results[0].boxes is not None and len(person_results[0].boxes) > 0:
                    boxes = person_results[0].boxes
                    self._person_stack.append(boxes)  # LIFO push for persons
                    vector = self._get_movement_vector(frame, boxes)

                # --- CLEANUP CODE for IDs ---
                # When a ID leaves frame we need to remove it from the database
                current_ids = set()
                if boxes is not None:
                    current_ids = {int(box.id[0]) for box in boxes if box.id is not None}

                with self._known_names_lock:
                    self._active_track_ids = set(current_ids)
                    # line responsible for the cleanup of the known_names dict, it checks if the track_ids currently
                    # in the dict are still present in the current frame's detections. If not, it removes them from the
                    # dict to prevent stale data.
                    stale_ids = [tid for tid in self.known_names.keys() if tid not in current_ids]
                    for tid in stale_ids:
                        del self.known_names[tid]
                # -----------------------------

                # 2. plot persons
                annotated_frame = person_results[0].plot()

                # 2.1 because of face recognition
                if boxes is not None:
                    for box in boxes:
                        track_id = int(box.id[0]) if box.id is not None else -1
                        x1, y1 = map(int, box.xyxy[0][:2])  # Get top-left corner

                        # Read known_names atomically under the lock because the worker
                        # thread may update the dictionary concurrently.
                        with self._known_names_lock:
                            name = self.known_names.get(track_id)

                        if name is not None:
                            label = f"{name}"
                            (tw, th), baseline = cv2.getTextSize(label, self._person_name_font, self._person_name_font_scale, self._person_name_thickness)

                            # Draw a filled rectangle as background for the name label
                            rect_x1 = x1
                            rect_y1 = y1 - th - baseline - 10
                            rect_x2 = x1 + tw + 10
                            rect_y2 = y1

                            # Ensure the rectangle is within frame bounds
                            rect_x1 = max(0, rect_x1)
                            rect_y1 = max(0, rect_y1)
                            rect_x2 = min(annotated_frame.shape[1], rect_x2)
                            rect_y2 = min(annotated_frame.shape[0], rect_y2)

                            # Draw filled rectangle (BGR: Blue background)
                            cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), -1)

                            # Put white text on top of the rectangle
                            text_org = (rect_x1 + 5, rect_y2 - baseline - 5)
                            cv2.putText(annotated_frame, label, text_org, self._person_name_font, self._person_name_font_scale, (255, 255, 255), self._person_name_thickness, cv2.LINE_AA)

                # ========== face recognition logic
                with self._known_names_lock:
                    recognized_name = next(iter(self.known_names.values()), None)

                if recognized_name:
                    # Display one recognized name on the video feed.
                    cv2.putText(annotated_frame,
                                f"Recognized: {recognized_name}",
                                (10, 60),
                                self._debug_info_font,
                                self._debug_info_font_scale,
                                (255, 255, 0),
                                self._debug_info_thickness)


                if self._frame_face_skip > 3:
                    # 3. now boxes is set — process faces on person crops
                    if boxes is not None:
                        annotated_frame, face_crops, self._last_face_boxes = self._process_faces(frame, boxes, annotated_frame)
                    else:
                        face_crops = []
                        self._last_face_boxes = []
                    self._frame_face_skip = 0

                else:
                    self._frame_face_skip += 1
                    face_crops = []
                    if boxes is not None:
                        # Draw cached face boxes from previous frame (all normalized to 5-tuple format)
                        for fx1, fy1, fx2, fy2, conf_val in self._last_face_boxes:
                            # Draw red face box (BGR)
                            box_color = (0, 0, 255)
                            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), box_color, 2)

                            # Draw confidence label if available
                            if conf_val is not None:
                                label = f"Face {conf_val:.2f}"
                                (tw, th), baseline = cv2.getTextSize(label, self._face_label_font, self._face_label_font_scale, self._face_label_thickness)

                                rect_x1 = fx1
                                rect_x2 = fx1 + tw + 6
                                rect_y2 = fy1 - 4
                                rect_y1 = rect_y2 - th - baseline - 4

                                # If there's not enough space above, clamp to top edge
                                if rect_y1 < 0:
                                    rect_y1 = max(0, fy1)
                                    rect_y2 = rect_y1 + th + baseline + 6

                                rect_x1 = max(0, rect_x1)
                                rect_y1 = max(0, rect_y1)
                                rect_x2 = min(annotated_frame.shape[1], rect_x2)
                                rect_y2 = min(annotated_frame.shape[0], rect_y2)

                                # Filled red rectangle background
                                cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), box_color, -1)

                                # Put white text on top
                                text_org = (rect_x1 + 3, rect_y2 - baseline - 3)
                                cv2.putText(annotated_frame, label, text_org, self._face_label_font, self._face_label_font_scale, (255, 255, 255), self._face_label_thickness, cv2.LINE_AA)
                    else:
                        self._last_face_boxes = []


                # 4. draw vector debug line
                if vector:
                    # handling mqtt output if enabled, publishing the vector as a JSON string to the "Movement" topic
                    if self.use_mqtt_out:
                        payload = json.dumps({"magnitude": vector[0], "angle": vector[1]})
                        self.client.publish("Movement", payload)

                    h, w, _ = frame.shape

                    # Calculate same top 30% point for drawing
                    c_obj_x, c_obj_y, box_w, box_h = boxes[0].xywh[0]
                    obj_x = int(c_obj_x)
                    obj_y = int(c_obj_y - (box_h * 0.2))

                    cv2.line(annotated_frame, (int(w / 2), int(h / 2)), (obj_x, obj_y), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"V: {vector[0]} @ {vector[1]}deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Results and outputs
                self._display_performance(start_time)

                # Yield the data to the external loop
                yield vector, annotated_frame, boxes

        finally:
            self.cleanup()