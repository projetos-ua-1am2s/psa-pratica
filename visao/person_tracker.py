import cv2
import torch
import time
import os
import math
from ultralytics import YOLO


class PersonTracker:
    """
    A class to handle person detection and tracking using YOLOv8.
    Thus creating a cleaner way to import code into other packages.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.3, accept_threshold=None):
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
        self.performance_log_interval = 1.0
        self._last_perf_print_time = 0.0

        # Resolve model_path relative to this file if it is not absolute
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, model_path)

        self.model = YOLO(model_path)
        self.cap = None

        print(f"Using device: {self.device}")

    def _get_device(self):
        """Internal method to detect the best available hardware."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _setup_camera(self):
        """Initializes the camera and checks if it's working."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")
        print("Surveillance started... Press 'q' to quit.")




    def _process_face_recognition(self, frame, annotated_frame, person_box):
        """
        Handles cropping, face detection (optional), and drawing the face box.
        """
        px1, py1, px2, py2 = person_box

        # 1. Create the person crop
        person_crop = frame[py1:py2, px1:px2]

        if person_crop.size == 0:
            return

        # 2. Strategy: Identify Face Area
        # For now, let's use the 'top-30%' rule to simulate where the face is.
        # Later, you can replace this with: face_coords = face_model.predict(person_crop)
        face_h = int((py2 - py1) * 0.30)

        # 3. Global Coordinates for the Blue Box
        # We add px1 and py1 to map the crop-relative position back to the full frame
        fx1, fy1 = px1 + 5, py1 + 5
        fx2, fy2 = px2 - 5, py1 + face_h

        # 4. Draw directly on the annotated_frame
        cv2.rectangle(
            annotated_frame,
            (fx1, fy1),
            (fx2, fy2),
            (255, 0, 0),  # Blue BGR
            2
        )

        # 5. Potential Identity Text (Phase 2)
        # cv2.putText(annotated_frame, "Identifying...", (fx1, fy1-10), ...)



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
            self._setup_camera()

            while self.cap.isOpened():
                start_time = time.time()
                success, frame = self.cap.read()

                if not success:
                    break

                # Tracking only class 0 (People)
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.track_conf,
                    device=self.device,
                    classes=[0]
                )

                vector = None
                boxes = None
                annotated_frame = results[0].plot()

                # new -- calculating movement vector
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes

                    # calculating movement vector
                    vector = self._get_movement_vector(frame, boxes)

                    # Draw visual debug info on the annotated frame
                    if vector:
                        h, w, _ = frame.shape
                        obj_x, obj_y = boxes[0].xywh[0][:2]
                        cv2.line(annotated_frame, (int(w / 2), int(h / 2)), (int(obj_x), int(obj_y)), (0, 255, 0),
                                 2)
                        cv2.putText(annotated_frame, f"V: {vector[0]} @ {vector[1]}deg",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    for box in results[0].boxes:
                        # 2. DEFINE p_box HERE
                        # We take the coordinates, convert to list, and ensure they are integers >= 0
                        coords = box.xyxy[0].tolist()
                        p_box = [max(0, int(c)) for c in coords]  # This creates the [x1, y1, x2, y2] list

                        # 3. Now you can safely call the method
                        self._process_face_recognition(frame, annotated_frame, p_box)


                self._display_performance(start_time)

                # Yield the data to the external loop
                yield vector, annotated_frame, boxes

        finally:
            self.cleanup()

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
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

    def _get_movement_vector(self, frame, boxes):
        """
        Calculates the [magnitude, angle] vector for the person with the first ID
        """
        if boxes is None or len(boxes) == 0:
            return None

        # 1. Center of Frame
        h, w, _ = frame.shape
        c_x, c_y = w / 2, h / 2


        # Target the first person in the list (tracked ID)
        obj_x, obj_y, _, _ = tuple(float(v) for v in boxes[0].xywh[0])

        dx = obj_x - c_x
        dy = obj_y - c_y

        # Normalized Magnitude (0.0 to 1.0)
        max_dist = math.sqrt(c_x ** 2 + c_y ** 2)
        magnitude = math.sqrt(dx ** 2 + dy ** 2) / max_dist

        # 5. Angle in degrees (-180 a 180)
        # 0º is right, 90º is down, -90º is up
        angle = math.degrees(math.atan2(dy, dx))

        return [round(float(magnitude), 3), round(float(angle), 2)]