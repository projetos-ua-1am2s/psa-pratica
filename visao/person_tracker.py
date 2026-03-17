import cv2
import torch
import time
import csv
import os
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

    def run(self, output_file="surveillance_data.csv"):
        """Main loop for real-time tracking and data logging."""
        try:
            self._setup_camera()

            with open(output_file, 'w', newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Timestamp", "ID", "Confidence", "Status"])

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

                    # Log data if detections exist
                    if results[0].boxes is not None:
                        self._log_detections(writer, results[0].boxes)

                    # Visuals
                    annotated_frame = results[0].plot()
                    cv2.imshow("PSA Surveillance Camera", annotated_frame)

                    # Performance metrics
                    self._display_performance(start_time)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.cleanup()

    def _log_detections(self, csv_writer, boxes):
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


# --- How to use it ---
if __name__ == "__main__":
    tracker = PersonTracker()
    tracker.run()
    # tracker.validate() # Uncomment if you want to validate after running