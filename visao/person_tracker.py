import cv2
import torch
import time
import csv
from ultralytics import YOLO


class PersonTracker:
    """
    A class to handle person detection and tracking using YOLOv8.
    Thus creating a cleaner way to import code into other packages.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.device = self._get_device()
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
        self._setup_camera()

        with open(output_file, 'w', newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Timestamp", "ID", "Confidence", "Status"])

            try:
                while self.cap.isOpened():
                    start_time = time.time()
                    success, frame = self.cap.read()

                    if not success:
                        break

                    # Tracking only class 0 (People)
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.conf_threshold,
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
            status = "Accepted" if conf >= self.conf_threshold else "Rejected"

            csv_writer.writerow([
                time.strftime("%H:%M:%S"),
                track_id,
                f"{conf:.2f}",
                status
            ])

    def _display_performance(self, start_time):
        """Calculates and prints FPS and frame processing time."""
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        print(f"FPS: {fps:.2f} | Time per frame: {elapsed_time:.3f}s")

    def validate(self, data_config="data.yaml"):
        """Runs model validation metrics."""
        print("Running validation...")
        metrics = self.model.val(data=data_config)
        print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")

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