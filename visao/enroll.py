import cv2
import os
import numpy as np
from ultralytics import YOLO
import re


difference_threshold_definition = 10000.0  # Tweak this. Higher = requires more head movement.

def calculate_mse(imageA, imageB):
    # Resize to identical dimensions for pixel math
    imgA = cv2.resize(imageA, (100, 100))
    imgB = cv2.resize(imageB, (100, 100))

    # Quadratic difference (Mean Squared Error)
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err


def sanitize_person_name(person_name):
    # Allow only safe filename characters and normalize everything else to '_'.
    sanitized_name = re.sub(r"[^A-Za-z0-9_-]", "_", person_name).strip("_")
    if sanitized_name.lower() == "none":
        return ""
    return sanitized_name


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "known_faces")
    db_path_abs = db_path
    os.makedirs(db_path, exist_ok=True)

    raw_person_name = input("Enter person name (e.g., john_doe): ").strip()
    person_name = sanitize_person_name(raw_person_name)
    if not person_name:
        print("Name empty or invalid after sanitization. Exit.")
        return

    # Use same face model as tracker
    face_model = YOLO(os.path.join(base_dir, "yolov8n-face.pt"))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    saved_crops = []
    max_images = 3
    mse_threshold = difference_threshold_definition  # Tweak this. Higher = requires more head movement.
    # initial value was 4000 (requires very little movement)

    print(f"\n--- Enrolling: {person_name} ---")
    print("Look at camera. Press 's' to attempt save.")
    print("Move head (front, left, right) between saves.")
    print("Press 'q' to quit early.")

    while len(saved_crops) < max_images:
        success, frame = cap.read()
        if not success:
            break

        results = face_model(frame, conf=0.5, verbose=False)
        annotated_frame = frame.copy()
        current_crop = None

        # Process first detected face
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bound checks
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                current_crop = frame[y1:y2, x1:x2]

        # UI Text
        status_text = f"Saved: {len(saved_crops)}/{max_images}"
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Enrollment", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_crop is not None and current_crop.size > 0:
                is_different = True

                # Check quadratic difference against all previously saved crops
                for saved_img in saved_crops:
                    diff = calculate_mse(current_crop, saved_img)
                    print(f"MSE vs saved image: {diff:.2f}")

                    if diff < mse_threshold:
                        is_different = False
                        print("[REJECTED] Picture too similar. Move head.")
                        break

                if is_different:
                    saved_crops.append(current_crop)
                    file_name = f"{person_name}_{len(saved_crops)}.jpg"
                    file_path = os.path.abspath(os.path.join(db_path, file_name))

                    # Enforce that the final path remains inside known_faces.
                    if os.path.commonpath([db_path_abs, file_path]) != db_path_abs:
                        print("[ERROR] Refusing to write outside known_faces.")
                        continue

                    cv2.imwrite(file_path, current_crop)
                    print(f"[SAVED] {file_path}")
            else:
                print("[ERROR] No face detected. Cannot save.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nEnrollment complete.")


if __name__ == "__main__":
    main()