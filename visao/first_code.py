import cv2
from ultralytics import YOLO
import time
import torch
import csv # for performance reports


# important variables
conf_threshold = 0.3

# 1. Loads the model (o 'n' é de nano, o mais leve para Mac)
model = YOLO("yolov8n.pt") 

# 2. Define o dispositivo para o chip da Apple (M1/M2/M3)
# Se o teu Mac for muito antigo (Intel),
#  apaga a linha abaixo

# Deteta automaticamente o melhor hardware disponível
if torch.backends.mps.is_available():
    device = "mps"     # Mac M1/M2/M3
elif torch.cuda.is_available():
    device = "cuda"    # Windows/Linux com NVIDIA
else:
    device = "cpu"     # PC comum

print(f"A usar o dispositivo: {device}")

# 3. Abre a câmara do Mac
cap = cv2.VideoCapture(0)

# Verifica se a câmara foi aberta corretamente
if not cap.isOpened():
    print("Error in opening the camera. Verifica se a câmara está ligada e se a aplicação tem permissões para a utilizar.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit(1)
print("A iniciar vigilância... Prime 'q' para sair.")
with open("surveillance_data.csv", 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp", "ID", "Confidence", "Status"])  # Header
    try:
        while cap.isOpened():
            initial_time = time.time()
            success, frame = cap.read()
            if not success:
                break

            # 4. Detection and tracking
            # classes=[0] filters tracking to 'People'
            results = model.track(frame, persist=True, conf=0.3, device=device, classes=[0]) # conf defines the threshold, thus anything with less that x% of conf won't be considered as a positive.
            # low conf threshold to minimize false negatives seeing as they would be more problematic in a security system


            # 4.1. Saving data in csv file
            if results[0].boxes is not None: # Avoiding saving empty data
                for box in results[0].boxes: # stepping through every result
                    # extracting data
                    conf = float(box.conf[0])

                    # ID may be None if the tracker hasn't provided one yet
                    track_id = int(box.id[0]) if box.id is not None else "N/A"

                    # Decidir o Status para o CSV
                    status = "Accepted" if conf >= conf_threshold else "Rejected"

                    # Storing in file
                    writer.writerow([time.strftime("%H:%M:%S"), track_id, f"{conf:.2f}", status])


            # 5. IDs and frames (squares)
            annotated_frame = results[0].plot()
            final_time = time.time()

            # 6. View window
            cv2.imshow("PSA surveillance camera", annotated_frame)

            # 7. Fps print in the terminal
            total_time = final_time - initial_time
            if total_time > 0:
                fps = 1 / total_time
            else:
                fps = 0.0
            print(f"FPS: {fps:.2f} and total elapsed time for frame: {total_time:.2f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# getting metrics for the model
metrics = model.val(data="data.yaml")
# 3. Imprime os resultados principais
print(f"Precisão (mAP50): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")