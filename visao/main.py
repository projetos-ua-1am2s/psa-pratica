import cv2
from ultralytics import YOLO

# 1. Carrega o modelo (o 'n' é de nano, o mais leve para Mac)
model = YOLO("yolov8n.pt") 

# 2. Define o dispositivo para o chip da Apple (M1/M2/M3)
# Se o teu Mac for muito antigo (Intel),
#  apaga a linha abaixo
import torch

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
    print("Erro ao abrir a câmara. Verifica se a câmara está ligada e se a aplicação tem permissões para a utilizar.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit(1)
print("A iniciar vigilância... Prime 'q' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. Faz a deteção e o rastreamento usando o hardware da Apple
    # classes=[0] filtra apenas para detetar 'Pessoas'
    results = model.track(frame, persist=True, device=device, classes=[0])

    # 5. Desenha os quadrados e IDs no frame
    annotated_frame = results[0].plot()

    # 6. Janela de visualização
    cv2.imshow("Vigilancia Mac IA", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()