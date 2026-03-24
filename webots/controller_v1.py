import os
import sys

# Faz o mesmo que o my_controller no webots, mas com a versão inicial da deteção de pessoas

# 1. Configurar o caminho do Webots (ATENÇÃO: confirma se o caminho no teu PC é este!)
# Se o Webots estiver instalado noutro sítio, altera a variável webots_path.
webots_path = 'C:/Program Files/Webots'

print(f"1. A procurar Webots na pasta: {webots_path}")

if not os.path.exists(webots_path):
    print("ERRO FATAL: A pasta do Webots não existe nesse caminho! Tens de descobrir onde o instalaste.")
else:
    print("2. Pasta do Webots encontrada!")
    
    # Adicionar o caminho do Python do Webots
    python_path = os.path.join(webots_path, 'lib', 'controller', 'python')
    sys.path.append(python_path)
    print(f"3. Caminho do Python adicionado: {python_path}")
    
    # --- O TRUQUE MÁGICO PARA O WINDOWS ---
    # Adicionar a pasta das DLLs do Webots para o Python não bloquear
    dll_path = os.path.join(webots_path, 'msys64', 'mingw64', 'bin')
    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
    try:
        os.add_dll_directory(dll_path)
        print("4. Caminho das DLLs adicionado com sucesso!")
    except Exception as e:
        print(f"Aviso nas DLLs: {e}")
    # --------------------------------------

    print("5. A tentar importar o Robot...")
    try:
        from controller import Robot
        print("\n SUCESSO ABSOLUTO! O VS Code já consegue falar com o Webots!")
    except Exception as e:
        print(f"\n ERRO AO IMPORTAR: {e}")

os.environ['WEBOTS_HOME'] = webots_path
sys.path.append(os.path.join(webots_path, 'lib', 'controller', 'python'))

# 2. Dizer ao programa para se ligar à porta padrão do Webots
# O nome 'robot' aqui deve ser o nome que aparece na mensagem do terminal do Webots
os.environ['WEBOTS_CONTROLLER_URL'] = 'tcp://127.0.0.1:1234/robot'
#from controller import Robot
import numpy as np
import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt") 

import torch

# Deteta automaticamente o melhor hardware disponível
if torch.backends.mps.is_available():
    device = "mps"     # Mac M1/M2/M3
elif torch.cuda.is_available():
    device = "cuda"    # Windows/Linux com NVIDIA
else:
    device = "cpu"     # PC comum


#--------------------------------------------------------
# Inicialização dos robos e variaveis
#--------------------------------------------------------

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("tracker_camera")
camera.enable(timestep)

motor_pan = robot.getDevice("pan_motor")
motor_pan.setPosition(float('inf'))
motor_pan.setVelocity(0.0)

motor_tilt = robot.getDevice("tilt_motor")
motor_tilt.setPosition(float('inf'))
motor_tilt.setVelocity(0.0)

width = camera.getWidth()
height = camera.getHeight()

frame_counter = 0
velocidade_pan = 0.0
velocidade_tilt = 0.0
caixa_atual = None


#-----------------------------------------------------------
# Principal
#-----------------------------------------------------------

print("A iniciar a câmara e o rastreio...")

while robot.step(timestep) != -1:
    try:
        image = camera.getImage()

        if image:
            # 1. Preparar a imagem do Webots
            img_bgra = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            img = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, (640, 480))

            # 2. Deteção e Rastreio com o YOLO
            # verbose=False esconde as mensagens de texto que inundam o terminal em cada frame
            results = model.track(img, persist=True, device=device, classes=[0], verbose=False)
            
            # 3. Extrair as caixas de deteção para mexer os motores
            boxes = results[0].boxes.xyxy.cpu().numpy() # Formato: [x_min, y_min, x_max, y_max]
            
            if len(boxes) > 0:
                # Pegamos na primeira pessoa detetada
                x_min, y_min, x_max, y_max = boxes[0]
                
                # Calcular o centro da pessoa
                centro_pessoa_x = (x_min + x_max) / 2
                centro_pessoa_y = (y_min + y_max) / 2
                
                # Calcular o erro em relação ao centro da imagem (640x480)
                erro_x = 320 - centro_pessoa_x
                erro_y = 240 - centro_pessoa_y
                
                velocidade_pan = float(erro_x * 0.005)
                velocidade_tilt = float(-erro_y * 0.005) # Sinal negativo para corrigir o eixo
            else:
                # Se não houver ninguém, para os motores
                velocidade_pan = 0.0
                velocidade_tilt = 0.0

            # 4. Enviar velocidades para o Webots
            motor_pan.setVelocity(velocidade_pan)
            motor_tilt.setVelocity(velocidade_tilt)

            # 5. Desenhar o resultado maravilhoso do YOLO
            annotated_frame = results[0].plot()

            # 6. Mostrar no ecrã
            cv2.imshow("Webots + YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n[ERRO CRÍTICO NO PYTHON]: {e}\n")
        break

cv2.destroyAllWindows()