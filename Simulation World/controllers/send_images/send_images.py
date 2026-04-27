from controller import Robot, Camera, Motor
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
import numpy as np
import cv2
import json

# Inicialização
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- CONFIGURAR DISPOSITIVOS ---
cam = robot.getDevice("camera")
cam.enable(timestep)
width, height = cam.getWidth(), cam.getHeight()

pan_motor = robot.getDevice("pan_motor")
tilt_motor = robot.getDevice("tilt_motor")

# Variáveis para armazenar a posição atual (acumuladores)
current_pan = 0.0
current_tilt = 0.0
# Sensibilidade do movimento (ajusta conforme necessário)
K_SERVO = 0.05 

# --- MQTT CALLBACK ---
def on_message(client, userdata, msg):
    global current_pan, current_tilt
    try:
        # Recebe o JSON do Vision Brain: {"magnitude": 0.2, "angle": 45.0}
        data = json.loads(msg.payload)
        mag = data.get("magnitude", 0)
        ang_deg = data.get("angle", 0)
        
        # Converter polar (mag, ang) para coordenadas cartesianas (dx, dy)
        ang_rad = np.radians(ang_deg)
        dx = mag * np.cos(ang_rad)
        dy = mag * np.sin(ang_rad)

        # Atualizar a posição incrementalmente
        # dx positivo -> pan para a direita (negativo no Webots costuma ser direita)
        # dy positivo -> tilt para baixo
        current_pan -= dx * K_SERVO
        current_tilt += dy * K_SERVO

        # Aplicar limites simples para não forçar o motor do Webots
        current_pan = max(min(current_pan, 1.5), -1.5)
        current_tilt = max(min(current_tilt, 1.0), -1.0)

        pan_motor.setPosition(current_pan)
        tilt_motor.setPosition(current_tilt)
        
    except Exception as e:
        print(f"Erro no processamento do vetor: {e}")

# --- MQTT SETUP ---
client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("Movement") # Ouvindo diretamente o Cérebro
client.loop_start()

print("Webots Bridge ativa: Ouvindo 'Movement' e enviando para 'Camera'")

while robot.step(timestep) != -1:
    # 1. Envio de Imagem (Mantém-se igual)
    img_data = cam.getImage()
    if img_data:
        frame = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 4))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if _:
            client.publish("Camera", buffer.tobytes())

client.loop_stop()