from controller import Robot, Camera
import paho.mqtt.client as mqtt

# Envia a imagem através do mqtt

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Configurar Câmera
cam = robot.getDevice("tracker_camera")
cam.enable(timestep)

# MQTT Setup
client = mqtt.Client()
client.connect("localhost", 1883)

while robot.step(timestep) != -1:
    # 1. Captura os bytes da imagem (BGRA)
    img_data = cam.getImage()
    
    # 2. Envia os bytes brutos diretamente para o tópico
    if img_data:
        client.publish("webots/camera/raw", img_data)
            

    # print("Imagem enviada")