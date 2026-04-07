from controller import Robot
import numpy as np
import cv2

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

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

width = camera.getWidth()
height = camera.getHeight()

frame_counter = 0
velocidade_pan = 0.0
velocidade_tilt = 0.0
caixa_atual = None

print("A iniciar a câmara e o rastreio...")

while robot.step(timestep) != -1:
    try:
        image = camera.getImage()

        if image:
            # 1. Converter a imagem do Webots
            img_bgra = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            img = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

            # Forçar um tamanho seguro para o HOG não "rebentar" (640x480)
            img = cv2.resize(img, (640, 480))
            
            # Converter para escala de cinzentos apenas para a deteção (é mais rápido e seguro)
            img_cinzenta = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            frame_counter += 1

            # 2. Deteção apenas a cada x frames
            if frame_counter % 1 == 0:
                # Passamos a img_cinzenta para o detetor!
                boxes, weights = hog.detectMultiScale(img_cinzenta, winStride=(8, 8), scale=1.05)
                
                if len(boxes) > 0:
                    caixa_atual = boxes[0] 
                    
                    x, y, w, h = int(caixa_atual[0]), int(caixa_atual[1]), int(caixa_atual[2]), int(caixa_atual[3])
                    
                    centro_pessoa_x = x + (w / 2)
                    
                    # Como redimensionámos para 640x480, o centro da imagem é sempre 320
                    centro_imagem_x = 640 / 2
                    
                    erro = centro_imagem_x - centro_pessoa_x
                    velocidade_pan = float(erro * 0.005)
                    
                    centro_pessoa_y = y + (h / 2)
                    centro_imagem_y = 480 / 2  # A altura da tua câmara a dividir por 2
                    erro_y = centro_imagem_y - centro_pessoa_y
                    velocidade_tilt = float(-erro_y * 0.005) 
                else:
                    caixa_atual = None
                    velocidade_pan = 0.0
                    velocidade_tilt = 0.0

            # 3. Atualizar o motor
            motor_pan.setVelocity(velocidade_pan)
            motor_tilt.setVelocity(velocidade_tilt)

            # 4. Desenhar a caixa verde na imagem a cores original
            if caixa_atual is not None:
                x, y, w, h = int(caixa_atual[0]), int(caixa_atual[1]), int(caixa_atual[2]), int(caixa_atual[3])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 5. Mostrar a imagem
            cv2.imshow("Camera Webots", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n[ERRO CRÍTICO NO PYTHON]: {e}\n")
        break

cv2.destroyAllWindows()