import numpy as np
import cv2
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

# 1. Função que é chamada sempre que chega uma imagem
def on_message(client, userdata, msg, *args, **kwargs):
    try:
        # Transformar os bytes recebidos num array de números (NumPy)
        nparr = np.frombuffer(msg.payload, np.uint8)
        
        # DESCOMPRIMIR o JPEG (O OpenCV deteta automaticamente que é 720p)
        img_recebida = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_recebida is not None:
            # Mostrar a imagem numa janela
            cv2.imshow("Stream do Webots (720p)", img_recebida)
            
            # ESPERAR 1ms para a janela processar a imagem (Obrigatório no OpenCV)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
        else:
            print("Erro: Não foi possível decodificar a imagem.")
            
    except Exception as e:
        print(f"Erro no processamento: {e}")

# 2. Configuração do Cliente MQTT (Versão 2)
client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.on_message = on_message

# 3. Conectar ao Broker
client.connect("localhost", 1883)

# 4. Subscrever ao tópico
client.subscribe("webots/camera")

print("Aguardando imagens do Webots... (Pressione 'q' na janela da imagem para sair)")

# 5. Manter o script a correr
client.loop_forever()