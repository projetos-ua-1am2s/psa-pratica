import paho.mqtt.client as mqtt

# Esta função é chamada sempre que uma nova mensagem chega
def ao_receber_mensagem(client, userdata, message):
    print(f"Recebi do Webots: {message.payload.decode()} no tópico {message.topic}")

# 1. Criamos o cliente
cliente = mqtt.Client()

# 2. Dizemos ao cliente qual função usar quando chegar mensagem
cliente.on_message = ao_receber_mensagem

# 3. Conectamos ao Broker (localhost porque está no seu PC)
cliente.connect("localhost", 1883)

# 4. Escolhemos o que queremos ouvir
cliente.subscribe("webots/sensor")

print("Aguardando mensagens... (Pressione Ctrl+C para sair)")

# 5. Mantemos o script rodando para sempre
cliente.loop_forever()