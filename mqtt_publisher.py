import json
import time
import paho.mqtt.client as mqtt
from database import Database

class Servidor:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.db = Database()

        # Conectar antes de iniciar o loop
        self.conectar_mqtt()

    def conectar_mqtt(self):
        """Tenta conectar ao broker MQTT com tentativas automaticas."""
        for _ in range(5):  # Tenta conectar ate 5 vezes
            try:
                self.client.connect("localhost", 1883, 60)  # Altere para o endereco do seu broker
                self.client.loop_start()
                time.sleep(2)  # Aguarda um tempo para estabilizar a conexao
                print("Conexao MQTT estabelecida com sucesso")
                return
            except Exception as e:
                print(f"Falha ao conectar ao MQTT. Tentando novamente... Erro: {e}")
                time.sleep(5)  # Aguarda antes de tentar novamente

        print("Nao foi possivel conectar ao MQTT apos varias tentativas.")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Conectado ao MQTT com sucesso")
            self.client.subscribe("project/inference")
        else:
            print(f"Falha ao conectar ao MQTT. Codigo de retorno: {rc}")

    def on_disconnect(self, client, userdata, rc):
        print("Desconectado do MQTT. Tentando reconectar...")
        self.conectar_mqtt()

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            print(f"Recebido via MQTT: {payload}")
            self.db.insert_result(payload)
        except Exception as e:
            print(f"Erro ao processar mensagem MQTT: {e}")

    def publish_result(self, num_objects, speed, infractions, distance, timestamp, placa, image_path, type_infraction , tracker_id ):
        """Publica os resultados apenas se o cliente MQTT estiver conectado."""
        if not self.client.is_connected():
            print("Erro: Cliente MQTT nao esta conectado. Tentando reconectar...")
            self.conectar_mqtt()  # Tenta reconectar antes de publicar
            return

        try:
            data = {
                "type infraction": type_infraction,
                "timestamp": timestamp,
                "vehicle id": tracker_id,
                "distance": distance,
                "speed": speed,
                "context and license plate": placa,
                "image": image_path,
                "num_objects": num_objects,
                "infractions": infractions
            }
            self.client.publish("project/inference", json.dumps(data))
            print(f"Publicado MQTT: {data}")
        except Exception as e:
            print(f"Erro ao publicar MQTT: {e}")

