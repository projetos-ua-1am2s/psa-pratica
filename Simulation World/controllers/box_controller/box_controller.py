from controller import Supervisor
import math
import json

class BoxFollower(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        
        # --- Configuração do Trajeto ---
        self.filename = "C:/Users/diana/Desktop/psa-pratica/Simulation World/controllers/record_pedestrian/trajeto_A_editado.json"
        
        self.waypoints = []
        self.target_index = 0
        self.load_path()

        # --- Parâmetros de Movimento ---
        self.current_speed = 4
        self.current_direction = 0.0
        
        # --- Configuração do Offset da Caixa ---
        self.DISTANCE_OFFSET = 0.45  # x centímetros à frente
        self.HEIGHT_OFFSET = 1.3   # Altura aproximada do peito

    def load_path(self):
        try:
            with open(self.filename, 'r') as f:
                self.waypoints = json.load(f)
            print(f"Caixa carregou {len(self.waypoints)} pontos de {self.filename}")
        except:
            print(f"Erro ao carregar {self.filename} no controlador da caixa.")

    def run(self):
        time_step = int(self.getBasicTimeStep())
        root_node = self.getSelf()
        trans_field = root_node.getField("translation")
        rot_field = root_node.getField("rotation")
        
        p = self.waypoints[0]
        self.current_x, self.current_y, self.current_z = p[0], p[1], p[2]
        
        while self.step(time_step) != -1:
            # 1. Sincronização do compasso de espera de 2 segundos
            if self.getTime() < 2:
                # Posiciona a caixa inicialmente à frente do peão parado
                dx = self.waypoints[0][0] - self.current_x
                dy = self.waypoints[0][1] - self.current_y
                init_dir = math.atan2(dy, dx) if len(self.waypoints) > 1 else 0.0
                
                caixa_x = self.current_x + math.cos(init_dir) * self.DISTANCE_OFFSET
                caixa_y = self.current_y + math.sin(init_dir) * self.DISTANCE_OFFSET
                caixa_z = self.current_z + self.HEIGHT_OFFSET
                
                trans_field.setSFVec3f([caixa_x, caixa_y, caixa_z])
                rot_field.setSFRotation([0, 0, 1, init_dir])
                continue
                
            if not self.waypoints:
                continue

            dt = time_step / 1000.0
            target = self.waypoints[self.target_index]
            
            # 2. Calcular vetor para o alvo (Cálculo idêntico ao do peão)
            dx = target[0] - self.current_x
            dy = target[1] - self.current_y
            dist_ao_alvo = math.sqrt(dx**2 + dy**2)

            if dist_ao_alvo < 0.1: 
                self.target_index = (self.target_index + 1) % len(self.waypoints)
            else:
                self.current_direction = math.atan2(dy, dx)
                
                # Avançar a posição base da caixa (que espelha o centro do peão)
                deslocamento = self.current_speed * dt
                self.current_x += math.cos(self.current_direction) * deslocamento
                self.current_y += math.sin(self.current_direction) * deslocamento
                self.current_z += (target[2] - self.current_z) * 0.1 

            # 3. APLICAR OFFSETS TRIGONOMÉTRICOS
            # Empurra a caixa para a frente com base no ângulo atual de caminhada
            caixa_x = self.current_x + math.cos(self.current_direction) * self.DISTANCE_OFFSET
            caixa_y = self.current_y + math.sin(self.current_direction) * self.DISTANCE_OFFSET
            caixa_z = self.current_z + self.HEIGHT_OFFSET

            # 4. Atualização Física da Caixa no Mundo
            trans_field.setSFVec3f([caixa_x, caixa_y, caixa_z])
            rot_field.setSFRotation([0, 0, 1, self.current_direction])

controller = BoxFollower()
controller.run()