from controller import Supervisor
import math
import json

# Senta-se no sofá

class PedestrianFollower(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        
        # --- Estados ---
        self.STATE_WALKING = 0
        self.STATE_SITTING = 1
        self.STATE_IDLE = 2
        self.state = self.STATE_WALKING
        
        # --- Configuração do Trajeto ---
        self.filename = "C:/Users/diana/Desktop/psa-pratica/Simulation World/controllers/record_pedestrian/trajeto_D.json"
        self.waypoints = []
        self.target_index = 0
        self.load_path()

        # --- Variáveis de Controle ---
        self.current_speed = 1.5
        self.current_direction = 0.0
        self.walk_step = 0.0
        self.wait_timer = 0.0
        self.SITTING_DURATION = 10.0 # Tempo que fica sentado em segundos
        
        # Índice do waypoint onde ele deve sentar (ex: o último do JSON)
        self.SOFA_WAYPOINT_INDEX = len(self.waypoints) - 1 

        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        
        # Ângulos para a posição sentada (em radianos)
        # Ordem segue self.joint_names
        self.SITTING_ANGLES = [
            0.0, -0.2, 0.0,       # Braço Esq
            0.0, -0.2, 0.0,       # Braço Dir
            -1.4, 1.4, 0.1,       # Perna Esq (Coxa, Joelho, Pé)
            -1.4, 1.4, 0.1,       # Perna Dir
            0.0                   # Cabeça
        ]

        self.height_offsets = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
        self.angles = [ # ... (seus ângulos de caminhada mantêm-se iguais)
            [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74], [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
            [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0], [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
            [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38], [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
            [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4], [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
            [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07], [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
            [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49], [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
            [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09]
        ]

    def load_path(self):
        try:
            with open(self.filename, 'r') as f:
                self.waypoints = json.load(f)
        except:
            print("Erro ao carregar ficheiro.")

    def run(self):
        time_step = int(self.getBasicTimeStep())
        root_node = self.getSelf()
        trans_field = root_node.getField("translation")
        rot_field = root_node.getField("rotation")
        joints_fields = [root_node.getField(name) for name in self.joint_names]
        
        p = self.waypoints[0]
        self.current_x, self.current_y, self.current_z = p[0], p[1], p[2]
        
        while self.step(time_step) != -1:
            dt = time_step / 1000.0

            if self.state == self.STATE_WALKING:
                target = self.waypoints[self.target_index]
                dx = target[0] - self.current_x
                dy = target[1] - self.current_y
                dist_ao_alvo = math.sqrt(dx**2 + dy**2)

                if dist_ao_alvo < 0.1:
                    # Se chegou no waypoint do sofá
                    if self.target_index == self.SOFA_WAYPOINT_INDEX:
                        self.state = self.STATE_SITTING
                        self.wait_timer = self.getTime()
                    else:
                        self.target_index = (self.target_index + 1) % len(self.waypoints)
                else:
                    # Lógica de Caminhada
                    self.current_direction = math.atan2(dy, dx)
                    deslocamento = self.current_speed * dt
                    self.current_x += math.cos(self.current_direction) * deslocamento
                    self.current_y += math.sin(self.current_direction) * deslocamento
                    self.current_z += (target[2] - self.current_z) * 0.1 

                    # Animação de Caminhada
                    self.walk_step += self.current_speed * dt / self.CYCLE_TO_DISTANCE_RATIO
                    curr_seq = int(self.walk_step) % self.WALK_SEQUENCES_NUMBER
                    ratio = self.walk_step - int(self.walk_step)
                    
                    for i in range(self.BODY_PARTS_NUMBER):
                        angle = self.angles[i][curr_seq] * (1 - ratio) + \
                                self.angles[i][(curr_seq + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                        joints_fields[i].setSFFloat(angle)

                    h_offset = self.height_offsets[curr_seq] * (1 - ratio) + \
                               self.height_offsets[(curr_seq + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                    
                    trans_field.setSFVec3f([self.current_x, self.current_y, self.ROOT_HEIGHT + self.current_z + h_offset])
                    rot_field.setSFRotation([0, 0, 1, self.current_direction])

            elif self.state == self.STATE_SITTING:
                # Aplica ângulos de sentado e reduz a altura do ROOT
                for i in range(self.BODY_PARTS_NUMBER):
                    # Transição simples (pode usar interpolação para ser mais suave)
                    joints_fields[i].setSFFloat(self.SITTING_ANGLES[i])
                
                # Reajusta a altura Y para o boneco "entrar" no sofá
                trans_field.setSFVec3f([self.current_x, self.current_y, self.current_z + 0.85])
                
                # Ajusta a direção
                direcao_sofa = 3.14 
                rot_field.setSFRotation([0, 0, 1, direcao_sofa])
                
                # Verifica se o tempo de descanso acabou
                if self.getTime() - self.wait_timer > self.SITTING_DURATION:
                    # Inverter a lista de waypoints para ele refazer o caminho de volta
                    self.waypoints.reverse() 
                    self.target_index = 1 # Começa a ir para o próximo ponto da lista invertida
                    self.state = self.STATE_WALKING
                    
controller = PedestrianFollower()
controller.run()