from controller import Supervisor
import math
import json
import os

# Sobe escadas e caminha no patamar superior

class PedestrianFollower(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        
        # --- Configuração do Trajeto ---
        self.filename = "C:/Users/diana/Desktop/psa-pratica/Simulation World/controllers/record_pedestrian/trajeto_A_editado.json"
        
        self.waypoints = []
        self.target_index = 0
        self.load_path()

        self.current_speed = 4
        self.ratio_anim = 0.5
        self.current_direction = 0.0
        self.walk_step = 0.0
        
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        
        self.height_offsets = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
        self.angles = [
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
            print(f"Pedestre carregou {len(self.waypoints)} pontos de {self.filename}")
        except:
            print(f"Erro ao carregar {self.filename}. Verifique se o ficheiro existe.")

    def run(self):
        time_step = int(self.getBasicTimeStep())
        root_node = self.getSelf()
        trans_field = root_node.getField("translation")
        rot_field = root_node.getField("rotation")
        joints_fields = [root_node.getField(name) for name in self.joint_names]
        p = self.waypoints[0]
        self.current_x, self.current_y, self.current_z = p[0], p[1], p[2]
        
        while self.step(time_step) != -1:
            if self.getTime() < 2:
                continue
            if not self.waypoints:
                continue

            dt = time_step / 1000.0
            
            target = self.waypoints[self.target_index]
            
            # 3. Calcular vetor para o alvo
            dx = target[0] - self.current_x
            dy = target[1] - self.current_y
            dist_ao_alvo = math.sqrt(dx**2 + dy**2)

            if dist_ao_alvo < 0.1: # Chegou muito perto do waypoint
                self.target_index = (self.target_index + 1) % len(self.waypoints)
            else:
                # 4. MOVIMENTO REAL: Deslocar apenas o necessário baseado na velocidade e tempo
                self.current_direction = math.atan2(dy, dx)
                
                # Avançar a posição interna
                deslocamento = self.current_speed * dt
                self.current_x += math.cos(self.current_direction) * deslocamento
                self.current_y += math.sin(self.current_direction) * deslocamento
                
                # Interpolar o Z suavemente entre a posição atual e o alvo
                # (Isso evita que ele dê "saltos" de altura na escada)
                self.current_z += (target[2] - self.current_z) * 0.1 

            # 5. Animação das pernas (agora sincronizada com o deslocamento real)
            self.walk_step += self.current_speed * self.ratio_anim * dt / self.CYCLE_TO_DISTANCE_RATIO
            current_sequence = int(self.walk_step) % self.WALK_SEQUENCES_NUMBER
            ratio = self.walk_step - int(self.walk_step)
            
            for i in range(self.BODY_PARTS_NUMBER):
                angle = self.angles[i][current_sequence] * (1 - ratio) + \
                        self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                joints_fields[i].setSFFloat(angle)
                
            # 0: leftArmAngle, 1: leftLowerArmAngle, 3: rightArmAngle, 4: rightLowerArmAngle
            joints_fields[0].setSFFloat(0.2)   # Ombro esquerdo para a frente
            joints_fields[1].setSFFloat(-1.4)  # Dobrar cotovelo esquerdo a ~90º
            joints_fields[3].setSFFloat(0.2)   # Ombro direito para a frente
            joints_fields[4].setSFFloat(-1.4)  # Dobrar cotovelo direito a ~90º
            
            # 5. Atualização Física (Seguir o rastro do ficheiro)
            h_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                       self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
            
            trans_field.setSFVec3f([self.current_x, self.current_y, self.ROOT_HEIGHT + self.current_z + h_offset])
            rot_field.setSFRotation([0, 0, 1, self.current_direction])

controller = PedestrianFollower()
controller.run()