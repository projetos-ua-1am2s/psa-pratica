from controller import Supervisor, Keyboard
import math

class Pedestrian(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        
        # Variáveis de estado persistentes
        self.current_speed = 0.0      # Velocidade atual
        self.target_speed = 0.0       # Velocidade desejada
        self.current_direction = 0.0  # Ângulo atual
        self.walk_step = 0.0          # Progresso da animação
        
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        
        # Coeficientes de animação originais
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

    def run(self):
        time_step = int(self.getBasicTimeStep())
        keyboard = self.getKeyboard()
        keyboard.enable(time_step)

        root_node = self.getSelf()
        trans_field = root_node.getField("translation")
        rot_field = root_node.getField("rotation")
        joints_fields = [root_node.getField(name) for name in self.joint_names]
        
        pos = trans_field.getSFVec3f()
        current_x, current_y = pos[0], pos[1]


        while self.step(time_step) != -1:
            key = keyboard.getKey()
            dt = time_step / 1000.0

            # --- Lógica de Input (Não zera a velocidade se nada for pressionado) ---
            if key == Keyboard.UP:
                self.current_speed = 1.15
            elif key == Keyboard.DOWN:
                self.current_speed = 0

            if key == Keyboard.LEFT:
                self.current_direction += 2.0 * dt # Rotação suave baseada no tempo
            elif key == Keyboard.RIGHT:
                self.current_direction -= 2.0 * dt

            # --- Movimentação ---
            # 1. Calcular a posição POTENCIAL (onde ele quer ir)
            next_x = current_x + self.current_speed * math.cos(self.current_direction) * dt
            next_y = current_y + self.current_speed * math.sin(self.current_direction) * dt
            
            # 2. Verificar a altura do chão na posição atual e na próxima
            z_atual = self.get_ground_z(current_x, current_y)
            z_proximo = self.get_ground_z(next_x, next_y)
            
            # 3. Definir o limite de subida (ex: ele não consegue subir mais que 5cm por frame)
            MAX_STEP_UP = 0.5
            
            # Se a diferença de altura for positiva (subindo) e maior que o limite: PAREDE
            z_diff = abs(z_proximo - z_atual)
            if z_diff > MAX_STEP_UP:
                # Bloqueia o movimento: não atualiza current_x e current_y
                self.current_speed = 0 # Opcional: faz ele parar ao bater na parede
            else:
                # Se for descida ou subida suave, permite o movimento
                current_x = next_x
                current_y = next_y
                ground_z = z_proximo
    
            # --- Animação ---
            # A animação progride de acordo com o deslocamento real
            if self.current_speed != 0:
                self.walk_step += abs(self.current_speed) * dt / self.CYCLE_TO_DISTANCE_RATIO
            
            current_sequence = int(self.walk_step) % self.WALK_SEQUENCES_NUMBER
            ratio = self.walk_step - int(self.walk_step)

            for i in range(self.BODY_PARTS_NUMBER):
                angle = self.angles[i][current_sequence] * (1 - ratio) + \
                        self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                joints_fields[i].setSFFloat(angle)

            # --- Atualização Física no Webots ---
            h_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                       self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
            

            
            # 3. Aplica a translação com o offset da animação
            trans_field.setSFVec3f([current_x, current_y, self.ROOT_HEIGHT + ground_z + h_offset])
            
            rot_field.setSFRotation([0, 0, 1, self.current_direction])
            
    def get_ground_z(self, x, y):
        # 1. Defina os limites da sua escada (ajuste conforme o seu cenário no Webots)
        escada_inicio_x = -1.7
        escada_fim_x = 4.0
        altura_maxima = 2.5
        
        # 2. Calcula a altura do chão (Exemplo de Rampa)
        if x > escada_inicio_x and x < escada_fim_x and y < -6:
            # Calcula a inclinação
            proporcao = (x - escada_inicio_x) / (escada_fim_x - escada_inicio_x)
            ground_z = proporcao * altura_maxima
        elif x >= escada_fim_x:
            ground_z = altura_maxima
        else:
            ground_z = 0.0
        
        return ground_z
    
controller = Pedestrian()
controller.run()