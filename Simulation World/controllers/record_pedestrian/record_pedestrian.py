from controller import Supervisor, Keyboard
import math
import json
import os

class Pedestrian(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        
        # --- Configurações de Gravação ---
        self.recording = False
        self.waypoints = []
        # Nome do arquivo que será salvo
        self.filename = "trajeto_D.json" 
        
        self.current_speed = 0.0
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

    def get_ground_z(self, x, y):
        escada_inicio_x = -1.7
        escada_fim_x = 4.0
        altura_maxima = 2.5
        
        if x > escada_inicio_x and x < escada_fim_x and y < -6:
            proporcao = (x - escada_inicio_x) / (escada_fim_x - escada_inicio_x)
            ground_z = proporcao * altura_maxima
        elif x >= escada_fim_x:
            ground_z = altura_maxima
        else:
            ground_z = 0.0
        return ground_z

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
        ground_z = pos[2] - self.ROOT_HEIGHT

        print("--- CONTROLADOR PRONTO ---")
        print("UP/DOWN: Velocidade | LEFT/RIGHT: Direção")
        print("R: Iniciar Gravação | S: Salvar Trajeto")

        while self.step(time_step) != -1:
            key = keyboard.getKey()
            dt = time_step / 1000.0

            # --- Inputs do Teclado ---
            if key == Keyboard.UP:
                self.current_speed = 1.15
            elif key == Keyboard.DOWN:
                self.current_speed = 0

            if key == Keyboard.LEFT:
                self.current_direction += 2.0 * dt
            elif key == Keyboard.RIGHT:
                self.current_direction -= 2.0 * dt

            # --- Lógica de Gravação ---
            if key == ord('R'):
                if not self.recording:
                    self.recording = True
                    self.waypoints = []
                    print(f"Gravação iniciada para: {self.filename}")

            if key == ord('S'):
                if self.recording:
                    with open(self.filename, "w") as f:
                        json.dump(self.waypoints, f)
                    self.recording = False
                    print(f"Trajeto salvo! Total de pontos: {len(self.waypoints)}")

            # --- Movimentação e Física ---
            next_x = current_x + self.current_speed * math.cos(self.current_direction) * dt
            next_y = current_y + self.current_speed * math.sin(self.current_direction) * dt
            
            z_proximo = self.get_ground_z(next_x, next_y)
            MAX_STEP_UP = 0.5
            
            if abs(z_proximo - ground_z) <= MAX_STEP_UP:
                current_x, current_y = next_x, next_y
                ground_z = z_proximo

            # --- Armazenar Ponto (Apenas se estiver gravando e se moveu) ---
            if self.recording and self.current_speed > 0:
                # Salva X, Y, e o Z calculado da escada
                pos_to_save = [current_x, current_y, ground_z]
                # Evita salvar pontos duplicados muito próximos (distância > 0.1m)
                if not self.waypoints or math.dist(pos_to_save[:2], self.waypoints[-1][:2]) > 0.1:
                    self.waypoints.append(pos_to_save)

            # --- Animação ---
            if self.current_speed != 0:
                self.walk_step += abs(self.current_speed) * dt / self.CYCLE_TO_DISTANCE_RATIO
            
            current_sequence = int(self.walk_step) % self.WALK_SEQUENCES_NUMBER
            ratio = self.walk_step - int(self.walk_step)

            for i in range(self.BODY_PARTS_NUMBER):
                angle = self.angles[i][current_sequence] * (1 - ratio) + \
                        self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                joints_fields[i].setSFFloat(angle)

            # --- Atualização no Webots ---
            h_offset = self.height_offsets[current_sequence] * (1 - ratio) + \
                       self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
            
            trans_field.setSFVec3f([current_x, current_y, self.ROOT_HEIGHT + ground_z + h_offset])
            rot_field.setSFRotation([0, 0, 1, self.current_direction])

controller = Pedestrian()
controller.run()