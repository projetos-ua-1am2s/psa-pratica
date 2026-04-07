from controller import Robot
import random
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# parâmetros de movimento
speed = 0.02          # metros por passo
change_dir_steps = 120
step_count = 0

# direção inicial
angle = random.uniform(0, 2 * math.pi)

# campo de rotação do robô
rotation_field = robot.getField("rotation")
translation_field = robot.getField("translation")

while robot.step(timestep) != -1:
    pos = translation_field.getSFVec3f()

    step_count += 1
    if step_count > change_dir_steps:
        angle = random.uniform(0, 2 * math.pi)
        step_count = 0

    dx = speed * math.cos(angle)
    dz = speed * math.sin(angle)

    # mover o robô
    translation_field.setSFVec3f([
        pos[0] + dx,
        pos[1],
        pos[2] + dz
    ])

    # rodar para a direção do movimento
    rotation_field.setSFRotation([0, 1, 0, -angle + math.pi / 2])
