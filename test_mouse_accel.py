#%%
import time
from pynput.mouse import Controller
import numpy as np
from arduino_test import ArduinoMouse

mouse_ctrl = Controller()
mouse = ArduinoMouse(2560, 1440, 2, 1)

def reset_x_axis():
    delta_x_pixels = -mouse_ctrl.position[0]

    while np.abs(delta_x_pixels) > 10:
        moviment_x = int(np.clip(delta_x_pixels, -25, 25))
        mouse.move_mouse(moviment_x, 0)
        time.sleep(0.1)
        delta_x_pixels = -mouse_ctrl.position[0]


def reset_y_axis():
    delta_y_pixels = -mouse_ctrl.position[1]

    while np.abs(delta_y_pixels) > 10:
        moviment_y = int(np.clip(delta_y_pixels, -25, 25))
        mouse.move_mouse(0, moviment_y)
        time.sleep(0.1)
        delta_y_pixels = -mouse_ctrl.position[1]

mouse_sensi_y = []
for moviment in [25, 50, 75, 100, 125]:
    reset_y_axis()
    pos_before = mouse_ctrl.position[1]
    mouse.move_mouse(0, moviment)
    time.sleep(0.1)
    pos_after = mouse_ctrl.position[1]
    mouse_sensi_y.append((pos_after - pos_before) / moviment)

print(f'Mean sensitivity on Y axis: {np.mean(mouse_sensi_y)} - Variance: {np.std(mouse_sensi_y)} (a high variance indicates acceleration)')

mouse_sensi_x = []
for moviment in [25, 50, 75, 100, 125]:
    reset_x_axis()
    pos_before = mouse_ctrl.position[0]
    mouse.move_mouse(moviment, 0)
    time.sleep(0.1)
    pos_after = mouse_ctrl.position[0]
    mouse_sensi_x.append((pos_after - pos_before) / moviment)

print(f'Mean sensitivity on X axis: {np.mean(mouse_sensi_x)} - Variance: {np.std(mouse_sensi_x)} (a high variance indicates acceleration)')