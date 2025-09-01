from pose_estimator_icam import PoseEstimator
import time
import numpy as np
import math
import psutil
import os

import time
from arduino_test import ArduinoAbsMouse
# from receive_udp import MAMECursor

from pynput import keyboard
from screeninfo import get_monitors
from light_gun_input import LightGunInput

if __name__ == '__main__':

    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    p = psutil.Process(os.getpid())
    p.cpu_affinity([2, 3])  # Pin to cores 0 and 1

    mode = 'none'
    offscreen_reload = False

    def on_press(key):
        global mode, offscreen_reload
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
            if key.char == 'm':
                mode = 'mame'
            if key.char == 'w':
                mode = 'windows'
            if key.char == 'n':
                mode = 'none'
            if key.char == 'e':
                mode = 'exit'
            if key.char == 'o':
                if offscreen_reload:
                    offscreen_reload = False
                else:
                    offscreen_reload = True
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    def on_release(key):
        print('{0} released'.format(
            key))

    # ...or, in a non-blocking fashion
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    def calculate_pos_and_dir_from_pose(pose_matrix):
        tracker_position = np.copy(pose_matrix[:3, -1])
        tracker_direction = -np.copy(pose_matrix[:3, 2])

        return (tracker_position, tracker_direction)

    # Pose estimator
    pose_estimator = PoseEstimator()
    light_gun_input = LightGunInput()
    monitor = get_monitors()[0]

    running = True
    dt = 0
    mouse = ArduinoAbsMouse()

    # Load calibration
    transform_to_aim = np.load("tracker_to_gun.npy")
    x_opt = np.load("rectangle_params.npy")
        
    C_opt = x_opt[0:3]
    u_opt = x_opt[3:6]
    v_opt = x_opt[6:9]
    l_u_opt = x_opt[9] + 0.0625 * x_opt[9]
    l_v_opt = x_opt[10] + 0.11111111111111111 * x_opt[10]

    N_opt = np.cross(u_opt, v_opt)
    N_opt /= np.linalg.norm(N_opt)

    mouse_buttom = 0
    frame = 0
    st = time.time()

    save_number = 0

    button_trigger, button_left, button_right = False, False, False
    last_time = 0.0
    while mode != 'exit':

        pose_matrix, valid_pose = pose_estimator.get_pose()
        if light_gun_input.get_input() is not None:
            button_trigger, button_left, button_right = light_gun_input.get_input()

        if pose_matrix is None or np.isnan(np.sum(pose_matrix)):
            time.sleep(0.002)
            continue

        pose_matrix = np.copy(pose_matrix)

        pose_matrix = transform_to_aim @ pose_matrix

        tracker_position, tracker_direction = calculate_pos_and_dir_from_pose(pose_matrix)

        denom = np.dot(tracker_direction, N_opt)

        if denom == 0.0:
            time.sleep(0.002)
            continue
        
        numerator = np.dot(C_opt - tracker_position, N_opt)
        t = numerator / denom
        P_intersect = tracker_position + t * tracker_direction
        V = P_intersect - C_opt
        a = np.dot(V, u_opt)
        b = np.dot(V, v_opt)

        u_coord = (a + l_u_opt) / (2 * l_u_opt)
        # v_coord = (l_v_opt - b) / (2 * l_v_opt)
        v_coord = (b + l_v_opt) / (2 * l_v_opt)

        if math.isnan(u_coord) or math.isnan(v_coord):
            time.sleep(0.002)
            continue

        mouse_buttom = 0
        button_trigger, button_left, button_right
        if offscreen_reload and (u_coord < 0 or u_coord > 1 or v_coord < 0 or v_coord > 1): 
            # if button_right:
            #     mouse_buttom += 4
            if button_trigger:
                mouse_buttom += 2
            # if button_left:
            #     mouse_buttom += 3
        else:
            # if not offscreen_reload:
            #     if button_left:
            #         mouse_buttom += 1
            if button_right:
                mouse_buttom += 4
            if button_trigger:
                mouse_buttom += 1
            if button_left:
                mouse_buttom += 2


        if mode in ["windows", "mame"]:
            mouse.move_mouse(u_coord, v_coord, mouse_buttom, mode == "mame")

        time.sleep(0.001)
    mouse.close()
    listener.stop()
    pose_estimator.cleanup()