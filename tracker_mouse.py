from pose_estimator_icam import PoseEstimator
import time
import numpy as np
import math

import time
import json
from arduino_test import ArduinoMouse, WindowsCursor
from receive_udp import MAMECursor

from pynput import keyboard
from screeninfo import get_monitors

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

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def project_ray_to_plane(plane_point, plane_normal, point, dir):
    epsilon=1e-6

    ndotu = plane_normal.dot(dir)

    if abs(ndotu) < epsilon:
        return np.array([1, 1, 1])

    # Project point in the screen/plane
    w = point - plane_point
    si = -plane_normal.dot(w) / ndotu
    p_screen = point + (si*dir)
    return p_screen


def point_to_uv_coordinates(U_axis, V_axis, axis_center, P):

    # Calculate the P_vector
    P_vector = P - axis_center

    # Calculate the dot products
    u_dot = np.dot(P_vector, U_axis)
    v_dot = np.dot(P_vector, V_axis)

    # Calculate the squared magnitudes of U_axis and V_axis
    u_magnitude_squared = np.dot(U_axis, U_axis)
    v_magnitude_squared = np.dot(V_axis, V_axis)

    # Calculate u and v coordinates
    u = u_dot / u_magnitude_squared
    v = v_dot / v_magnitude_squared

    return u, v


def point_to_uv_coordinates_multi(calib, P):

    u_coord1, v_coord1 = point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  P)
    u_coord2, v_coord2 = point_to_uv_coordinates(calib['bottom_right'] - calib['bottom_left'], calib['top_left'] - calib['bottom_left'], calib['bottom_left'], P)
    u_coord3, v_coord3 = point_to_uv_coordinates(calib['bottom_left'] - calib['bottom_right'], calib['top_right'] - calib['bottom_right'], calib['bottom_right'], P)
    u_coord4, v_coord4 = point_to_uv_coordinates(calib['top_left'] - calib['top_right'], calib['bottom_right'] - calib['top_right'], calib['top_right'], P)

    u_coord4 = 1 - u_coord4
    v_coord2 = 1 - v_coord2
    u_coord3 = 1 - u_coord3
    v_coord3 = 1 - v_coord3

    u_coord = (u_coord1 * (1-v_coord1) + u_coord2 * v_coord1 + u_coord3 * v_coord1 + u_coord4* (1-v_coord1)) / 2
    v_coord = (v_coord1 * (1-u_coord1) + v_coord2 * (1-u_coord1) + v_coord3 * u_coord1 + v_coord4 * u_coord1) / 2

    return (u_coord, v_coord)

def calculate_pos_and_dir_from_pose(pose_matrix):
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = np.copy(pose_matrix[2, :3])
    tracker_direction[2] = -tracker_direction[2]

    return (tracker_position, tracker_direction)

# Pose estimator
pose_estimator = PoseEstimator()
monitor = get_monitors()[0]

running = True
dt = 0
mouse = ArduinoMouse(1.0)
windows_cursor = WindowsCursor(monitor.width, monitor.height)
mame_cursor = MAMECursor(monitor.width, monitor.height)

# Load calibration
with open("calibration_new_1.json", 'r') as f:
    calib = json.load(f)

# Convert calib to numpy arrays
for key, val in calib.items():
    calib[key] = np.asfarray(val)

device_to_gun_1 = np.load('device_to_gun_1.npy')

mouse_buttom = 0
frame = 0
st = time.time()

save_number = 0
while mode != 'exit':

    pose_matrix = pose_estimator.get_pose()

    if pose_matrix is None or np.isnan(np.sum(pose_matrix)):
        time.sleep(0.002)
        continue

    pose_matrix = np.copy(pose_matrix)

    pose_matrix = pose_matrix @ device_to_gun_1

    tracker_position, tracker_direction = calculate_pos_and_dir_from_pose(pose_matrix)

    # controller_input = v.devices["tracker_1"].get_controller_inputs()

    p_screen = project_ray_to_plane(calib['top_left'], calib['plane_normal'], tracker_position, tracker_direction)

    # Project point into the u/v screen coordinate system
    u_coord, v_coord = point_to_uv_coordinates_multi(calib, p_screen)

    if math.isnan(u_coord) or math.isnan(v_coord):
        time.sleep(0.002)
        continue

    # mouse_buttom = 0
    # if offscreen_reload and (u_coord < 0 or u_coord > 1 or v_coord < 0 or v_coord > 1): 
    #     if controller_input['trackpad_touched']:
    #         mouse_buttom += 4
    #     # if controller_input['trigger'] > 0.5:
    #     #     mouse_buttom += 2
    #     if controller_input['menu_button']:
    #         mouse_buttom += 3
    # else:
    #     # if not offscreen_reload:
    #     #     if controller_input['menu_button']:
    #     #         mouse_buttom += 1
    #     if controller_input['trackpad_touched']:
    #         mouse_buttom += 2
    #     # if controller_input['trigger'] > 0.5:
    #     #     mouse_buttom += 1
    #     if controller_input['menu_button']:
    #         mouse_buttom += 1


    if mode == 'windows':
        pos_x, pos_y = windows_cursor.get_position()
        mouse.update_position(pos_x, pos_y)
        target_x, target_y = windows_cursor.process_target(u_coord, v_coord)
        mouse.move_mouse_absolute(target_x, target_y, mouse_buttom)
    elif mode == 'mame':
        if mame_cursor.has_new_pos():
            pos_x, pos_y = mame_cursor.get_position()
            mouse.update_position(pos_x, pos_y)
        target_x, target_y = mame_cursor.process_target(u_coord, v_coord)
        mouse.move_mouse_absolute(target_x, target_y, mouse_buttom)

mouse.close()
mame_cursor.close()
listener.stop()
pose_estimator.cleanup()