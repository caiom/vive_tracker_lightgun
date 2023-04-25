import triad_openvr
import time
import numpy as np
import math

import time
import json
from arduino_test import ArduinoMouse2, WindowsCursor
from receive_udp import MAMECursor


from pynput import keyboard

mode = 'none'
c_top_left = False
c_top_right = False
c_bottom_left = False
c_bottom_right = False
save_info = False
use_device_rot = False

def on_press(key):
    global mode, c_top_left, c_top_right, c_bottom_left, c_bottom_right, save_info, use_device_rot
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        if key.char == 'a':
            c_bottom_left = True
        if key.char == 'z':
            save_info = True
        if key.char == 's':
            c_top_left = True
        if key.char == 'd':
            c_top_right = True
        if key.char == 'f':
            c_bottom_right = True
        if key.char == 'x':
            if use_device_rot:
                use_device_rot = False
            else:
                use_device_rot = True
            use_device_rot = True
        if key.char == 'm':
            mode = 'mame'
        if key.char == 'w':
            mode = 'windows'
        if key.char == 'n':
            mode = 'none'
        if key.char == 'e':
            mode = 'exit'
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

def unit_vector_to_euler_angles(vector, up=np.array([0, 1, 0])):
    vx, vy, vz = vector
    
    # Calculate yaw angle (θ)
    yaw = np.arctan2(vy, vx)
    
    pitch = np.arctan2(np.sqrt(vx**2 + vy**2), vz)
    
    # Calculate the right vector by taking the cross product of the direction vector and the up vector
    right = np.cross(vector, up)
    
    # Normalize the right vector
    right /= np.linalg.norm(right)

    # Calculate the new up vector by taking the cross product of the right vector and the direction vector
    new_up = np.cross(right, vector)
    
    # Calculate the roll angle (ψ) using the dot product of the new up vector and the original up vector
    roll = np.arccos(np.dot(new_up, up))

    # Convert the angles from radians to degrees, if necessary
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    roll_deg = np.degrees(roll)

    return pitch_deg, yaw_deg, roll_deg

def project_ray_to_plane(plane_point, plane_normal, point, dir):
    epsilon=1e-6

    ndotu = plane_normal.dot(dir)

    if abs(ndotu) < epsilon:
        return None

    # Project point in the screen/plane
    w = tracker_position - plane_point
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

    v_coord = v_coord / (1/0.8888888888888889) + 0.0555555555555556
    u_coord = u_coord / (1/0.9375) + 0.03125

    return (u_coord, v_coord)

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

running = True
dt = 0
mouse = ArduinoMouse2(1.0)
windows_cursor = WindowsCursor(1920, 1080)
mame_cursor = MAMECursor(1920, 1080)

# Load calibration
with open("calibration_new_1.json", 'r') as f:
    calib = json.load(f)

# Convert calib to numpy arrays
for key, val in calib.items():
    calib[key] = np.asfarray(val)

rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0.0, -1, 0, 0], [0, 0, 0, 0]])

device_to_gun_1 = np.load('device_to_gun_1.npy')
device_to_gun_2 = np.load('device_to_gun_2.npy')
# print(device_to_gun)
print(calib)


mouse_buttom = 0
frame = 0
st = time.time()


save_number = 0
while mode != 'exit':

    pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))

    if pose_matrix is None or np.isnan(np.sum(pose_matrix)):
        time.sleep(0.002)
        continue

    pose_matrix = np.asarray(pose_matrix)
    # pose_matrix = pose_matrix @ rotation_matrix
    pose_matrix_4x4 = np.eye(4)
    pose_matrix_4x4[:3, :] = pose_matrix
    pose_matrix = (device_to_gun_2 @ pose_matrix_4x4 @ device_to_gun_1)[:3, :]

    
    # The last column is the tracker translation (x, y, z)
    tracker_position = np.copy(pose_matrix[:, -1])

    # Zero the translation
    # pose_matrix[:, -1] = 0.0

    # print(original_pose_matrix)
    # print(pose_matrix)

    # Ajust the rotation in the expected axis
    # print(triad_openvr.convert_to_euler(pose_matrix))
    pose_matrix = pose_matrix @ rotation_matrix
    # if pose_matrix[1][0] >= 0 and pose_matrix[0][0] >= 0 or pose_matrix[1][0] < 0
    original = np.copy(pose_matrix)
    # pose_matrix[0][0]  = original[0][0] * -1
    # pose_matrix[1][0]  = original[1][0] * -1
    # pose_matrix[2][0]  = original[2][0] * -1

    # pose_matrix[0][1]  = original[2][1] * -1
    # pose_matrix[1][1]  = original[1][2] * -1
    # pose_matrix[2][1]  = original[0][1]

    # pose_matrix[0][2]  = original[2][2]
    # pose_matrix[1][2]  = original[1][1] * -1
    # pose_matrix[2][2]  = original[0][2] * -1

    # print(pose_matrix)

    tracker_direction = -pose_matrix[:3, 2]

    # print(tracker_position)

    # tracker_position -= tracker_direction * (0.115)
    # print(tracker_position)

    # print(triad_openvr.convert_to_euler(pose_matrix))
    # print('pos', tracker_position)
    # print('dir', tracker_direction)
    # print('dir, euler', unit_vector_to_euler_angles(list(tracker_direction)))

    # print(unit_vector_to_euler_angles(list(tracker_direction)))

    controller_input = v.devices["tracker_1"].get_controller_inputs()
    # print(controller_input)

    # print(controller_input)

    mouse_buttom = 0

    if controller_input['menu_button']:
        mouse_buttom += 2
    if controller_input['trackpad_touched']:
        mouse_buttom += 4
    if controller_input['trigger'] > 0.5:
        mouse_buttom += 1


    p_screen = project_ray_to_plane(calib['top_left'], calib['plane_normal'], tracker_position, tracker_direction)

    # if c_top_left:
    #     print('Setting top_left')
    #     calib['top_left'] = p_screen1
    #     c_top_left = False
    # if c_top_right:
    #     print('Setting top right')
    #     calib['top_right'] = p_screen1
    #     c_top_right = False
    # if c_bottom_left:
    #     print('Setting bottom_left')
    #     calib['bottom_left'] = p_screen1
    #     c_bottom_left = False
    # if c_bottom_right:
    #     print('Setting bottom_right')
    #     calib['bottom_right'] = p_screen1
    #     c_bottom_right = False

    # print(p_screen1, p_screen2, p_screen3, p_screen4)

    # Project point into the u/v screen coordinate system
    u_coord, v_coord = point_to_uv_coordinates_multi(calib, p_screen)
    # print(f'U: {u_coord}, V: {v_coord}')

    if math.isnan(u_coord) or math.isnan(v_coord):
        time.sleep(0.002)
        continue

    if mode == 'windows':
        pos_x, pos_y = windows_cursor.get_position()
        mouse.update_position(pos_x, pos_y)
        # print(pos_x, pos_y)
        target_x, target_y = windows_cursor.process_target(u_coord, v_coord)
        mouse.set_sensi(1)
        mouse.move_mouse_absolute(target_x, target_y, mouse_buttom)
    elif mode == 'mame':
        if mame_cursor.has_new_pos():
            pos_x, pos_y = mame_cursor.get_position()
            mouse.update_position(pos_x, pos_y)
        target_x, target_y = mame_cursor.process_target(u_coord, v_coord)
        mouse.set_sensi(1.0)
        # mouse.set_sensi(1)
        mouse.move_mouse_absolute(target_x, target_y, mouse_buttom)

    # print('Done setting pos')
    # print(target_x)
    # print(target_y)
    # print(tracker_position, tracker_direction)
    time.sleep(0.002)

mouse.close()
mame_cursor.close()
listener.stop()