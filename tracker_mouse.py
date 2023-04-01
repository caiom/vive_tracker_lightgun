import triad_openvr
import time
import numpy as np

import time
import json
from arduino_test import ArduinoMouse2, WindowsCursor
from receive_udp import MAMECursor


from pynput import keyboard

use_mame = False

def on_press(key):
    global use_mame
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        if key.char == 'm':
            use_mame = True
        if key.char == 'w':
            use_mame = False
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def unit_vector_to_euler_angles(vector, up=np.array([0, 1, 0])):
    vx, vy, vz = vector
    
    # Calculate yaw angle (θ)
    yaw = np.arctan2(vy, vx)
    
    # Calculate pitch angle (ϕ)
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

    return np.clip(u, 0, 1), np.clip(v, 0, 1)

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

running = True
dt = 0
mouse = ArduinoMouse2(1.0)
windows_cursor = WindowsCursor(2560, 1440)
mame_cursor = MAMECursor(2560, 1440)
from pynput import keyboard

# Load calibration
with open("calibration.json", 'r') as f:
    calib = json.load(f)

# Convert calib to numpy arrays
for key, val in calib.items():
    calib[key] = np.asfarray(val)



# controller_input = v.devices["tracker_1"].get_controller_inputs()


# from sklearn import linear_model

# points = np.stack([calib['bottom_left'], calib['bottom_right'], calib['top_right'], calib['top_left']])
# linear_reg = linear_model.LinearRegression()
# print(points[:2])
# plane_model = linear_reg.fit(points[:, :2], points[:, 2:])
# print(calib['bottom_left'][:2][None, ...])
# calib['bottom_left'][2] = linear_reg.predict(calib['bottom_left'][:2].reshape(1, -1))[0]
# calib['bottom_right'][2] = linear_reg.predict(calib['bottom_right'][:2][None, ...])[0]
# calib['top_right'][2] = linear_reg.predict(calib['top_right'][:2][None, ...])[0]
# calib['top_left'][2] = linear_reg.predict(calib['top_left'][:2][None, ...])[0]

# calib['plane_normal1'] = np.cross(calib['bottom_left'] - calib['top_left'], calib['top_right'] - calib['top_left'])
# calib['plane_normal2'] = np.cross(calib['bottom_right'] - calib['bottom_left'], calib['top_left'] - calib['bottom_left'])
# calib['plane_normal3'] = np.cross(calib['top_right'] - calib['bottom_right'], calib['bottom_left'] - calib['bottom_right'])
# calib['plane_normal4'] = np.cross(calib['bottom_right'] - calib['top_right'], calib['top_left'] - calib['top_right'])

rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0.0, -1, 0, 0], [0, 0, 0, 0]])
mouse_buttom = 0
frame = 0
st = time.time()



pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
original_pose_matrix = np.asarray(pose_matrix)


# Zero the translation
original_pose_matrix[:, -1] = 0.0
while running:
    # st = time.time()
    # controller_input = v.devices["tracker_1"].get_controller_inputs()

    # if controller_input['trigger'] > 0.0:
    #     mouse_buttom = 1
    # else:
    #     mouse_buttom = 0

    # print(v.devices["tracker_1"].get_pose_euler())
    pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
    pose_matrix = np.asarray(pose_matrix)

    # The last column is the tracker translation (x, y, z)
    tracker_position = np.copy(pose_matrix[:, -1])

    # Zero the translation
    pose_matrix[:, -1] = 0.0

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
    
    # print(triad_openvr.convert_to_euler(pose_matrix))
    tracker_direction = pose_matrix @ np.asarray([[0], [0], [-1.0], [1.0]])
    # print(tracker_direction.shape, rotation_matrix.shape)
    # tracker_direction = rotation_matrix @ tracker_direction
    # print(pose_matrix)
    tracker_direction = tracker_direction[:, 0]
    tracker_direction /= np.linalg.norm(tracker_direction)


    # print(unit_vector_to_euler_angles(list(tracker_direction)))

    


    controller_input = v.devices["tracker_1"].get_controller_inputs()

    if controller_input['trigger'] > 0.0:
        mouse_buttom = 1
    else:
        mouse_buttom = 0


    p_screen1 = project_ray_to_plane(calib['top_left'], calib['plane_normal'], tracker_position, tracker_direction)
    # p_screen2 = project_ray_to_plane(calib['bottom_left'], calib['plane_normal2'], tracker_position, tracker_direction)
    # p_screen3 = project_ray_to_plane(calib['bottom_right'], calib['plane_normal3'], tracker_position, tracker_direction)
    # p_screen4 = project_ray_to_plane(calib['top_right'], calib['plane_normal4'], tracker_position, tracker_direction)

    # print(p_screen1, p_screen2, p_screen3, p_screen4)

    # Project point into the u/v screen coordinate system
    u_coord1, v_coord1 = point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  p_screen1)
    # u_coord2, v_coord2 = point_to_uv_coordinates(calib['bottom_right'] - calib['bottom_left'], calib['top_left'] - calib['bottom_left'], calib['bottom_left'], p_screen2)
    # u_coord3, v_coord3 = point_to_uv_coordinates(calib['bottom_left'] - calib['bottom_right'], calib['top_right'] - calib['bottom_right'], calib['bottom_right'], p_screen3)
    # u_coord4, v_coord4 = point_to_uv_coordinates(calib['top_left'] - calib['top_right'], calib['bottom_right'] - calib['top_right'], calib['top_right'], p_screen4)

    # u_coord4 = 1 - u_coord4
    # v_coord2 = 1 - v_coord2
    # u_coord3 = 1 - u_coord3
    # v_coord3 = 1 - v_coord3

    # print(u_coord1, v_coord1, u_coord2, v_coord2, u_coord3, v_coord3, u_coord4, v_coord4)

    # u_coord = (u_coord1 * (1-v_coord1) + u_coord2 * v_coord1 + u_coord3 * v_coord1 + u_coord4* (1-v_coord1)) / 2
    # # u_coord = (u_coord1 * (1-v_coord1) + u_coord2 * v_coord1)
    # v_coord = (v_coord1 * (1-u_coord1) + v_coord2 * (1-u_coord1) + v_coord3 * u_coord1 + v_coord4 * u_coord1) / 2
    # # v_coord = (v_coord1 + v_coord2) / 2



    # print(u_coord1, v_coord1)
    if not use_mame:
        pos_x, pos_y = windows_cursor.get_position()
        mouse.update_position(pos_x, pos_y)
        # print(pos_x, pos_y)
        target_x, target_y = windows_cursor.process_target(u_coord1, v_coord1)
        mouse.move_mouse_absolute(target_x, target_y, mouse_buttom)
    else:
        pos_x, pos_y = mame_cursor.get_position()
        mouse.update_position(pos_x, pos_y)
        # print(pos_x, pos_y)
        target_x, target_y = mame_cursor.process_target(u_coord1, v_coord1)
        mouse.move_mouse_absolute(target_x, target_y, mouse_buttom)
    # print(target_x)
    # print(target_y)
    # print(tracker_position, tracker_direction)
    time.sleep(0.01)