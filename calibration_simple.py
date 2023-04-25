#%%
import time
import numpy as np
import json
from scipy.optimize import least_squares, minimize, differential_evolution, basinhopping
from sklearn import linear_model
from itertools import combinations
from scipy.spatial.transform import Rotation

def find_closest_point_and_distances(ray_positions, ray_directions):
    N = len(ray_positions)
    assert N == len(ray_directions), "Mismatch in number of positions and directions"

    P = np.asarray(ray_positions)
    D = np.asarray(ray_directions)

    A = np.zeros((3 * N, N))
    b = np.zeros(3 * N)

    for i in range(N):
        A[3 * i:3 * i + 3, i] = -D[i]
        b[3 * i:3 * i + 3] = P[i]

    A_pseudo_inverse = np.linalg.pinv(A.T @ A) @ A.T
    t_values = A_pseudo_inverse @ b

    Q = P + np.outer(t_values, np.ones(3)) * D
    C = np.mean(Q, axis=0)

    distances = np.linalg.norm(C - Q, axis=1)

    return C, distances


def calculate_pos_and_dir_from_pose(pose_matrix, sensor_to_gun):
    # The last column is the tracker translation (x, y, z)
    pose_matrix = np.copy(pose_matrix)
    tracker_position = np.copy(pose_matrix[:, -1])

    # Zero the translation
    pose_matrix[:, -1] = 0.0


    # rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0.0, -1, 0, 0], [0, 0, 0, 0]])

    # pose_matrix = pose_matrix @ rotation_matrix

    # Ajust the rotation in the expected axis
    tracker_direction = pose_matrix @ sensor_to_gun @ np.asarray([[0], [0], [1.0], [1.0]])
    tracker_direction = tracker_direction[:, 0]
    tracker_direction /= np.linalg.norm(tracker_direction)

    return (tracker_position, tracker_direction)

def error_function2(device_to_gun, pose_matrices, mode='rotation', take_mean=False):


    transform_matrix = np.copy(base_matrix)
    if mode == 'rotation':
        transform_matrix[:3, :3] = np.reshape(device_to_gun, (3, 3))
    elif mode == 'translation':
        transform_matrix[:3, 3] = device_to_gun
    elif mode == 'both':
        transform_matrix[:3, :3] = np.reshape(device_to_gun[:9], (3, 3))
        transform_matrix[:3, 3] = device_to_gun[9:]

    pose_matrices_4x4 = []
    for pose_matrix in pose_matrices:
        pose_matrix_4x4 = np.eye(4)
        pose_matrix_4x4[:3, :] = pose_matrix
        # pose_matrix_4x4[:3, 3] += device_to_gun[9:]
        pose_matrices_4x4.append(pose_matrix_4x4)

    # gun_to_world = [(pose_matrix @ transform_matrix)[:3, :] for pose_matrix in pose_matrices_4x4]

    pose_matrices_copy = [np.copy(pose_matrix) for pose_matrix in pose_matrices]

    # for g_w, pm in zip(gun_to_world, pose_matrices_copy):
    #     g_w[:, 3] = pm[:, 3] + device_to_gun[9:]

    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix, transform_matrix) for pose_matrix in pose_matrices]
    # print(pos_dir_points)

    errors = []

    for i in range(9):
        ray_positions = []
        ray_directions = []

        for j in range(i, len(pos_dir_points), 9):
            ray_positions.append(pos_dir_points[j][0])
            ray_directions.append(pos_dir_points[j][1])

        ray_positions = np.stack(ray_positions)
        ray_directions = np.stack(ray_directions)
        closest_point, distances = find_closest_point_and_distances(ray_positions, ray_directions)
        errors += list(distances)

    # print(errors)

    if take_mean:
        return np.mean(errors)
    else:
        return errors

pose_matrices = []
for i in range(9, 45):
    pose_matrix = np.load(f'pose_matrix_{i}.npy')
    pose_matrices.append(pose_matrix)

device_to_gun_rot = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
device_to_gun_trans = np.array([0.0, 0.0, 0.])

bounds_low_rot = [-1 for _ in range(9)]
bounds_low_trans = [-0.22 for _ in range(3)]
bounds_low = bounds_low_rot + bounds_low_trans


bounds_high_rot = [1 for _ in range(9)]
bounds_high_trans = [0.22 for _ in range(3)]
bounds_high = bounds_high_rot + bounds_high_trans

base_matrix = np.eye(4)

device_to_gun_rot_trans = np.concatenate((device_to_gun_rot, device_to_gun_trans))
device_to_gun_rot_trans2 = np.copy(device_to_gun_rot_trans)


device_to_gun_rot_trans_double = np.concatenate((device_to_gun_rot_trans, device_to_gun_rot_trans2))

bounds_low_double = bounds_low + bounds_low
bounds_high_double = bounds_high + bounds_high

print(error_function2(device_to_gun_rot_trans, pose_matrices, 'both', True))
error_list = error_function2(device_to_gun_rot_trans, pose_matrices, 'both', False)
print(error_list)

for i in range(9):
    curr_point_list = []
    for j in range(i, len(error_list), 9):
        curr_point_list.append(error_list[j])
    print(f'Point {i}: error: {np.mean(curr_point_list):.3f}')


for j in range(0, len(error_list), 9):
    curr_read_list = []
    for i in range(9):
        curr_read_list.append(error_list[j+i])
    print(f'Read {j/9}: error: {np.mean(curr_read_list):.3f}')

res = least_squares(error_function2, device_to_gun_rot_trans, args=(pose_matrices, 'both', False), bounds=(bounds_low, bounds_high), ftol=1e-15, xtol=1e-20, gtol=1e-15)

print(error_function2(res.x, pose_matrices, 'both', True))
error_list = error_function2(res.x, pose_matrices, 'both', False)
print(error_list)

for i in range(9):
    curr_point_list = []
    for j in range(i, len(error_list), 9):
        curr_point_list.append(error_list[j])
    print(f'Point {i}: error: {np.mean(curr_point_list):.5f}')


for j in range(0, len(error_list), 9):
    curr_read_list = []
    for i in range(9):
        curr_read_list.append(error_list[j+i])
    print(f'Read {j/9}: error: {np.mean(curr_read_list):.5f}')



