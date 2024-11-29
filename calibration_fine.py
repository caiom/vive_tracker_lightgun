#%%
import time
import numpy as np
import json
from scipy.optimize import least_squares, minimize, differential_evolution, basinhopping
from sklearn import linear_model
from itertools import combinations
from scipy.spatial.transform import Rotation
import copy

NUM_POINTS = 5


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


def point_to_ray_distance(P, R, D):
    """
    Calculate the minimum distance from a point to a ray in 3D space.
    
    Parameters:
    - P: Coordinates of the point as a numpy array [px, py, pz].
    - R: Coordinates of a point on the ray as a numpy array [rx, ry, rz].
    - D: Direction vector of the ray as a numpy array [rdx, rdy, rdz].
    
    Returns:
    - The minimum distance from point P to the ray defined by R and D.
    """
    # Vector from point on the ray to P
    RP = P - R
    
    # Projection of RP onto the direction vector D
    proj_RP_D = (np.dot(RP, D) / np.dot(D, D)) * D
    
    # Vector from P to the closest point on the ray
    closest_vector = RP - proj_RP_D
    
    # Distance is the magnitude of the closest_vector
    distance = np.linalg.norm(closest_vector)
    
    return distance


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

    # v_coord = v_coord / (1/0.8888888888888889) + 0.0555555555555556
    # u_coord = u_coord / (1/0.9375) + 0.03125

    return (u_coord, v_coord)

def closest_points_on_rays(o1, d1, o2, d2):

    # Calculate the vector connecting the origins
    w0 = o1 - o2

    # Calculate dot products
    A = np.dot(d1, d1)
    B = np.dot(d1, d2)
    C = np.dot(d2, d2)
    D = np.dot(d1, w0)
    E = np.dot(d2, w0)

    # Check if the rays are parallel (or almost parallel)
    denominator = A * C - B * B
    if abs(denominator) < 1e-6:
        # Rays are almost parallel
        # In this case, we can just return the origin points as the closest points
        return o1, o2

    # Calculate the scalar parameters s and t
    s = (B * E - C * D) / denominator
    t = (A * E - B * D) / denominator

    # Calculate the closest points P1 and P2
    P1 = o1 + s * d1
    P2 = o2 + t * d2

    return P1, P2

def plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    t = np.dot(plane_point - ray_origin, plane_normal) / np.dot(ray_direction, plane_normal)
    intersection_point = ray_origin + t * ray_direction
    return intersection_point

def calculate_pos_and_dir_from_pose(pose_matrix):
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = np.copy(pose_matrix[2, :3])
    tracker_direction[2] = -tracker_direction[2]

    return (tracker_position, tracker_direction)

def calculate_plane_points(screen_corners):

    print("Calculate_plane_points")

    calibration_dict = {}
    # calibration_dict['error'] = 0.0
    # for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center', 'qbl', 'qtl', 'qtr', 'qbr']):
    for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center']):
    # for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right']):
        # print(corner)
        corner_points = []
        for read, j in enumerate(range(i, len(screen_corners), NUM_POINTS)):
            corner_points.append(screen_corners[j])

        point_comb = list(combinations(corner_points, 2))
        # for j in range(0, len(corner_points), 2):
        #     for k in range(1, len(corner_points), 2):
        #         point_comb.append((corner_points[j], corner_points[k]))

        comb_points = []
        comb_points_dist = []
        print(point_comb)
        print(len(point_comb))
        for ray1, ray2 in point_comb:
            p1, p2 = closest_points_on_rays(ray1[0], ray1[1], ray2[0], ray2[1])
            mean_point = (p1 + p2) / 2
            # print(mean_point)
            comb_points += [mean_point]
            comb_points_dist += [np.linalg.norm(p1-p2)]

        print(f'{corner}: mean dist {np.mean(comb_points_dist)} max dist: {np.max(comb_points_dist)}')

        # print(f'{corner}: {comb_points}')
        calibration_dict[corner] = np.mean(comb_points, axis=0)

        # for corner_point in corner_points:
        #     calibration_dict['error'] += point_to_ray_distance(calibration_dict[corner], corner_point[0], corner_point[1]) / len(corner_points) / NUM_POINTS

    print(calibration_dict)

    return calibration_dict

def recalculate_plane_points(screen_corners, calib):

    calibration_dict = copy.deepcopy(calib)
    # calibration_dict['plane_error'] = 0.0

    # for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center', 'qbl', 'qtl', 'qtr', 'qbr']):
    for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center']):
    # for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right']):
        # print(corner)
        corner_points = []
        for j in range(i, len(screen_corners), NUM_POINTS):
            corner_points.append(screen_corners[j])

        comb_points = []
        for corner_point in corner_points:
            comb_points += [project_ray_to_plane(calib['ref_plane_normal'], calib['plane_normal'], corner_point[0], corner_point[1])]

        # calibration_dict['plane_error'] += np.linalg.norm(calibration_dict[corner] - np.median(comb_points, axis=0)) / NUM_POINTS
        calibration_dict[corner] = np.median(comb_points, axis=0)

    return calibration_dict

def calculate_plane(calib):

    # points = np.stack([calib['bottom_left'], calib['bottom_right'], calib['top_right'], calib['top_left'], calib['center'], calib['qbl'], calib['qtl'], calib['qtr'], calib['qbr']])
    points = np.stack([calib['bottom_left'], calib['bottom_right'], calib['top_right'], calib['top_left'], calib['center']])
    # points = np.stack([calib['bottom_left'], calib['bottom_right'], calib['top_right'], calib['top_left']])
    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(points[:, :2], points[:, 2:])
    # print('pre', calib['bottom_left'][2])
    # calib['bottom_left'][2] = linear_reg.predict(calib['bottom_left'][:2].reshape(1, -1))[0]
    # # print('after', calib['bottom_left'][2])
    # calib['bottom_right'][2] = linear_reg.predict(calib['bottom_right'][:2][None, ...])[0]
    # calib['top_right'][2] = linear_reg.predict(calib['top_right'][:2][None, ...])[0]
    # calib['top_left'][2] = linear_reg.predict(calib['top_left'][:2][None, ...])[0]

    # calib['plane_normal'] = np.cross(calib['bottom_left'] - calib['top_left'], calib['top_right'] - calib['top_left'])
    calib['plane_normal'] = np.array([linear_reg.coef_[0, 0], linear_reg.coef_[0, 1], -1])
    plane_ref = np.array([0, 0]).reshape(1, -1)
    plane_ref_z = linear_reg.predict(plane_ref)[0][0]
    calib['ref_plane_normal'] = np.array([0, 0, plane_ref_z])

    return calib

def error_function2(device_to_gun, base_matrix, pose_matrices, target_points_2d, mode='rotation', take_mean=True, save=False):

    transform_matrix = np.eye(4)
    if mode == 'rotation':
        transform_matrix[:3, :3] = Rotation.from_euler('xyz', device_to_gun[:3], degrees=True).as_matrix()
    elif mode == 'translation':
        transform_matrix[:3, 3] = device_to_gun
    elif mode == 'both':
        # transform_matrix[:3, :3] = Rotation.from_euler('xyz', device_to_gun[:3], degrees=True).as_matrix()
        # transform_matrix[:3, 3] = device_to_gun[3:6]
        transform_matrix[:3, :3] = Rotation.from_mrp(device_to_gun[:3]).as_matrix()
        transform_matrix[:3, 3] = device_to_gun[3:6]

    transform_matrix2 = np.eye(4)
    transform_matrix2[:3, :3] = Rotation.from_euler('xyz', device_to_gun[6:9], degrees=True).as_matrix()
    transform_matrix2[:3, 3] = device_to_gun[9:12]


    if save:
        np.save('device_to_gun_1.npy', transform_matrix)
        np.save('device_to_gun_2.npy', transform_matrix2)


    # transform_matrix2 = np.eye(4)
    # transform_matrix2[:3, :3] = np.reshape(device_to_gun[12:21], (3, 3))
    # transform_matrix2[:3, 3] = device_to_gun[21:]


    # print(transform_matrix)

    pose_matrices_4x4 = []
    for pose_matrix in pose_matrices:
        pose_matrix_4x4 = np.eye(4)
        pose_matrix_4x4[:3, :] = pose_matrix
        # pose_matrix_4x4[:3, 3] += device_to_gun[9:]
        pose_matrices_4x4.append(pose_matrix_4x4)

    # pose_matrices_copy = [np.copy(pose_matrix) for pose_matrix in pose_matrices]
    translation_rotated_4x4 = np.eye(4)
    # transform_matrix[:3, :3] = Rotation.from_mrp(device_to_gun[3:6]).as_matrix()
    translation_rotated_4x4[:3, 3] = device_to_gun[3:6]

    if save:
        np.save('device_to_gun_translation.npy', translation_rotated_4x4)

    # gun_to_world = [(transform_matrix @ pose_matrix @ translation_rotated_4x4)[:3, :] for pose_matrix in pose_matrices_4x4]
    gun_to_world = [(transform_matrix2 @ pose_matrix @ transform_matrix)[:3, :] for pose_matrix in pose_matrices_4x4]
    # gun_to_world = [(transform_matrix2 @ pose_matrix)[:3, :] for pose_matrix in gun_to_world]


    # transform_matrix = np.copy(base_matrix)
    # if mode == 'rotation':
    #     transform_matrix[:3, :3] = np.reshape(device_to_gun, (3, 3))
    # elif mode == 'translation':
    #     transform_matrix[:3, 3] = device_to_gun
    # elif mode == 'both':
    #     transform_matrix[:3, :3] = np.reshape(device_to_gun[:9], (3, 3))
    #     # transform_matrix[:3, 3] = device_to_gun[9:]


    # # print(transform_matrix)

    # pose_matrices_4x4 = []
    # for pose_matrix in pose_matrices:
    #     pose_matrix_4x4 = np.eye(4)
    #     pose_matrix_4x4[:3, :] = pose_matrix
    #     # pose_matrix_4x4[:3, 3] += device_to_gun[9:]
    #     pose_matrices_4x4.append(pose_matrix_4x4)

    pose_matrices_copy = [np.copy(pose_matrix) for pose_matrix in pose_matrices]

    # gun_to_world = [(transform_matrix @ pose_matrix)[:3, :] for pose_matrix in pose_matrices_4x4]

    # for g_w, pm in zip(gun_to_world, pose_matrices_copy):
    #     g_w[:, 3] = pm[:, 3] + device_to_gun[9:]

    # print(gun_to_world[0])
    # print(pose_matrices[0])
    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in gun_to_world]
    # pos_dir_points_orig = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in pose_matrices_copy]
    # print(pos_dir_points[0])
    calib = calculate_plane_points(pos_dir_points)
    calib = calculate_plane(calib)
    calib = recalculate_plane_points(pos_dir_points, calib)
    if save:
        calibration_dict = {}
        # # Convert to lists
        for key, val in calib.items():
            calibration_dict[key] = list(val)

        # Save
        with open("calibration_new_1.json", 'w') as f:
            json.dump(calibration_dict, f, indent=2)
    # print(calib)
    plane_points_3d = [project_ray_to_plane(calib['top_left'], calib['plane_normal'], pos, dir) for pos, dir in pos_dir_points]
    # print(plane_points_3d[0])
    # plane_points_u_v = [point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  p_screen, device_to_gun[6], device_to_gun[7]) for p_screen in plane_points_3d]
    plane_points_u_v = [point_to_uv_coordinates_multi(calib, p_screen) for p_screen in plane_points_3d]
    # print(plane_points_u_v[0])
    # print(plane_points_u_v[9])
    # print(plane_points_u_v[18])

    if not take_mean:

        errors = []

        for i, target_point in enumerate(target_points_2d):
            for read, j in enumerate(range(i, len(plane_points_u_v), NUM_POINTS)):
                    errors.extend((np.array(plane_points_u_v[j]) - target_point).flatten())

        return errors

    else:

        error = 0.0

        for i, target_point in enumerate(target_points_2d):
            for j in range(i, len(plane_points_u_v), NUM_POINTS):
                error += np.linalg.norm(np.array(plane_points_u_v[j]) - target_point)


        # print(f'Error: {error}')

        return error / len(plane_points_u_v)
    

def error_function3(device_to_gun, base_matrix, pose_matrices, target_points_2d, mode='rotation', take_mean=True, save=False):

    transform_matrix = np.eye(4)
    if mode == 'rotation':
        transform_matrix[:3, :3] = Rotation.from_euler('xyz', device_to_gun[:3], degrees=True).as_matrix()
    elif mode == 'translation':
        transform_matrix[:3, 3] = device_to_gun
    elif mode == 'both':
        transform_matrix[:3, :3] = Rotation.from_euler('xyz', device_to_gun[:3], degrees=True).as_matrix()
        transform_matrix[:3, 3] = device_to_gun[3:6]


    if save:
        np.save('device_to_gun_1.npy', transform_matrix)

    gun_to_world_list = []
    for pose_matrix in pose_matrices:
        gun_to_world = np.eye(4)
        gun_to_world = pose_matrix @ transform_matrix

        # gun_to_world[:3, :3] = pose_matrix[:3,:3] @ transform_matrix[:3, :3]
        # gun_to_world[:3, 3] = (pose_matrix[:3,:3] @ transform_matrix[:3, 3]) + pose_matrix[:3,3]
        gun_to_world_list.append(gun_to_world)

    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in gun_to_world_list]
    calib = calculate_plane_points(pos_dir_points)
    calib = calculate_plane(calib)
    calib = recalculate_plane_points(pos_dir_points, calib)
    if save:
        calibration_dict = {}
        # # Convert to lists
        for key, val in calib.items():
            calibration_dict[key] = list(val)

        # Save
        with open("calibration_new_1.json", 'w') as f:
            json.dump(calibration_dict, f, indent=2)
    # print(calib)
    plane_points_3d = [project_ray_to_plane(calib['top_left'], calib['plane_normal'], pos, dir) for pos, dir in pos_dir_points]
    # print(plane_points_3d[0])
    # plane_points_u_v = [point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  p_screen, device_to_gun[6], device_to_gun[7]) for p_screen in plane_points_3d]
    plane_points_u_v = [point_to_uv_coordinates_multi(calib, p_screen) for p_screen in plane_points_3d]
    # print(plane_points_u_v[0])
    # print(plane_points_u_v[9])
    # print(plane_points_u_v[18])

    if not take_mean:

        errors = []

        for i, target_point in enumerate(target_points_2d):
            for read, j in enumerate(range(i, len(plane_points_u_v), NUM_POINTS)):
                    errors.extend((np.array(plane_points_u_v[j]) - target_point).flatten())

        return errors

    else:

        error = 0.0

        for i, target_point in enumerate(target_points_2d):
            for j in range(i, len(plane_points_u_v), NUM_POINTS):
                error += np.linalg.norm(np.array(plane_points_u_v[j]) - target_point)


        # print(f'Error: {error}')

        return error / len(plane_points_u_v)
        # return calib['error'] 


def error_function2_s(device_to_gun, pose_matrices, target_points_2d, mode):


    transform_matrix = np.copy(base_matrix)
    if mode == 'rotation':
        transform_matrix[:3, :3] = np.reshape(device_to_gun, (3, 3))
    elif mode == 'translation':
        transform_matrix[:3, 3] = device_to_gun
    elif mode == 'both':
        transform_matrix[:3, :3] = np.reshape(device_to_gun[:9], (3, 3))
        # transform_matrix[:3, 3] = device_to_gun[9:12]


    # transform_matrix2 = np.eye(4)
    # transform_matrix2[:3, :3] = np.reshape(device_to_gun[12:21], (3, 3))
    # transform_matrix2[:3, 3] = device_to_gun[21:]


    # print(transform_matrix)

    pose_matrices_4x4 = []
    for pose_matrix in pose_matrices:
        pose_matrix_4x4 = np.eye(4)
        pose_matrix_4x4[:3, :] = pose_matrix
        pose_matrix_4x4[:3, 3] += device_to_gun[9:]
        pose_matrices_4x4.append(pose_matrix_4x4)

    pose_matrices_copy = [np.copy(pose_matrix) for pose_matrix in pose_matrices]

    gun_to_world = [(transform_matrix @ pose_matrix)[:3, :] for pose_matrix in pose_matrices_4x4]
    # gun_to_world = [(transform_matrix2 @ pose_matrix)[:3, :] for pose_matrix in gun_to_world]

    # for g_w, pm in zip(gun_to_world, pose_matrices_copy):
    #     g_w[:, 3] = pm[:, 3] + device_to_gun[9:]
    # print(gun_to_world[0])
    # print(pose_matrices[0])
    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in gun_to_world]
    # pos_dir_points_orig = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in pose_matrices_copy]
    # print(pos_dir_points[0])
    calib = calculate_plane_points(pos_dir_points)
    calib = calculate_plane(calib)
    # print(calib)
    plane_points_3d = [project_ray_to_plane(calib['top_left'], calib['plane_normal'], pos, dir) for pos, dir in pos_dir_points]
    # print(plane_points_3d[0])
    plane_points_u_v = [point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  p_screen, ) for p_screen in plane_points_3d]
    # print(plane_points_u_v[0])
    # print(plane_points_u_v[9])
    # print(plane_points_u_v[18])

    error = 0.0

    for i, target_point in enumerate(target_points_2d):
        for j in range(i, len(plane_points_u_v), 9):
            error += np.linalg.norm(np.array(plane_points_u_v[j]) - target_point)


    # print(f'Error: {error}')

    return error / len(plane_points_u_v)


def error_function_p(device_to_gun, base_matrix, pose_matrices, target_points_2d, mode, save=False):


    transform_matrix = np.copy(base_matrix)
    if mode == 'rotation':
        transform_matrix[:3, :3] = np.reshape(device_to_gun, (3, 3))
    elif mode == 'translation':
        transform_matrix[:3, 3] = device_to_gun
    elif mode == 'both':
        transform_matrix[:3, :3] = np.reshape(device_to_gun[:9], (3, 3))
        # transform_matrix[:3, 3] = device_to_gun[9:]

    if save:
        np.save('device_to_gun_transform.npy', transform_matrix)

    # print(transform_matrix)

    pose_matrices_4x4 = []
    for pose_matrix in pose_matrices:
        pose_matrix_4x4 = np.eye(4)
        pose_matrix_4x4[:3, :] = pose_matrix
        pose_matrix_4x4[:3, 3] += device_to_gun[9:]
        pose_matrices_4x4.append(pose_matrix_4x4)

    pose_matrices_copy = [np.copy(pose_matrix) for pose_matrix in pose_matrices]

    gun_to_world = [(transform_matrix @ pose_matrix)[:3, :] for pose_matrix in pose_matrices_4x4]
    # print(gun_to_world[0])
    # print(pose_matrices[0])
    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in gun_to_world]
    pos_dir_points_orig = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in pose_matrices_copy]
    # print(pos_dir_points[0])
    calib = calculate_plane_points(pos_dir_points)
    calib = calculate_plane(calib)
    # print(calib)
    if save:
        calibration_dict = {}
        # # Convert to lists
        for key, val in calib.items():
            calibration_dict[key] = list(val)

        # Save
        with open("calibration_new_1.json", 'w') as f:
            json.dump(calibration_dict, f, indent=2)
    plane_points_3d = [project_ray_to_plane(calib['top_left'], calib['plane_normal'], pos, dir) for pos, dir in pos_dir_points]
    # print(plane_points_3d[0])
    plane_points_u_v = [point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  p_screen) for p_screen in plane_points_3d]
    # print(plane_points_u_v[0])
    # print(plane_points_u_v[9])
    # print(plane_points_u_v[18])

    errors = []

    for i, target_point in enumerate(target_points_2d):

        plane_points = []
        for j in range(i, len(plane_points_u_v), 9):
            plane_points.append(np.array(plane_points_u_v[j]))

        np.stack(plane_points)
        mean_point = np.mean(plane_points, axis=0)

        print(mean_point)
        print(target_point)

        errors.extend((mean_point - target_point).flatten())

    return errors


pos_mapping = {0: (0.03125, 1-0.0555555555555556),
               1: (0.03125, 0.0555555555555556),
               2: (1-0.03125, 0.0555555555555556),
               3: (1-0.03125, 1-0.0555555555555556),
               4: (0.5, 0.5),
               5: (1/4, 3/4),
               6: (1/4, 1/4),
               7: (3/4, 1/4),
               8: (3/4, 3/4),}

pos_mapping = {0: (0.0, 1.0),
               1: (0.0, 0.0),
               2: (1.0, 0.0),
               3: (1.0, 1.0),
               4: (0.5, 0.5),
               5: (1/4, 3/4),
               6: (1/4, 1/4),
               7: (3/4, 1/4),
               8: (3/4, 3/4),}


pos_mapping = {0: (0.03125, 1-0.0555555555555556),
               1: (0.03125, 0.0555555555555556),
               2: (1-0.03125, 0.0555555555555556),
               3: (1-0.03125, 1-0.0555555555555556),
               4: (0.5, 0.5),}

pos_mapping = {0: (0, 1),
               1: (0, 0),
               2: (1, 0),
               3: (1, 1),
               4: (0.5, 0.5),}

target_points_2d = list(pos_mapping.values())
pose_matrices = []
for i in range(20):
    pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
    pose_matrices.append(pose_matrix)

print(len(pose_matrices))

device_to_gun_rot = np.array([0.0, 0.0, 0.])
# device_to_gun_rot = np.array([0.3088855 ,  0.05939724,  0.33346251,  0.03281174,  0.00630954, 0.03542247,  0.34026668,  0.06543169,  0.36734058])
device_to_gun_trans = np.array([0.0, 0.0, 0.])
# device_to_gun_trans = np.array([ -0.0073907 ,-0.21132067,  0.02597444])
# device_to_gun = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])

# errors = error_function2(device_to_gun, pose_matrices, target_points_2d)
# print(errors)

bounds_x = 20
bounds_y = 40
bounds_z = 300
angle_bound = 15

bounds_low_rot = [-angle_bound for _ in range(3)]
bounds_low_trans = [-bounds_x, -bounds_y, -bounds_z]
bounds_low = bounds_low_rot + bounds_low_trans


bounds_high_rot = [angle_bound for _ in range(3)]
bounds_high_trans = [bounds_x, bounds_y, bounds_z]
bounds_high = bounds_high_rot + bounds_high_trans



rot_bounds = [(-angle_bound, angle_bound) for _ in range(3)]
trans_bounds = [(neg, pos) for neg, pos in zip(bounds_low_trans, bounds_high_trans)]
dif_e_bounds = rot_bounds + trans_bounds

base_matrix = np.eye(4)

# errors = error_function_p(None, base_matrix, pose_matrices, target_points_2d, 'none', False)
# print(errors)

device_to_gun_rot_trans = np.concatenate((device_to_gun_rot, device_to_gun_trans))
# device_to_gun_rot_trans2 = np.copy(device_to_gun_rot)


# device_to_gun_rot_trans_double = np.concatenate((device_to_gun_rot_trans, device_to_gun_rot_trans2))

bounds_low_double = bounds_low + bounds_low
bounds_high_double = bounds_high + bounds_high



st = time.time()
print('Initial Error')
print(error_function3(device_to_gun_rot_trans, base_matrix, pose_matrices, target_points_2d, 'both', True, True))
import sys
sys.exit(0)
errors = error_function3(device_to_gun_rot_trans, base_matrix, pose_matrices, target_points_2d, 'both', False)
print(errors)

def print_errors(errors):

    num_reads = len(errors)//(NUM_POINTS*2)
    point_size = num_reads*2

    for point in range(NUM_POINTS):
        curr_point_list_x = []
        curr_point_list_y = []
        for reading in range(num_reads):
            p_s = point*point_size
            r_s = reading*2
            curr_point_list_x.append(errors[p_s+r_s])
            curr_point_list_y.append(errors[p_s+r_s+1])
        mean_norm = np.linalg.norm(np.stack((curr_point_list_x, curr_point_list_y)), axis=0)
        print(f'Point {point}: error x: {np.mean(curr_point_list_x):.3f} error y: {np.mean(curr_point_list_y):.3f}, error norm: {np.mean(mean_norm)}')

    for reading in range(num_reads):
        curr_point_list_x = []
        curr_point_list_y = []
        for point in range(NUM_POINTS):
            p_s = point*point_size
            r_s = reading*2
            curr_point_list_x.append(errors[p_s+r_s])
            curr_point_list_y.append(errors[p_s+r_s+1])

        print(f'Read z: {pose_matrices[reading*4][2, 3]}')
        print(f'Read x: {pose_matrices[reading*4][0, 3]}')
        mean_norm = np.linalg.norm(np.stack((curr_point_list_x, curr_point_list_y)), axis=0)
        print(f'Read {reading}: error x: {np.mean(curr_point_list_x):.3f} error y: {np.mean(curr_point_list_y):.3f}, error norm: {np.mean(mean_norm)}')


print(f'Time 2s: {time.time()-st}')

# res = least_squares(error_function2, device_to_gun_rot_trans, args=(base_matrix, pose_matrices, target_points_2d, 'both', False), method='trf', loss='huber')  bounds=(bounds_low, bounds_high),
res = least_squares(error_function3, device_to_gun_rot_trans, args=(base_matrix, pose_matrices, target_points_2d, 'both', False), ftol=1e-15, xtol=1e-20, gtol=1e-15, bounds=(bounds_low, bounds_high))
print(res.x)
print('Error after Min squares 1')
print(error_function3(res.x, base_matrix, pose_matrices, target_points_2d, 'both', True))
errors = error_function3(res.x, base_matrix, pose_matrices, target_points_2d, 'both', False, False)

print_errors(errors)

# import sys
# sys.exit(0)


print(res)

# base_matrix[:3, :3] = np.reshape(res.x[:9], (3, 3))

# base_matrix[:3, 3] = res.x[9:]

# print(base_matrix)

# for _ in range(2):
#     print(base_matrix)
#     res = least_squares(error_function2, device_to_gun_rot, args=(base_matrix, pose_matrices, target_points_2d, 'rotation'), bounds=(bounds_low_rot, bounds_high_rot), ftol=1e-15, xtol=1e-20, gtol=1e-15)
#     print(res.cost)
#     base_matrix[:3, :3] = np.reshape(res.x, (3, 3))
#     res = least_squares(error_function2, device_to_gun_trans, args=(base_matrix, pose_matrices, target_points_2d, 'translation'), bounds=(bounds_low_trans, bounds_high_trans), ftol=1e-15, xtol=1e-20, gtol=1e-15)
#     print(res.cost)
#     base_matrix[:3, 3] = res.x
#     print(base_matrix)
    

# res = minimize(error_function2_s, device_to_gun, args=(pose_matrices, target_points_2d), bounds=bounds)
# res = differential_evolution(error_function2_s, args=(pose_matrices, target_points_2d), bounds=bounds, workers=-1, disp=True)
# res = basinhopping(error_function2_s, device_to_gun, args=(pose_matrices, target_points_2d), bounds=bounds)

# st = time.time()
# print(error_function2_s(res.x, pose_matrices, target_points_2d, 'both'))
# print(f'Time 2s: {time.time()-st}')
res = differential_evolution(error_function3, args=(device_to_gun_rot_trans, pose_matrices, target_points_2d, 'both', True), bounds=dif_e_bounds, workers=1, maxiter=500, disp=True, tol=0.000000001)
mean_error = error_function3(res.x, base_matrix, pose_matrices, target_points_2d, 'both', True, False)
errors = error_function3(res.x, base_matrix, pose_matrices, target_points_2d, 'both', False, False)
print(res.x)

print('Error after diff eval')
print(mean_error)
print_errors(errors)

res = least_squares(error_function3, res.x, args=(base_matrix, pose_matrices, target_points_2d, 'both', False), ftol=1e-15, xtol=1e-20, gtol=1e-15, bounds=(bounds_low, bounds_high))
print(res.x)
print('Error least squares 2')
print(error_function3(res.x, base_matrix, pose_matrices, target_points_2d, 'both', True))
errors = error_function3(res.x, base_matrix, pose_matrices, target_points_2d, 'both', False, True)

print_errors(errors)
# res = least_squares(error_function2, res.x, args=(base_matrix, pose_matrices, target_points_2d, 'both'), bounds=(bounds_low, bounds_high), ftol=1e-15, xtol=1e-20, gtol=1e-15)
# base_matrix[:3, :3] = np.reshape(res.x[:9], (3, 3))
# base_matrix[:3, 3] = res.x[9:]

# st = time.time()
# errors = error_function2_s(res.x, pose_matrices, target_points_2d, 'both')
# print(time.time()-st)
# print(errors)

# base_matrix[:3, :3] = np.reshape(res.x[:9], (3, 3))
# base_matrix[:3, 3] = res.x[9:]

# errors = error_function_p(None, base_matrix, pose_matrices, target_points_2d, 'none', True)
# print(errors)









# # controller_input = v.devices["tracker_1"].get_controller_inputs()


# from sklearn import linear_model

# points = np.stack([calib['bottom_left'], calib['bottom_right'], calib['top_right'], calib['top_left'], calib['center'], calib['qbl'], calib['qtl'], calib['qtr'], calib['qbr']])
# linear_reg = linear_model.LinearRegression()
# print(points[:2])
# plane_model = linear_reg.fit(points[:, :2], points[:, 2:])
# print(calib['bottom_left'][:2][None, ...])
# calib['bottom_left'][2] = linear_reg.predict(calib['bottom_left'][:2].reshape(1, -1))[0]
# calib['bottom_right'][2] = linear_reg.predict(calib['bottom_right'][:2][None, ...])[0]
# calib['top_right'][2] = linear_reg.predict(calib['top_right'][:2][None, ...])[0]
# calib['top_left'][2] = linear_reg.predict(calib['top_left'][:2][None, ...])[0]

# calib['top_left_screen'] = calib['top_left']
# calib['top_right_screen'] = calib['top_right']
# calib['bottom_left_screen'] = calib['bottom_left']

# calib['plane_normal'] = np.cross(calib['bottom_left'] - calib['top_left'], calib['top_right'] - calib['top_left'])

# # Example data
# rays_origins = [
#     [np.array([0, 0, 0]), np.array([0, 0, 0])],
#     [np.array([0, 0, 0]), np.array([0, 0, 0])]
# ]

# rays_directions = [
#     [np.array([1, 0, 1]), np.array([1, 0.1, 1])],
#     [np.array([0, 1, 1]), np.array([0.1, 1, 1])]
# ]

# target_points = [np.array([3, 0, 3]), np.array([0, 3, 3])]
# device_to_world = np.eye(4)

# # Optimize the transformation matrix
# initial_transform = np.zeros(12)
# res = least_squares(error_function, initial_transform, args=(rays_origins, rays_directions, target_points, device_to_world))

# optimized_transform = np.eye(4)
# optimized_transform[:3, :3] = np.reshape(res.x[:9], (3, 3))
# optimized_transform[:3, 3] = res.x[9:]

# # Apply the transformation
# new_device_to_world = np.dot(device_to_world, np.linalg.inv(optimized_transform))

# print("Optimized transform:")
# print(optimized_transform)

# print("New device to world matrix:")
# print(new_device_to_world)