#%%

import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, minimize, differential_evolution, basinhopping
from scipy.linalg import svd


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

def pose_to_pos_ori(pose):
     pos = pose[:3, 3]
     ori = np.array([-pose[0, 2], pose[0, 1], -pose[0, 0]])
     return (pos, ori)

pose1 = np.load('pose_matrix_0.npy')
ol1 = pose1[:, 3]
dl1 = np.array([-pose1[0, 2], pose1[0, 1], -pose1[0, 0]])

pose2 = np.load('pose_matrix_5.npy')
ol2 = pose2[:, 3]
dl2 = np.array([-pose2[0, 2], pose2[0, 1], -pose2[0, 0]])

o1 = np.array([0.16, -0.14, 0.27])
d1 = np.array([-0.12, 0.98, -0.13])
o2 = np.array([0.57, -0.09, 0.29])
d2 = np.array([-0.57, 0.8, -0.14])

print(closest_points_on_rays(o1, d1, o2, d2))
print(closest_points_on_rays(ol1, dl1, ol2, dl2))



poses = []

for i in range(15):
     poses.append(np.load(f'pose_matrix_{i}.npy'))

def error_func(device_to_gun, poses, verbose=False, return_calib=False):

    NUM_POINTS=5
    calibration_dict = {}
    error_close = 0.0

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = Rotation.from_euler('xyz', device_to_gun[:3], degrees=True).as_matrix()
    transform_matrix[:3, 3] = device_to_gun[3:6]

    gun_to_world_list = []
    for pose_matrix in poses:
        gun_to_world = np.eye(4)
        gun_to_world[:3, :3] = pose_matrix[:3,:3] @ transform_matrix[:3, :3]
        gun_to_world[:3, 3] = (pose_matrix[:3,:3] @ transform_matrix[:3, 3]) + pose_matrix[:3,3]
        gun_to_world_list.append(gun_to_world)

    screen_corners = [pose_to_pos_ori(pose) for pose in gun_to_world_list]

    for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center']):
            if verbose:
                print(corner)
            corner_points = []
            for read, j in enumerate(range(i, len(screen_corners), NUM_POINTS)):
                corner_points.append(screen_corners[j])

            point_comb = list(combinations(corner_points, 2))
            # for j in range(0, len(corner_points), 2):
            #     for k in range(1, len(corner_points), 2):
            #         point_comb.append((corner_points[j], corner_points[k]))
                
            # point_comb = [(corner_points[0], corner_points[1])]

            comb_points = []
            comb_points_dist = []
            for ray1, ray2 in point_comb:
                p1, p2 = closest_points_on_rays(ray1[0], ray1[1], ray2[0], ray2[1])
                if verbose:
                    print('Points')
                    print(p1, p2)
                mean_point = (p1 + p2) / 2
                if verbose:
                    print(f'Mean: {mean_point}')
                comb_points += [mean_point]
                # comb_points_dist += [np.linalg.norm(p1-p2)]

            # print(f'{corner}: mean dist {np.mean(comb_points_dist)} max dist: {np.max(comb_points_dist)}')

            # print(f'{corner}: {comb_points}')
            calibration_dict[corner] = np.mean(comb_points, axis=0)

            for n_p, corner_point in enumerate(corner_points):
                dist = point_to_ray_distance(calibration_dict[corner], corner_point[0], corner_point[1])
                if verbose:
                    print(f'Point {n_p} dist: {dist}')
                error_close += (dist / len(corner_points)) / NUM_POINTS


    plane_points = np.stack([calibration_dict['bottom_left'],
                         calibration_dict['top_left'],
                         calibration_dict['top_right'],
                         calibration_dict['bottom_right'],
                         calibration_dict['center']])

    # Step 2: Compute the centroid
    centroid = plane_points.mean(axis=0)

    # Step 3: Apply PCA to find the normal of the plane
    # Center the points at the origin
    points_centered = plane_points - centroid
    # Compute SVD
    U, S, Vt = svd(points_centered, full_matrices=False)
    # The normal vector of the plane is the last column of Vt
    normal = Vt[-1, :]
    residuals = np.abs(points_centered.dot(normal))
    if verbose:
        print('Residuals')
        print(residuals)
        print(np.mean(residuals))

        print('calib_error')
        print(error_close)

    calib = calibration_dict

    for key, val in list(calib.items()):

        # Calculate d using the plane equation ax + by + cz + d = 0
        d = -np.dot(normal, centroid)

        # Calculate the distance D from the point to the plane
        D = (np.dot(normal, val) + d) / np.linalg.norm(normal)

        # Project the point onto the plane
        P_prime = val - D * (normal / np.linalg.norm(normal))

        calib[key] = P_prime

    pos_mapping = {0: (0, 1),
               1: (0, 0),
               2: (1, 0),
               3: (1, 1),
               4: (0.5, 0.5),}

    target_points_2d = list(pos_mapping.values())
    target_points_2d = [np.array(val) for val in target_points_2d]

    mean_error = 0.0
    for i, target_point in enumerate(target_points_2d):
        for j in range(i, len(screen_corners), NUM_POINTS):
            p_point = project_ray_to_plane(centroid, normal, screen_corners[j][0], screen_corners[j][1])
            u_coord1, v_coord1 = point_to_uv_coordinates_multi(calib,  p_point)
            array_u_v = np.array([u_coord1, v_coord1])
            error = np.linalg.norm(array_u_v - target_point)
            mean_error += error / len(screen_corners)

    if return_calib:
        return calibration_dict
    # return error_close + np.mean(residuals) + mean_error
    return mean_error

print(error_func([0, 0, 0, 0, 0, 0.0], poses, True))
device_to_gun_rot_trans = np.array([0.0, 0, 0, 0, 0, 0])
rot_bounds = [(-5, 5) for _ in range(3)]
# trans_bounds = [(-0.25, 0.25) for _ in range(3)]
trans_bounds = [(-1, 1),  (-1, 1), (-1, 1)]
dif_e_bounds = rot_bounds + trans_bounds


res = differential_evolution(error_func, args=(poses,), bounds=dif_e_bounds, workers=1, maxiter=500, disp=True, tol=0.000000001)

#%%


print(error_func(res.x, poses, True))
calib = error_func(res.x, poses, False, True)
#%%

from scipy.linalg import svd

plane_points = np.stack([calib['bottom_left'],
                         calib['top_left'],
                         calib['top_right'],
                         calib['bottom_right'],
                         calib['center']])

# Step 2: Compute the centroid
centroid = plane_points.mean(axis=0)

# Step 3: Apply PCA to find the normal of the plane
# Center the points at the origin
points_centered = plane_points - centroid
# Compute SVD
U, S, Vt = svd(points_centered, full_matrices=False)
# The normal vector of the plane is the last column of Vt
normal = Vt[-1, :]
residuals = np.abs(points_centered.dot(normal))

print(normal)
print(residuals)


for key, val in list(calib.items()):

    # Calculate d using the plane equation ax + by + cz + d = 0
    d = -np.dot(normal, centroid)

    # Calculate the distance D from the point to the plane
    D = (np.dot(normal, val) + d) / np.linalg.norm(normal)

    # Project the point onto the plane
    P_prime = val - D * (normal / np.linalg.norm(normal))

    calib[key] = P_prime


NUM_POINTS=5
transform_matrix = np.eye(4)
transform_matrix[:3, :3] = Rotation.from_euler('xyz', res.x[:3], degrees=True).as_matrix()
transform_matrix[:3, 3] = res.x[3:6]
verbose = True

gun_to_world_list = []
for pose_matrix in poses:
    gun_to_world = np.eye(4)
    gun_to_world[:3, :3] = pose_matrix[:3,:3] @ transform_matrix[:3, :3]
    gun_to_world[:3, 3] = (pose_matrix[:3,:3] @ transform_matrix[:3, 3]) + pose_matrix[:3,3]
    gun_to_world_list.append(gun_to_world)

screen_corners = [pose_to_pos_ori(pose) for pose in gun_to_world_list]

for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center']):
        print(corner)
        corner_points = []
        for read, j in enumerate(range(i, len(screen_corners), NUM_POINTS)):


            p_point = project_ray_to_plane(centroid, normal, screen_corners[j][0], screen_corners[j][1])
            dist = np.linalg.norm(calib[corner] - p_point)

            print(f'{corner}, {read}: {dist}')
            
#%%
            
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


u_coord1, v_coord1 = point_to_uv_coordinates_multi(calib,  calib['center'])
u_coord1, v_coord1 = point_to_uv_coordinates(calib['top_right'] - calib['top_left'], calib['bottom_left'] - calib['top_left'], calib['top_left'],  calib['center'])

#%%

pos_mapping = {0: (0, 1),
               1: (0, 0),
               2: (1, 0),
               3: (1, 1),
               4: (0.5, 0.5),}

target_points_2d = list(pos_mapping.values())
target_points_2d = [np.array(val) for val in target_points_2d]

mean_error = 0.0
for i, target_point in enumerate(target_points_2d):
    for j in range(i, len(screen_corners), NUM_POINTS):
        p_point = project_ray_to_plane(centroid, normal, screen_corners[j][0], screen_corners[j][1])
        u_coord1, v_coord1 = point_to_uv_coordinates_multi(calib,  p_point)
        array_u_v = np.array([u_coord1, v_coord1])
        error = np.linalg.norm(array_u_v - target_point)
        mean_error += error / len(screen_corners)
        print(f'target {i} p {j}: {error}')
print(f'Mean error: {mean_error}')

#%%
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from functools import partial


error_func_p = partial(error_func, poses=poses)


varbound=np.array([[-5,5], [-5,5], [-5,5], [-1, 1], [-1, 1], [-1, 1]])
# varbound=np.array([[-0.02, 0.02], [-0.25, -0.2], [0.02, 0.1]])

model=ga(function=error_func_p,dimension=6,variable_type='real',variable_boundaries=varbound, algorithm_parameters={'max_num_iteration': 1500, 'population_size': 100,'mutation_probability': 0.1,'elit_ratio': 0.01,'crossover_probability': 0.5,'parents_portion': 0.3,'crossover_type': 'uniform','max_iteration_without_improv': None})

model.run()