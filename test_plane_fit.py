import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

from itertools import combinations
from scipy.spatial.transform import Rotation

from scipy.optimize import least_squares, differential_evolution


POINTS_NAMES = ['bottom_left', 
                'top_left', 
                'top_right', 
                'bottom_right', 
                'center', 
                'left_center',
                'right_center',
                'bottom_center']

NUM_POINTS = len(POINTS_NAMES)

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

def calculate_pos_and_dir_from_pose(pose_matrix):
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = np.copy(pose_matrix[2, :3])
    tracker_direction[2] = -tracker_direction[2]

    return (tracker_position, tracker_direction)

def calculate_plane_points(screen_corners):

    Ps = []
    dists = []

    for i, corner in enumerate(POINTS_NAMES):
        corner_points = []
        for read, j in enumerate(range(i, len(screen_corners), NUM_POINTS)):
            corner_points.append(screen_corners[j])

        point_comb = list(combinations(corner_points, 2))

        comb_points = []
        comb_points_dist = []
        for ray1, ray2 in point_comb:
            p1, p2 = closest_points_on_rays(ray1[0], ray1[1], ray2[0], ray2[1])
            mean_point = (p1 + p2) / 2
            comb_points += [mean_point]
            comb_points_dist += [np.linalg.norm(p1-p2)]

        Ps.append(np.median(comb_points, axis=0))
        dists.append(np.median(comb_points_dist))

    return np.stack(Ps), np.array(dists)

def error_function(transform_to_aim, pose_matrices, pose_matrices_test, print_results=False, save_params=False):

    print(transform_to_aim)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = Rotation.from_euler('xyz', transform_to_aim[:3], degrees=True).as_matrix()
    transform_matrix[:3, 3] = transform_to_aim[3:6]

    if save_params:
        np.save("transform_to_aim.npy", transform_matrix)

    gun_to_world_list = []
    for pose_matrix in pose_matrices:
        gun_to_world = np.eye(4)
        gun_to_world = pose_matrix @ transform_matrix
        gun_to_world_list.append(gun_to_world)

    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix) for pose_matrix in gun_to_world_list]
    P, dists = calculate_plane_points(pos_dir_points)

    # Replace these with your actual points (P1, P2, P3, P4, Center)
    # Ensure that the points are ordered according to your mapping
    P1 = P[0]
    P2 = P[1]
    P3 = P[2]
    P4 = P[3]
    Center = P[4]

    P5 = P[5]
    P6 = P[6]
    P7 = P[7]

    # Additional edge points
    # Each entry represents an edge point with:
    # - 'P': observed point coordinates [x, y, z]
    # - 'edge': tuple indicating the edge ('R2', 'R3') means from R2 to R3
    # - 's': parameter along the edge, ranging from 0 to 1
    edge_points = [
        {'P': P5, 'edge': ('R1', 'R2'), 's': 0.5},  # Left-center point
        {'P': P6, 'edge': ('R3', 'R4'), 's': 0.5},  # Right-center point
        {'P': P7, 'edge': ('R1', 'R4'), 's': 0.5},  # bottom-center point
    ]

    # Create numpy arrays of your points
    P = np.array([P1, P2, P3, P4])

    # Initial guess for parameters
    # Center C is the centroid of the points
    C0 = np.mean(P, axis=0)

    # Initial guess for u and v: Use PCA components
    P_centered = P - C0
    U, S, Vt = np.linalg.svd(P_centered)
    u0 = Vt[0]
    v0 = Vt[1]

    # Ensure u0 and v0 are orthogonal and normalized
    u0 /= np.linalg.norm(u0)
    v0 -= np.dot(v0, u0) * u0  # Make v0 orthogonal to u0
    v0 /= np.linalg.norm(v0)

    # Initial guess for half-lengths
    l_u0 = np.linalg.norm(P_centered @ u0, ord=np.inf)
    l_v0 = np.linalg.norm(P_centered @ v0, ord=np.inf)

    # Combine all parameters into a single vector
    # Parameters: [C (3), u (3), v (3), l_u, l_v]
    x0 = np.hstack((C0, u0, v0, l_u0, l_v0))

    # Define the objective function
    def objective(x):
        C = x[0:3]
        u = x[3:6]
        v = x[6:9]
        l_u = x[9]
        l_v = x[10]
        
        # Reconstruct rectangle corners
        R = np.array([
            C - l_u * u - l_v * v,  # R1: Bottom Left
            C - l_u * u + l_v * v,  # R2: Top Left
            C + l_u * u + l_v * v,  # R3: Top Right
            C + l_u * u - l_v * v   # R4: Bottom Right
        ])
        R_dict = {'R1': R[0], 'R2': R[1], 'R3': R[2], 'R4': R[3]}
        
        # Sum of squared distances between P_i and R_i
        error_points = np.sum(np.linalg.norm(P - R, axis=1)**2)
        
        # Add the squared distance between optimized center and provided Center
        error_center = np.linalg.norm(C - Center)**2
        
            # Sum of squared distances for edge points
        error_edge_points = 0
        for edge_point in edge_points:
            P_i = np.array(edge_point['P'])
            edge_start_label, edge_end_label = edge_point['edge']
            s_i = edge_point['s']
            
            R_start = R_dict[edge_start_label]
            R_end = R_dict[edge_end_label]
            
            # Expected position on the edge
            E_i = (1 - s_i) * R_start + s_i * R_end
            
            # Add squared distance to total error
            error_edge_points += np.linalg.norm(P_i - E_i)**2
        
        # Total error
        total_error = error_points + error_center + error_edge_points
        return total_error

    # Define constraints
    def constraint_unit_vector_u(x):
        u = x[3:6]
        return np.dot(u, u) - 1

    def constraint_unit_vector_v(x):
        v = x[6:9]
        return np.dot(v, v) - 1

    def constraint_orthogonality(x):
        u = x[3:6]
        v = x[6:9]
        return np.dot(u, v)

    # Collect constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_unit_vector_u},
        {'type': 'eq', 'fun': constraint_unit_vector_v},
        {'type': 'eq', 'fun': constraint_orthogonality},
    ]

    # Bounds for lengths to ensure they are positive
    bounds = [(None, None)] * 9 + [(0, None), (0, None)]

    # Perform optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'disp': True}
    )

    if not result.success:
        return 100000
        raise ValueError("Optimization failed:", result.message)

    if save_params:
        np.save("rectangle_params.npy", result.x)

    if print_results:
        # Extract optimized parameters
        x_opt = result.x
        C_opt = x_opt[0:3]
        u_opt = x_opt[3:6]
        v_opt = x_opt[6:9]
        l_u_opt = x_opt[9]
        l_v_opt = x_opt[10]

        # Reconstruct rectangle corners
        R_opt = np.array([
            C_opt - l_u_opt * u_opt - l_v_opt * v_opt,  # R1: Bottom Left
            C_opt - l_u_opt * u_opt + l_v_opt * v_opt,  # R2: Top Left
            C_opt + l_u_opt * u_opt + l_v_opt * v_opt,  # R3: Top Right
            C_opt + l_u_opt * u_opt - l_v_opt * v_opt   # R4: Bottom Right
        ])
        R_dict = {'R1': R_opt[0], 'R2': R_opt[1], 'R3': R_opt[2], 'R4': R_opt[3]}

        # Compute errors
        errors = np.linalg.norm(P - R_opt, axis=1)
        print("\nErrors between original points and fitted rectangle corners:")
        for idx, error in enumerate(errors):
            print(f"Point {idx+1}: Error = {error:.6f}")

        center_error = np.linalg.norm(C_opt - Center)
        print(f"\nError between optimized center and provided Center: {center_error:.6f}")

        # Compute errors for edge points
        print("\nErrors between edge points and expected positions on rectangle edges:")
        for idx, edge_point in enumerate(edge_points):
            P_i = np.array(edge_point['P'])
            edge_start_label, edge_end_label = edge_point['edge']
            s_i = edge_point['s']
            
            R_start = R_opt[['R1', 'R2', 'R3', 'R4'].index(edge_start_label)]
            R_end = R_opt[['R1', 'R2', 'R3', 'R4'].index(edge_end_label)]
            
            E_i = (1 - s_i) * R_start + s_i * R_end
            error_edge_point = np.linalg.norm(P_i - E_i)
            print(f"Edge Point {idx+1}: Error = {error_edge_point:.6f}")

        # Print optimized parameters
        print("\nOptimized Parameters:")
        print(f"Center C: {C_opt}")
        print(f"Unit vector u: {u_opt}")
        print(f"Unit vector v: {v_opt}")
        print(f"Half-length along u (l_u): {l_u_opt}")
        print(f"Half-length along v (l_v): {l_v_opt}")

    # x_opt = result.x
    # C_opt = x_opt[0:3]
    # u_opt = x_opt[3:6]
    # v_opt = x_opt[6:9]
    # l_u_opt = x_opt[9]
    # l_v_opt = x_opt[10]

    # N_opt = np.cross(u_opt, v_opt)
    # N_opt /= np.linalg.norm(N_opt)

    # errors = []

    # for point_id, name in enumerate(POINTS_NAMES):

    #     pos_map = pos_mapping[point_id]
    #     errors_point = []

    #     for corner in range(point_id, len(pose_matrices_test), NUM_POINTS):

    #         aim_pose = pose_matrices_test[corner] @ transform_matrix

    #         P0, D = calculate_pos_and_dir_from_pose(aim_pose)

    #         denom = np.dot(D, N_opt)
    #         numerator = np.dot(C_opt - P0, N_opt)
    #         t = numerator / denom
    #         P_intersect = P0 + t * D

    #         V = P_intersect - C_opt
    #         a = np.dot(V, u_opt)
    #         b = np.dot(V, v_opt)

    #         u_coord = (a + l_u_opt) / (2 * l_u_opt)
    #         v_coord = (l_v_opt - b) / (2 * l_v_opt)

    #         error = np.linalg.norm(np.array(pos_map) - np.array([u_coord, v_coord]))
    #         errors_point.append(error)

    #     errors.append(np.mean(errors_point))

    # print(np.mean(errors))

    # return np.mean(np.array(errors)**2)
    # return result.fun + np.sum(transform_to_aim[:3]**2) + np.sum(np.abs(transform_to_aim[3:])) + np.sum(dists**2)
    return result.fun + np.sum(dists**2)


pos_mapping = {0: (0, 1),
               1: (0, 0),
               2: (1, 0),
               3: (1, 1),
               4: (0.5, 0.5),
               5: (0.0, 0.5),
               6: (1.0, 0.5),
               7: (0.5, 1.0),}

bounds_x = 20
bounds_y = 30
bounds_z = 250
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

pose_matrices = []
for i in range(48, 64):
    pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
    pose_matrices.append(pose_matrix)

# for i in range(48, 72):
#     pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
#     pose_matrices.append(pose_matrix)

# pose_matrices_test = []
# for i in range(16, 48):
#     pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
#     pose_matrices_test.append(pose_matrix)

# for i in range(64, 72):
#     pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
#     pose_matrices_test.append(pose_matrix)

pose_matrices_test = []
for i in range(48, 64):
    pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
    pose_matrices_test.append(pose_matrix)

# for i in range(64, 72):
#     pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\pose_matrix_{i}.npy')
#     pose_matrices_test.append(pose_matrix)

device_to_gun_rot = np.array([0.0, 0.0, 0.])
device_to_gun_trans = np.array([0.0, 0.0, 0.])
device_to_gun_rot_trans = np.concatenate((device_to_gun_rot, device_to_gun_trans))

error = error_function(device_to_gun_rot_trans, pose_matrices, pose_matrices_test, True, True)
print(error)
transform_to_aim = np.load("transform_to_aim.npy")
x_opt = np.load("rectangle_params.npy")
    
C_opt = x_opt[0:3]
u_opt = x_opt[3:6]
v_opt = x_opt[6:9]
l_u_opt = x_opt[9]
l_v_opt = x_opt[10]

N_opt = np.cross(u_opt, v_opt)
N_opt /= np.linalg.norm(N_opt)

errors = []

for point_id, name in enumerate(POINTS_NAMES):

    print(name)

    pos_map = pos_mapping[point_id]
    errors_point = []

    for corner in range(point_id, len(pose_matrices_test), NUM_POINTS):

        aim_pose = pose_matrices_test[corner] @ transform_to_aim

        P0, D = calculate_pos_and_dir_from_pose(aim_pose)

        denom = np.dot(D, N_opt)
        numerator = np.dot(C_opt - P0, N_opt)
        t = numerator / denom
        P_intersect = P0 + t * D

        V = P_intersect - C_opt
        a = np.dot(V, u_opt)
        b = np.dot(V, v_opt)

        within_u = (-l_u_opt <= a <= l_u_opt)
        within_v = (-l_v_opt <= b <= l_v_opt)

        u_coord = (a + l_u_opt) / (2 * l_u_opt)
        v_coord = (l_v_opt - b) / (2 * l_v_opt)

        print(u_coord, v_coord)

        error = np.linalg.norm(np.array(pos_map) - np.array([u_coord, v_coord]))
        errors_point.append(error)

    errors.append(np.mean(errors_point))
    print(np.mean(errors_point))

print(np.mean(errors))

import sys
sys.exit(0)

res = least_squares(error_function, device_to_gun_rot_trans, args=(pose_matrices, pose_matrices_test), ftol=1e-10, xtol=1e-10, gtol=1e-10)
print(res)
print(res.x)

error = error_function(res.x, pose_matrices, pose_matrices_test, True, True)
print(error)

transform_to_aim = np.load("transform_to_aim.npy")
x_opt = np.load("rectangle_params.npy")
    
C_opt = x_opt[0:3]
u_opt = x_opt[3:6]
v_opt = x_opt[6:9]
l_u_opt = x_opt[9]
l_v_opt = x_opt[10]

N_opt = np.cross(u_opt, v_opt)
N_opt /= np.linalg.norm(N_opt)

errors = []

for point_id, name in enumerate(POINTS_NAMES):

    print(name)

    pos_map = pos_mapping[point_id]
    errors_point = []

    for corner in range(point_id, len(pose_matrices_test), NUM_POINTS):

        aim_pose = pose_matrices_test[corner] @ transform_to_aim

        P0, D = calculate_pos_and_dir_from_pose(aim_pose)

        denom = np.dot(D, N_opt)
        # if np.isclose(denom, 0):
        #     print("The ray is parallel to the plane; no intersection.")
        # else:
        numerator = np.dot(C_opt - P0, N_opt)
        t = numerator / denom
        P_intersect = P0 + t * D

        V = P_intersect - C_opt
        a = np.dot(V, u_opt)
        b = np.dot(V, v_opt)

        u_coord = (a + l_u_opt) / (2 * l_u_opt)
        v_coord = (l_v_opt - b) / (2 * l_v_opt)

        print(u_coord, v_coord)

        error = np.linalg.norm(np.array(pos_map) - np.array([u_coord, v_coord]))
        errors_point.append(error)

    errors.append(np.mean(errors_point))
    print(np.mean(errors_point))

print(np.mean(errors))

# import sys
# sys.exit(0)

# res = differential_evolution(error_function, args=(pose_matrices,), workers=1, maxiter=100, disp=True, tol=0.000000001, bounds=dif_e_bounds)

# print(res)
# print(res.x)

# P1 = [-305.44291389,  392.97825836,  -13.45232624]
# P2 = [-296.40910023,   57.86076498,  -16.48643333]
# P3 = [326.76011871,  52.01827777, -38.34075176]
# P4 = [322.04597955, 368.64359904,  -1.24809361]
# Center = [17.15223776, 216.23642922, -12.48120527]

# # Create a 5x3 numpy array of your points
# P = np.array([P1, P2, P3, P4, Center])

# # Step 1: Compute centroid using all points
# C = np.mean(P, axis=0)

# # Step 2: Center the points
# P_centered = P - C

# # Step 3: Fit the plane using PCA
# pca = PCA(n_components=2)
# pca.fit(P_centered)

# # Basis vectors in the plane
# u = pca.components_[0]
# v = pca.components_[1]

# # Normal vector of the plane
# N = np.cross(u, v)
# N /= np.linalg.norm(N)  # Normalize the normal vector

# # Step 4: Project points onto the plane and compute 2D coordinates
# x = P_centered @ u
# y = P_centered @ v

# # Step 5: Perform 2D PCA on the projected points to align with the rectangle axes
# points_2d = np.column_stack((x, y))
# pca_2d = PCA(n_components=2)
# pca_2d.fit(points_2d)

# # Rotate points to align with principal axes
# rotated_points = pca_2d.transform(points_2d)

# # Step 6: Find the rectangle bounds in the rotated coordinate system
# x_min, y_min = np.min(rotated_points, axis=0)
# x_max, y_max = np.max(rotated_points, axis=0)

# Define rectangle corners in rotated coordinate system based on your mapping
# rectangle_corners_rotated = np.array([
#     [x_max, y_min],  # p3: Top Right (max_x, min_y)
#     [x_max, y_max],   # p4: Bottom Right (max_x, max_y)
#     [x_min, y_max],  # p1: Bottom Left (min_x, max_y)
#     [x_min, y_min],  # p2: Top Left (min_x, min_y)

# ])

# # Define rectangle corners in rotated coordinate system based on your mapping
# rectangle_corners_rotated = np.array([
#     [x_min, y_max],  # p1: Bottom Left (min_x, max_y)
#     [x_min, y_min],  # p2: Top Left (min_x, min_y)
#     [x_max, y_min],  # p3: Top Right (max_x, min_y)
#     [x_max, y_max]   # p4: Bottom Right (max_x, max_y)
# ])

# # Map the rectangle corners back to original 2D coordinates
# rectangle_corners_2d = pca_2d.inverse_transform(rectangle_corners_rotated)

# # Map the rectangle corners back to 3D space
# rectangle_corners_3d = C + rectangle_corners_2d @ np.array([u, v])

# # Assign each original point to its corresponding rectangle corner
# # Assuming the order of P1 to P4 corresponds to p1 to p4
# adjusted_points_3d = rectangle_corners_3d[:4]

# # For the center point, we can take the centroid of the rectangle corners
# rectangle_center_3d = np.mean(rectangle_corners_3d[:4], axis=0)
# adjusted_points_3d = np.vstack((adjusted_points_3d, rectangle_center_3d))

# # Compute the errors between the original points and their corresponding rectangle points
# errors = np.linalg.norm(P - adjusted_points_3d, axis=1)

# # Print the errors
# print("\nErrors between original points and corresponding rectangle corners:")
# for idx, error in enumerate(errors):
#     print(f"Point {idx+1}: Error = {error:.6f}")

# # Print the original and adjusted points
# print("\nOriginal vs. Adjusted Points:")
# for idx, (orig_pt, adj_pt) in enumerate(zip(P, adjusted_points_3d)):
#     print(f"Point {idx+1} Original: {orig_pt}, Adjusted: {adj_pt}")

# # Visualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Original points
# ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='blue', label='Original Points')

# # Adjusted points on rectangle
# ax.scatter(adjusted_points_3d[:, 0], adjusted_points_3d[:, 1], adjusted_points_3d[:, 2],
#            color='orange', label='Corresponding Rectangle Points')

# # Draw lines between original points and adjusted points to visualize errors
# for orig_pt, adj_pt in zip(P, adjusted_points_3d):
#     ax.plot([orig_pt[0], adj_pt[0]], [orig_pt[1], adj_pt[1]], [orig_pt[2], adj_pt[2]],
#             color='gray', linestyle='--')

# # Plot rectangle edges
# X_rect = np.append(rectangle_corners_3d[:, 0], rectangle_corners_3d[0, 0])
# Y_rect = np.append(rectangle_corners_3d[:, 1], rectangle_corners_3d[0, 1])
# Z_rect = np.append(rectangle_corners_3d[:, 2], rectangle_corners_3d[0, 2])

# ax.plot(X_rect, Y_rect, Z_rect, color='red', label='Best-fit Rectangle')

# # Plot centroid
# ax.scatter(C[0], C[1], C[2], color='green', marker='^', s=100, label='Computed Centroid')

# ax.legend()
# plt.show()