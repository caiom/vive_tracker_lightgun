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

# POINTS_NAMES = ['bottom_left', 
#                 'top_left', 
#                 'top_right', 
#                 'bottom_right', 
#                 'center']

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
    tracker_direction = -np.copy(pose_matrix[:3, 2])

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
        print(comb_points_dist)

    return np.stack(Ps), np.array(dists)

def error_function(pose_matrices, pose_matrices_test, print_results=False, save_params=False):

    pos_dir_points = [calculate_pos_and_dir_from_pose(pose_matrix.copy()) for pose_matrix in pose_matrices]
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
    print(Center)

    print(P)

    # Initial guess for parameters
    # Center C is the centroid of the points
    C0 = np.mean(P, axis=0)
    C0 = Center

    print(C0)

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

    print(l_v0)

    # Combine all parameters into a single vector
    # Parameters: [C (3), u (3), v (3), l_u, l_v]
    x0 = np.hstack((C0, u0, v0, l_u0, l_v0))
    # x0 = np.hstack((u0, v0, l_u0, l_v0))

    # Define the objective function
    def objective(x):
        C = x[0:3]
        u = x[3:6]
        v = x[6:9]
        l_u = x[9]
        l_v = x[10]


        # C = C0
        # u = x[0:3]
        # v = x[3:6]
        # l_u = x[6]
        # l_v = x[7]

        N = np.cross(u, v)
        N /= np.linalg.norm(N)

        errors = []

        for point_id, name in enumerate(POINTS_NAMES):

            # print(name)

            pos_map = pos_mapping[point_id]
            errors_point = []

            for corner in range(point_id, len(pose_matrices_test), NUM_POINTS):

                aim_pose = pose_matrices_test[corner]

                P0, D = calculate_pos_and_dir_from_pose(aim_pose)

                denom = np.dot(D, N)
                numerator = np.dot(C - P0, N)
                t = numerator / denom
                P_intersect = P0 + t * D

                V = P_intersect - C
                a = np.dot(V, u)
                b = np.dot(V, v)

                within_u = (-l_u <= a <= l_u)
                within_v = (-l_v <= b <= l_v)

                u_coord = (a + l_u) / (2 * l_u)
                # v_coord = (l_v - b) / (2 * l_v)
                v_coord = (b + l_v) / (2 * l_v)

                # print(u_coord, v_coord)

                error = np.linalg.norm(np.array(pos_map) - np.array([u_coord, v_coord]))
                errors_point.append(error)

            # print(np.mean(errors_point))
            errors.append(np.mean(errors_point))
        print(np.mean(errors))
        # import time
        # time.sleep(3)
        return np.mean(errors)
        
        # # Reconstruct rectangle corners
        # R = np.array([
        #     C - l_u * u - l_v * v,  # R1: Bottom Left
        #     C - l_u * u + l_v * v,  # R2: Top Left
        #     C + l_u * u + l_v * v,  # R3: Top Right
        #     C + l_u * u - l_v * v   # R4: Bottom Right
        # ])
        # R_dict = {'R1': R[0], 'R2': R[1], 'R3': R[2], 'R4': R[3]}
        
        # # Sum of squared distances between P_i and R_i
        # error_points = np.sum(np.linalg.norm(P - R, axis=1)**2)
        
        # # Add the squared distance between optimized center and provided Center
        # error_center = np.linalg.norm(C - Center)**2
        
        # # Sum of squared distances for edge points
        # error_edge_points = 0
        # for edge_point in edge_points:
        #     P_i = np.array(edge_point['P'])
        #     edge_start_label, edge_end_label = edge_point['edge']
        #     s_i = edge_point['s']
            
        #     R_start = R_dict[edge_start_label]
        #     R_end = R_dict[edge_end_label]
            
        #     # Expected position on the edge
        #     E_i = (1 - s_i) * R_start + s_i * R_end
            
        #     # Add squared distance to total error
        #     error_edge_points += np.linalg.norm(P_i - E_i)**2
        
        # # Total error
        # total_error = error_points + error_center + error_edge_points
        # total_error = error_points + error_center
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
    
    # def constraint_unit_vector_u(x):
    #     u = x[0:3]
    #     return np.dot(u, u) - 1

    # def constraint_unit_vector_v(x):
    #     v = x[3:6]
    #     return np.dot(v, v) - 1

    # def constraint_orthogonality(x):
    #     u = x[0:3]
    #     v = x[3:6]
    #     return np.dot(u, v)

    # Collect constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_unit_vector_u},
        {'type': 'eq', 'fun': constraint_unit_vector_v},
        {'type': 'eq', 'fun': constraint_orthogonality},
    ]

    # Bounds for lengths to ensure they are positive
    bounds = [(None, None)] * 9 + [(100, None), (100, None)]
    # bounds = [(None, None)] * 6 + [(100, None), (100, None)]

    # Perform optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'ftol': 1e-9, 'disp': True, "maxiter": 1000}
    )

    if not result.success:
        # return 100000
        raise ValueError("Optimization failed:", result.message)

    if save_params:
        np.save("rectangle_params.npy", result.x)
        # np.save("rectangle_params.npy", np.hstack((C0, result.x)))


    if print_results:
        # Extract optimized parameters
        x_opt = result.x
        C_opt = x_opt[0:3]
        u_opt = x_opt[3:6]
        v_opt = x_opt[6:9]
        l_u_opt = x_opt[9]
        l_v_opt = x_opt[10]


        # x_opt = result.x
        # C_opt = C0
        # u_opt = x_opt[0:3]
        # v_opt = x_opt[3:6]
        # l_u_opt = x_opt[6]
        # l_v_opt = x_opt[7]

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

        # # Compute errors for edge points
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


pos_mapping = {0: (0, 1),
               1: (0, 0),
               2: (1, 0),
               3: (1, 1),
               4: (0.5, 0.5),
               5: (0.0, 0.5),
               6: (1.0, 0.5),
               7: (0.5, 1.0),}

# pos_mapping = {0: (0, 1),
#                1: (0, 0),
#                2: (1, 0),
#                3: (1, 1),
#                4: (0.5, 0.5),}

pose_matrices = []
i = 0
read = 0
while i < 16:
    print(f"Reading {i}")
    pose_matrix = np.load(f'pose_matrix_{i}.npy')
    pose_matrices.append(pose_matrix)
    read += 1
    i += 1

    if read == 5:
        i += 0
        read = 0
        continue



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
# for i in range(0, 24):
#     pose_matrix = np.load(f'pose_matrix_{i}.npy')
#     pose_matrices_test.append(pose_matrix)

i = 0
read = 0
while i < 40:
    pose_matrix =  np.load(f'pose_matrix_{i}.npy')
    pose_matrices_test.append(pose_matrix)
    read += 1
    i += 1

    if read == 5:
        i += 0
        read = 0
        continue


error = error_function(pose_matrices, pose_matrices_test, True, True)
print(error)
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

        aim_pose = pose_matrices_test[corner]

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
        # v_coord = (l_v_opt - b) / (2 * l_v_opt)
        v_coord = (b + l_v_opt) / (2 * l_v_opt)

        print(u_coord, v_coord)

        error = np.linalg.norm(np.array(pos_map) - np.array([u_coord, v_coord]))
        errors_point.append(error)

    errors.append(np.mean(errors_point))
    print(np.mean(errors_point))

print(np.mean(errors))