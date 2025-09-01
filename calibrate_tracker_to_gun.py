import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# Example data: list of pose matrices P_i
pose_matrices = []

for i in range(42):
    pose_matrix = np.load(f'calib_shots_1/shoot_{i}.npy')
    pose_matrices.append((pose_matrix, np.array([81.91, 34.69, 620.79])))


# Example data: list of pose matrices P_i
for i in range(42, 81):
    pose_matrix = np.load(f'calib_shots_2/shoot_{i}.npy')
    pose_matrices.append((pose_matrix, np.array([-70.28, 151.78, 729.83])))

# Define the aiming direction in the gun's local frame (e.g., +Z axis)
aim_dir_local = np.array([0, 0, -1])

def residuals(params, pose_matrices):
    # Extract rotation (as rotation vector) and translation
    rot_vec = params[0:3]
    t = params[3:6]
    
    rotation = R.from_rotvec(rot_vec).as_matrix()
    
    res = []
    for P, target in pose_matrices:
        R_i = P[0:3, 0:3]
        t_i = P[0:3, 3]
        
        # Apply transformation T to get gun's pose G_i
        G_R = rotation @ R_i
        G_t = rotation @ t_i + t
        
        # Gun's position
        C_i = G_t
        
        # Gun's aiming direction in world frame
        d_i = G_R @ aim_dir_local
        d_i /= np.linalg.norm(d_i)
        
        # Compute residual: distance from X to the line L_i
        X_diff = target - C_i
        distance = np.linalg.norm(np.cross(X_diff, d_i))  # |(X - C_i) x d_i|
        res.append(distance)

    print(np.mean(res))
    print(np.max(res))
    
    return res

# Initial guess: no rotation, zero translation
bounds_x = 20
bounds_y = 40
bounds_z = 350
bounds = [(-np.inf, -np.inf, -np.inf, -bounds_x, -bounds_y, -bounds_z), (np.inf, np.inf, np.inf, bounds_x, 0, 0)]
initial_params = np.zeros(6)
initial_params[4] -= 35

# Run optimization
print(bounds)
result = least_squares(residuals, initial_params, args=(pose_matrices,), method='trf', bounds=bounds)

# Extract results
rot_vec_opt = result.x[0:3]
t_opt = result.x[3:6]

rotation_opt = R.from_rotvec(rot_vec_opt).as_matrix()


# Create a 4x4 transformation matrix
pose_matrix = np.eye(4)
pose_matrix[:3, :3] = rotation_opt
pose_matrix[:3, 3] = t_opt.flatten()

np.save("tracker_to_gun.npy", pose_matrix)

print("Optimized Rotation:\n", rotation_opt)
print("Optimized Translation:\n", t_opt)
