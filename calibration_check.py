#%%
import numpy as np

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

pose_matrices = []
for i in range(0, 10):
    pose_matrix = np.load(f'C:\\Users\\v3n0w\\Downloads\\Camera\\calibration_poses\\pose_matrix_{i}.npy')
    pose_matrices.append(pose_matrix)


#%%

tracker_pos_dir = []

for pose_matrix in [pose_matrices[4], pose_matrices[9]]:
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = pose_matrix[:3, 2]

    tracker_pos_dir.append((tracker_position, tracker_direction))


ps = closest_points_on_rays(tracker_pos_dir[0][0],
                            tracker_pos_dir[0][1],
                            tracker_pos_dir[1][0],
                            tracker_pos_dir[1][1])

print(ps)




