#%%

import triad_openvr
import numpy as np
import time

# Triad Init
v = triad_openvr.triad_openvr()
v.print_discovered_objects()


# pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
# pose_matrix = np.asarray(pose_matrix)
while True:
    time.sleep(0.1)
    print(eval(str(v.devices["tracker_1"].get_pose_matrix())))

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



for i, corner in enumerate(['bottom_left', 'top_left', 'top_right', 'bottom_right', 'center']):
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
        for ray1, ray2 in point_comb:
            p1, p2 = closest_points_on_rays(ray1[0], ray1[1], ray2[0], ray2[1])
            mean_point = (p1 + p2) / 2
            comb_points += [mean_point]
            comb_points_dist += [np.linalg.norm(p1-p2)]

        # print(f'{corner}: mean dist {np.mean(comb_points_dist)} max dist: {np.max(comb_points_dist)}')

        print(f'{corner}: {comb_points}')
        calibration_dict[corner] = np.median(comb_points, axis=0)

        for corner_point in corner_points:
            calibration_dict['error'] += point_to_ray_distance(calibration_dict[corner], corner_point[0], corner_point[1]) / len(corner_points)