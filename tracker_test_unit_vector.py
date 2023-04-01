import triad_openvr
import time
import sys
import math
import numpy as np

v = triad_openvr.triad_openvr()
v.print_discovered_objects()


# sqrt(x**2 + y**2 + z**2) == 1

if len(sys.argv) == 1:
    interval = 1/250
elif len(sys.argv) == 2:
    interval = 1/float(sys.argv[1])
else:
    print("Invalid number of arguments")
    interval = False
    
if interval:
    while(True):
        start = time.time()
        txt = ""
        pose = v.devices["tracker_1"].get_pose_euler()
        pose_matrix = eval(str(v.devices["tracker_1"].get_pose_matrix()))
        pose_matrix = np.asarray(pose_matrix)
        pose_matrix[:, -1] = 0.0
        print(type(pose_matrix))
        print(np.asarray(pose_matrix).shape)
        print(np.asarray(pose_matrix))
        dir_vector = pose_matrix @ np.asarray([[0], [0], [-1.0], [1.0]])
        # print(np.asarray(pose_matrix) @ np.asarray([[0], [0], [1.0], [1.0]]))
        # dir_vector[2, 0] = np.sqrt(- dir_vector[0, 0]**2 - dir_vector[1, 0]**2 + 1)

        # dir_vector /= np.linalg.norm(dir_vector)
        print(dir_vector)
        sin_yaw = -math.sin(math.radians(pose[4]))
        cos_yaw = -math.cos(math.radians(pose[4]))
        pitch = 180 - pose[5] if pose[5] >= 0 else -(180 - (-pose[5]))
        sin_pitch = math.sin(math.radians(pitch))
        cos_pitch = math.cos(math.radians(pitch))

        x = sin_yaw
        y = -(sin_pitch * cos_yaw)
        z = -(cos_pitch * cos_yaw)

        unit_vector = [x, y, z]
        # print("\r" + str(pose), end="")

        # for each in unit_vector:
        #     txt += "%.4f" % each
        #     txt += " "
        # print("\r" + txt, end="")

        for each in pose:
            txt += "%.4f" % each
            txt += " "
        print("\r" + txt, end="")
        sleep_time = interval-(time.time()-start)
        if sleep_time>0:
            time.sleep(sleep_time)