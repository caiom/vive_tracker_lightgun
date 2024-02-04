#%%
import pysurvive
import sys
import time

actx = pysurvive.SimpleContext(sys.argv)

for obj in actx.Objects():
    print(obj.Name())

while actx.Running():
    updated = actx.NextUpdated()
    time.sleep(0.005)
    if updated:
        pose = updated.Pose()[0]
        print(updated.Name(), pose.Pos[0], pose.Pos[1], pose.Pos[2])


#%%
import sys
import pysurvive
import numpy as np
import time
from scipy.spatial.transform import Rotation

ctx = pysurvive.init(sys.argv)

if ctx is None: # implies -help or similiar
    exit(-1)

def button_func(obj, eventtype, buttonid, axisids, axisvals):
    if eventtype == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_DOWN:
        eventstring = "DOWN"
    elif eventtype == pysurvive.SURVIVE_INPUT_EVENT_BUTTON_UP:
        eventstring = "UP"
    else:
        eventstring = "%d" % (eventtype)
    print("Button %d on %s generated event %s"%(buttonid, obj.contents.codename.decode('utf8'), eventstring))

def pose_func(ctx, timecode, pose):
    # print(pose[:3])
    rot = Rotation.from_quat(pose[3:])
    # print(rot.as_euler('xyz', degrees=True))
    # Rotation.as_matrix()
    pose_matrix = np.zeros((3, 4))
    pose_matrix[:, :3] = rot.as_matrix()
    pose_matrix[:, 3] = np.array(pose[:3])
    orientation = np.array([-pose_matrix[0, 2], pose_matrix[0, 1], -pose_matrix[0, 0]])
    print(f'Pos: {pose_matrix[:, 3]} - Orientation: {orientation}')
    # print('\n\n')

keepRunning = True

pysurvive.install_button_fn(ctx, button_func)
pysurvive.install_pose_fn(ctx, pose_func)
while keepRunning and pysurvive.poll(ctx) == 0:
    # print('oi')
    # time.sleep(0.01)
    pass

pysurvive.close(ctx)


#%%
import numpy as np
p = np.load('pose_matrix_34.npy')