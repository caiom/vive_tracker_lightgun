#%%
import json
import numpy as np

with open("calibration.json", 'r') as f:
    calib = json.load(f)

# Convert calib to numpy arrays
for key, val in calib.items():
    calib[key] = np.asfarray(val)


calib['plane_normal1'] = np.cross(calib['bottom_left'] - calib['top_left'], calib['top_right'] - calib['top_left'])
calib['plane_normal2'] = np.cross(calib['bottom_right'] - calib['bottom_left'], calib['top_left'] - calib['bottom_left'])
calib['plane_normal3'] = np.cross(calib['top_right'] - calib['bottom_right'], calib['bottom_left'] - calib['bottom_right'])
calib['plane_normal4'] = np.cross(calib['bottom_right'] - calib['top_right'], calib['top_left'] - calib['top_right'])