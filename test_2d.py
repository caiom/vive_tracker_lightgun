#%%
import numpy as np
from scipy.spatial.transform import Rotation
# Define the angle in degrees and convert it to radians
theta_degrees = 90
theta_radians = np.deg2rad(-theta_degrees)  # negative for clockwise rotation

# Calculate the cosine and sine of the angle
cos_theta = np.cos(theta_radians)
sin_theta = np.sin(theta_radians)

# Construct the rotation matrix for a clockwise rotation
rotation_matrix = np.array([[cos_theta, -sin_theta],
                            [sin_theta, cos_theta]])

t = np.array([-6, 7]).T


transform_matrix = np.eye(3)
transform_matrix[:2, :2] = rotation_matrix
transform_matrix[:2, 2] = t

t2 = transform_matrix.copy()

sensor_rot = np.eye(2)
s_p = np.array([1, 0]).T
base_point = np.array([0, 0]).T
sensor_rot@s_p