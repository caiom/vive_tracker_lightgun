import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


P1 = [-669.75576309,  834.55345554,  132.85246536]
P2 = [-648.79838403,  150.8316527,    46.6711027 ]
P3 = [635.86756713, 119.26755294, 121.67477274]
P4 = [598.30857947, 773.00869089, 262.0567923 ]


# Create a 4x3 numpy array of your points
P = np.array([P1, P2, P3, P4])

# Step 1: Compute centroid
C = np.mean(P, axis=0)

# Step 2: Center the points
P_centered = P - C

# PCA to find the best-fit plane
pca = PCA(n_components=2)
pca.fit(P_centered)

# Basis vectors in the plane
u = pca.components_[0]
v = pca.components_[1]

# Recompute normal vector using PCA components
N_pca = np.cross(u, v)
N_pca /= np.linalg.norm(N_pca)

print(N_pca)

# Step 4: Project points onto the plane and compute 2D coordinates
# Since u and v are orthonormal, we can project directly
x = P_centered @ u
y = P_centered @ v

# Step 5: Perform 2D PCA on the projected points to find the principal axes
points_2d = np.column_stack((x, y))
pca_2d = PCA(n_components=2)
pca_2d.fit(points_2d)

# Rotate points to align with principal axes
rotated_points = pca_2d.transform(points_2d)

# Step 6: Find the bounding rectangle in the rotated coordinate system
x_min, y_min = np.min(rotated_points, axis=0)
x_max, y_max = np.max(rotated_points, axis=0)

# Define rectangle corners in rotated coordinate system
rectangle_corners_rotated = np.array([
    [x_min, y_min],
    [x_max, y_min],
    [x_max, y_max],
    [x_min, y_max]
])

# Step 7: Rotate the rectangle corners back to original 2D coordinates
rectangle_corners_2d = pca_2d.inverse_transform(rectangle_corners_rotated)

# Step 8: Map the 2D rectangle corners back to 3D space
rectangle_corners_3d = []
for point in rectangle_corners_2d:
    # point is in the 2D plane coordinates
    # Map back to 3D space
    corner_3d = C + point[0] * u + point[1] * v
    rectangle_corners_3d.append(corner_3d)

rectangle_corners_3d = np.array(rectangle_corners_3d)

# Now rectangle_corners_3d contains the 3D coordinates of the rectangle corners
print("Best-fit rectangle corners in 3D space:")
for idx, corner in enumerate(rectangle_corners_3d):
    print(f"Corner {idx+1}: {corner}")
