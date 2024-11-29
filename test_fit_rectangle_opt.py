import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Replace these with your actual points (P1, P2, P3, P4, Center)
# Ensure that the points are ordered according to your mapping
P1 = [-305.44291389,  392.97825836,  -13.45232624]
P2 = [-296.40910023,   57.86076498,  -16.48643333]
P3 = [326.76011871,  52.01827777, -38.34075176]
P4 = [322.04597955, 368.64359904,  -1.24809361]
Center = [17.15223776, 216.23642922, -12.48120527]

P5 = 0.5 * np.array(P1) + 0.5 * np.array(P2)
P6 = 0.5 * np.array(P2) + 0.5 * np.array(P3)
P7 = 0.5 * np.array(P3) + 0.5 * np.array(P4)
P8 = 0.5 * np.array(P1) + 0.5 * np.array(P4)

# Additional edge points
# Each entry represents an edge point with:
# - 'P': observed point coordinates [x, y, z]
# - 'edge': tuple indicating the edge ('R2', 'R3') means from R2 to R3
# - 's': parameter along the edge, ranging from 0 to 1
edge_points = [
    {'P': P5, 'edge': ('R1', 'R2'), 's': 0.5},  # Left-center point
    {'P': P6, 'edge': ('R2', 'R3'), 's': 0.5},  # Top-center point
    {'P': P7, 'edge': ('R3', 'R4'), 's': 0.5},  # Right-center point
    {'P': P8, 'edge': ('R1', 'R4'), 's': 0.5},  # bottom-center point
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
    raise ValueError("Optimization failed:", result.message)

print(result)

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

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original points
ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='blue', label='Original Points')

# Fitted rectangle corners
ax.scatter(R_opt[:, 0], R_opt[:, 1], R_opt[:, 2], color='orange', label='Fitted Rectangle Corners')

# Draw lines between original points and fitted rectangle corners
for orig_pt, rect_pt in zip(P, R_opt):
    ax.plot([orig_pt[0], rect_pt[0]], [orig_pt[1], rect_pt[1]], [orig_pt[2], rect_pt[2]],
            color='gray', linestyle='--')

# Draw rectangle edges
X_rect = np.append(R_opt[:, 0], R_opt[0, 0])
Y_rect = np.append(R_opt[:, 1], R_opt[0, 1])
Z_rect = np.append(R_opt[:, 2], R_opt[0, 2])

ax.plot(X_rect, Y_rect, Z_rect, color='red', label='Fitted Rectangle')

# Plot optimized center
ax.scatter(C_opt[0], C_opt[1], C_opt[2], color='green', marker='^', s=100, label='Optimized Center')
ax.scatter(Center[0], Center[1], Center[2], color='purple', marker='x', s=100, label='Provided Center')

# Plot edge points and their expected positions
for edge_point in edge_points:
    P_i = np.array(edge_point['P'])
    edge_start_label, edge_end_label = edge_point['edge']
    s_i = edge_point['s']
    
    R_start = R_dict[edge_start_label]
    R_end = R_dict[edge_end_label]
    
    E_i = (1 - s_i) * R_start + s_i * R_end
    
    # Plot observed edge point
    ax.scatter(P_i[0], P_i[1], P_i[2], color='cyan', marker='o', label='Edge Point')
    
    # Plot expected position on edge
    ax.scatter(E_i[0], E_i[1], E_i[2], color='magenta', marker='x', label='Expected Edge Position')
    
    # Draw line between observed point and expected position
    ax.plot([P_i[0], E_i[0]], [P_i[1], E_i[1]], [P_i[2], E_i[2]], color='gray', linestyle='--')

ax.legend()
plt.show()
