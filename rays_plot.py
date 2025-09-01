import numpy as np
import plotly.graph_objects as go

# Sample list of rays: each ray is a tuple (starting point, direction vector)
def calculate_pos_and_dir_from_pose(pose_matrix):
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = -np.copy(pose_matrix[:3, 2])
    # tracker_direction[2] = -tracker_direction[2]

    return (tracker_position, tracker_direction)

tracker_to_gun = np.load("tracker_to_gun.npy")
print(tracker_to_gun)
# Example data: list of pose matrices P_i
rays = []

for i in range(0, 42):
    pose_matrix = np.load(f'calib_shots_1/shoot_{i}.npy')
    rays.append(calculate_pos_and_dir_from_pose(tracker_to_gun @ pose_matrix))
    # rays.append(calculate_pos_and_dir_from_pose(pose_matrix))


# rays = [
#     ((0, 0, 0), (1, 1, 1)),
#     ((1, 0, 0), (-1, 1, 0)),
#     ((0, 1, 0), (0, -1, 1))
# ]

# Create an empty figure
fig = go.Figure()

# For each ray, add a line trace
for start_point, direction in rays:
    x0, y0, z0 = start_point
    dx, dy, dz = direction

    # Define the length of the ray for plotting purposes
    t_max = 1200  # Adjust as needed
    t = np.linspace(0, t_max, 200)

    # Calculate the ray coordinates
    x = x0 + dx * t
    y = y0 + dy * t
    z = z0 + dz * t

    # Add the ray to the figure
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=4),
        name=f'Ray from'
    ))

    # Optionally, plot the starting point
    fig.add_trace(go.Scatter3d(
        x=[x0], y=[y0], z=[z0],
        mode='markers',
        marker=dict(size=5, color='red'),
        showlegend=False
    ))

fig.add_trace(go.Scatter3d(
        x=[81.91], y=[34.69], z=[620.79],
        mode='markers',
        marker=dict(size=5, color='red'),
        showlegend=False
    ))

fig.add_trace(go.Scatter3d(
        x=[-70.28], y=[151.78], z=[729.83],
        mode='markers',
        marker=dict(size=5, color='red'),
        showlegend=False
    ))


# X = np.array([122.28, 222.32, 857.25])

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
        aspectmode='data'  # Ensures equal scaling
    ),
    title='3D Rays Visualization',
    width=1920,
    height=1080
)

# Show the interactive plot
fig.show()