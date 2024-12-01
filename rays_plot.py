import numpy as np
import plotly.graph_objects as go

# Sample list of rays: each ray is a tuple (starting point, direction vector)
def calculate_pos_and_dir_from_pose(pose_matrix):
    tracker_position = np.copy(pose_matrix[:3, -1])
    tracker_direction = np.copy(pose_matrix[2, :3])
    tracker_direction[2] = -tracker_direction[2]

    return (tracker_position, tracker_direction)

i = 0
read = 0
rays = []
while i < 16:
    print(f"Reading {i}")
    pose_matrix = np.load(f'pose_matrix_{i}.npy')
    rays.append(calculate_pos_and_dir_from_pose(pose_matrix))
    read += 1
    i += 1

    if read == 5:
        i += 3
        read = 0
        continue


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
    t_max = 2000  # Adjust as needed
    t = np.linspace(0, t_max, 100)

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

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis',
        aspectmode='data'  # Ensures equal scaling
    ),
    title='3D Rays Visualization',
    width=800,
    height=600
)

# Show the interactive plot
fig.show()