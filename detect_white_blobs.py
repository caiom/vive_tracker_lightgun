import cv2
import numpy as np
import random

def fit_circle_3pts(p1, p2, p3):
    """
    Fit a circle to three points.
    Returns the center coordinates and radius of the circle.
    """
    # Convert points to numpy arrays
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    
    # Calculate the midpoints of the lines formed by the points
    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2
    
    # Calculate the direction vectors of the lines
    d1 = p2 - p1
    d2 = p3 - p2
    
    # Calculate the perpendicular vectors
    perp1 = np.array([-d1[1], d1[0]])
    perp2 = np.array([-d2[1], d2[0]])
    
    # Set up the system of equations to solve for the intersection point
    A = np.array([perp1, -perp2]).T
    b = mid2 - mid1
    
    # Solve for the intersection point (circle center)
    try:
        t = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Points are colinear or too close together
        return None, None
    
    center = mid1 + t[0] * perp1
    radius = np.linalg.norm(center - p1)
    
    return center, radius

def calculate_center_of_blob(image: np.ndarray):
    """
    Calculate the center of the white blob in a 2D uint8 NumPy array.
    
    Parameters:
    - image (np.ndarray): 2D array of type uint8 representing the image.
    
    Returns:
    - tuple: (x_center, y_center) coordinates of the blob's center.
             Returns None if the image has no white pixels.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    # Convert image to float for precise calculations
    img = image.astype(np.float32)
    
    # Find the maximum pixel value to normalize weights
    max_val = img.max()
    
    if max_val == 0:
        # No white pixels in the image
        return None
    
    img[img < 100] = 0.0
    
    # Normalize the pixel values to get weights in [0, 1]
    weights = img / max_val
    
    # Generate grid of x and y indices
    y_indices, x_indices = np.indices(img.shape)
    
    # Calculate the weighted sum of x and y indices
    weighted_sum_x = np.sum(x_indices * weights)
    weighted_sum_y = np.sum(y_indices * weights)
    
    # Calculate the sum of weights
    total_weight = np.sum(weights)
    
    # Compute the center coordinates and add 0.5 for precision
    x_center = (weighted_sum_x / total_weight) + 0.5
    y_center = (weighted_sum_y / total_weight) + 0.5
    
    return (x_center, y_center)

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length=50):
    # Define points for the axes in the object coordinate system
    axes = np.float32([
        [0, 0, 0],              # Origin
        [axis_length, 0, 0],    # X-axis point
        [0, axis_length, 0],    # Y-axis point
        [0, 0, axis_length]     # Z-axis point
    ])

    # Project the axes points onto the image plane
    imgpts, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2)

    # Convert projected points to integer coordinates
    imgpts = np.int32(imgpts)

    # Extract the projected points
    origin = tuple(imgpts[0])
    x_axis = tuple(imgpts[1])
    y_axis = tuple(imgpts[2])
    z_axis = tuple(imgpts[3])

    # Draw the axes on the image
    img = cv2.line(img, origin, x_axis, (0, 0, 255), 3)  # X-axis in red
    img = cv2.line(img, origin, y_axis, (0, 255, 0), 3)  # Y-axis in green
    img = cv2.line(img, origin, z_axis, (255, 0, 0), 3)  # Z-axis in blue
    return img

# Read the image
image = cv2.imread('C:\\Users\\v3n0w\\Downloads\\Camera\\sample_0.png')  # Replace 'image.jpg' with your image file
cam_matrix = np.load("cam_matrix.npy")
dist = np.load("distortion.npy")

# Step 1: Create red_distance image
# Convert to HSV color space
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (7, 7), sigmaX=0)
_, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)


# reference_red_hue = 172
# # Separate the hue channel
# hue = hsv_image[:, :, 0].astype(np.float32)
# cv2.imshow('Hue', hue.astype(np.uint8))
# cv2.imshow('S', hsv_image[:, :, 1])
# cv2.imshow('V', hsv_image[:, :, 2])

# print(np.max(hue)), print(np.min(hue))
# # Compute the distance to red hue (considering the circular nature)
# distance_to_red = np.minimum(np.abs(hue - reference_red_hue), np.abs(hue - (reference_red_hue - 180)))
# distance_to_brigtness = np.abs(hsv_image[:, :, 1].astype(np.float32) - 215)
# distance_to_brigtness /= 255
# distance_to_brigtness = 1 - distance_to_brigtness
# distance_to_brigtness *= (255/2.0)
# # distance_to_brigtness = distance_to_brigtness.astype(np.uint8)


# # Normalize the distance and invert it so that red pixels have high values
# max_distance = 90
# red_distance = (1 - distance_to_red / max_distance) * (255/2.0)
# red_distance += distance_to_brigtness
# red_distance = red_distance.astype(np.uint8)

# cv2.imshow('Red Distance', red_distance)

# # Step 2: Threshold the red_distance image and detect blobs
# _, thresh = cv2.threshold(red_distance, 235, 255, cv2.THRESH_BINARY)

# cv2.imshow('thresh', thresh)

# # Apply morphological operations to clean up the mask
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('thresh2', thresh)


# Detect connected components (blobs)
num_labels, labels_im = cv2.connectedComponents(thresh)
print(num_labels)

# Collect blob statistics
blob_stats = []
for label in range(1, num_labels):
    # Create a mask for the current blob
    blob_mask = (labels_im == label).astype(np.uint8) * 255

    # Find contours of the blob
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area > 4:  # Filter out small blobs
            x, y, w, h = cv2.boundingRect(cnt)
            blob_stats.append({'label': label, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area})


for blob in blob_stats:
    print(blob)

blobs_position = []

for blob_id, blob in enumerate(blob_stats):
    x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']

    # Expand the bounding rectangle
    padding = 1
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, image.shape[1] - 1)
    y2 = min(y + h + padding, image.shape[0] - 1)

    # Crop the gradient magnitude image
    cropped_grad = gray_image[y1:y2, x1:x2]
    cv2.imshow(f"Blob{blob_id}", cropped_grad)

    blob_x, blob_y = calculate_center_of_blob(cropped_grad)
    blob_x += x1
    blob_y += y1

    blobs_position.append((blob_x, blob_y))

sort_x = sorted(blobs_position, key=lambda x: x[0])
sort_y = sorted(blobs_position, key=lambda x: x[1])

blob_left_pos = sort_x[0]
blob_right_pos = sort_x[-1]

blob_top_pos = sort_y[0]
blob_bottom_pos = sort_y[-1]

print(blob_left_pos, blob_right_pos, blob_top_pos, blob_bottom_pos)

ball_positions = np.array([blob_left_pos, blob_right_pos, blob_top_pos, blob_bottom_pos])

known_horizontal_distance = 60
known_vertical_distance = 50
# Known distances between balls
dx = known_horizontal_distance / 2
dy = known_vertical_distance / 2

object_points = np.array([
    [-dx, 0, 0],    # Left marker
    [dx, 0, 0],     # Right marker
    [0, -dy, 0],    # Top marker
    [0, dy, 0]    # Bottom marker
], dtype=np.float32)

success, rotation_vector, translation_vector = cv2.solvePnP(
    object_points,
    ball_positions,
    cam_matrix,
    dist
)

# Convert rotation vector to rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

print("Rotation Matrix:\n", rotation_matrix)
print("Translation Vector:\n", translation_vector)

# Assuming you have an image loaded as 'img'
img_with_axes = draw_axes(
    image,
    cam_matrix,
    dist,
    rotation_vector,
    translation_vector
)




# # Step 3: Calculate the gradient of the red_distance image
# grad_x = cv2.Sobel(red_distance, cv2.CV_64F, 1, 0, ksize=3)
# grad_y = cv2.Sobel(red_distance, cv2.CV_64F, 0, 1, ksize=3)
# grad_magnitude = cv2.magnitude(grad_x, grad_y)
# grad_magnitude = cv2.convertScaleAbs(grad_magnitude)

# cv2.imshow('gradient', grad_magnitude)

# # Step 4: For each candidate blob, fit circles by sampling edge points
# circle_params_list = []

# for blob_id, blob in enumerate(blob_stats):
#     x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']

#     # Expand the bounding rectangle
#     padding = 10
#     x1 = max(x - padding, 0)
#     y1 = max(y - padding, 0)
#     x2 = min(x + w + padding, image.shape[1] - 1)
#     y2 = min(y + h + padding, image.shape[0] - 1)

#     # Crop the gradient magnitude image
#     cropped_grad = grad_magnitude[y1:y2, x1:x2]

#     # Threshold to get edge points
#     _, edge_mask = cv2.threshold(cropped_grad, 240, 255, cv2.THRESH_BINARY)

#     cv2.imshow(f"Blob edge {blob_id}", edge_mask)

#     # Find edge points
#     edge_points = np.column_stack(np.nonzero(edge_mask))

#     if edge_points.shape[0] < 3:
#         continue  # Not enough points to fit a circle

#     # Convert to original coordinates (x, y)
#     edge_points_global = edge_points[:, ::-1] + np.array([x1, y1])

#     # Initialize lists to store circle parameters
#     centers = []
#     radii = []

#     # Perform multiple iterations of circle fitting
#     num_iterations = 50
#     for _ in range(num_iterations):
#         # Randomly sample 3 edge points
#         sampled_indices = random.sample(range(edge_points_global.shape[0]), 3)
#         p1 = edge_points_global[sampled_indices[0]]
#         p2 = edge_points_global[sampled_indices[1]]
#         p3 = edge_points_global[sampled_indices[2]]

#         # Fit a circle to these points
#         center, radius = fit_circle_3pts(p1, p2, p3)
#         if center is not None and radius is not None:
#             centers.append(center)
#             radii.append(radius)

#     if len(centers) == 0:
#         continue  # No valid circles were found

#     # Convert lists to numpy arrays for median calculation
#     centers = np.array(centers)
#     radii = np.array(radii)

#     # Calculate the median of the circle parameters
#     median_center = np.median(centers, axis=0)
#     median_radius = np.median(radii)

#     circle_params_list.append({
#         'center': median_center,
#         'radius': median_radius,
#         'area': blob['area']
#     })

# # Step 5: Select the 4 best circles based on area or consistency
# circle_params_list.sort(key=lambda x: x['area'], reverse=True)
# selected_circles = circle_params_list[:4]

# # Draw the selected circles on the image
# output_image = image.copy()
# for circle in selected_circles:
#     center = (int(round(circle['center'][0])), int(round(circle['center'][1])))
#     radius = int(round(circle['radius']))
#     cv2.circle(output_image, center, radius, (0, 255, 0), 2)

# # Display the result
# cv2.imshow('Detected Circles', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow('Image', img_with_axes)
cv2.imshow('Gray Image', gray_image)
cv2.imshow('Detected White', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
