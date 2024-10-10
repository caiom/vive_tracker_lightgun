import cv2
import numpy as np
import time

def calculate_center_of_blob(image: np.ndarray, max_val):
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
    
    if max_val == 0:
        # No white pixels in the image
        return None
    
    img[img > max_val] = max_val
    
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

def put_multiline_text(img, text_lines, position, font, font_scale, color, thickness, line_type=cv2.LINE_AA):
    x, y = position
    for i, line in enumerate(text_lines):
        y_line = y + i * int(20 * font_scale)
        cv2.putText(img, line, (x, y_line), font, font_scale, color, thickness, line_type)

def draw_axes_with_text(img, camera_matrix, dist_coeffs, rvec, tvec, z_vector, axis_length=50):
    # Draw the axes
    img = draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    
    # Format rotation vector and translation vector
    rotation_vector_flat = rvec.flatten()
    translation_vector_flat = tvec.flatten()
    
    # rotation_text = f"Rotation Vector:\n[{rotation_vector_flat[0]:.3f}, {rotation_vector_flat[1]:.3f}, {rotation_vector_flat[2]:.3f}]"
    rotation_text = f"Rotation Vector:\n[{z_vector[0]:.3f}, {z_vector[1]:.3f}, {z_vector[2]:.3f}]"
    translation_text = f"Translation Vector:\n[{translation_vector_flat[0]:.2f}, {translation_vector_flat[1]:.2f}, {translation_vector_flat[2]:.2f}]"
    
    # Positions for the text
    rotation_text_position = (10, 30)
    translation_text_position = (10, 100)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White color
    thickness = 2
    
    # Overlay the text
    rotation_lines = rotation_text.split('\n')
    translation_lines = translation_text.split('\n')
    
    put_multiline_text(img, rotation_lines, rotation_text_position, font, font_scale, color, thickness)
    put_multiline_text(img, translation_lines, translation_text_position, font, font_scale, color, thickness)
    
    return img

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Try to set focus (if supported by the webcam)
focus_value = 10  # You can try different values (depends on the camera model)
if cap.set(cv2.CAP_PROP_FOCUS, focus_value):
    print(f"Focus set to {focus_value}")
else:
    print("Failed to set focus. Your camera may not support focus control.")

# Set resolution (e.g., 1280x720 for HD resolution)
width = 1280
height = 800
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -12) 
cap.set(cv2.CAP_PROP_GAIN, 100)
cap.set(cv2.CAP_PROP_FPS, 120.0)
cap.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)

# Check if the resolution was set correctly
current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {int(current_width)}x{int(current_height)}")


real_ball_diameter = 19.9

cam_matrix = np.load("cam_matrix.npy")
dist = np.load("distortion.npy")
mapx = np.load("mapx.npy")
mapy = np.load("mapy.npy")

fx = cam_matrix[0, 0]
fy = cam_matrix[1, 1]
cx = cam_matrix[0, 2]
cy = cam_matrix[1, 2]

focal_length = (fx + fy) / 2.0
frame_number = 0
prev_rotation_vector = None
prev_translation_vector = None
# Loop to continuously grab frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    # frame = cv2.undistort(frame, cam_matrix, dist, None, cam_matrix)
    # print(cap.get(cv2.CAP_PROP_EXPOSURE))


    # If the frame was not grabbed correctly, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)

    num_labels, labels_im = cv2.connectedComponents(thresh)
    print(num_labels)

    # Collect blob statistics
    blob_stats = []
    for label in range(1, num_labels):
        # Create a mask for the current blob
        blob_mask = labels_im == label
        perc_25 = np.percentile(gray_image[blob_mask], 25)
        blob_mask = blob_mask.astype(np.uint8) * 255

        # Find contours of the blob
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            if area > 4:  # Filter out small blobs
                x, y, w, h = cv2.boundingRect(cnt)
                blob_stats.append({'label': label, 'x': x, 'y': y, 'w': w, 'h': h, 'area': area, "perc_25": perc_25})


    # for blob in blob_stats:
    #     print(blob)

    if len(blob_stats) > 6:
        blob_stats = sorted(blob_stats, key=lambda x: x["area"], reverse=True)
        blob_stats = blob_stats[:6]

    blobs_position = []

    for blob_id, blob in enumerate(blob_stats):
        x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']

        # Expand the bounding rectangle
        padding = 4
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, gray_image.shape[1] - 1)
        y2 = min(y + h + padding, gray_image.shape[0] - 1)

        # Crop the gradient magnitude image
        cropped_grad = gray_image[y1:y2, x1:x2]
        cropped_grad[cropped_grad > blob['perc_25']] = blob['perc_25']
        cv2.imshow(f"Blob{blob_id}", cropped_grad)

        blob_x, blob_y = calculate_center_of_blob(cropped_grad, blob['perc_25'])
        blob_x += x1
        blob_y += y1

        blobs_position.append((blob_x, blob_y))

    if len(blobs_position) == 6:

        
        sort_y = sorted(blobs_position, key=lambda x: x[1])

        sort_middle = sorted([sort_y[1], sort_y[2], sort_y[3]], key=lambda x: x[0])
        sort_bottom = sorted([sort_y[4], sort_y[5]], key=lambda x: x[0])

        top_right = sort_y[0]
        middle_left = sort_middle[0]
        middle_center = sort_middle[1]
        middle_right = sort_middle[2]
        bottom_left = sort_bottom[0]
        bottom_right = sort_bottom[1]

        # print(blob_left_pos, blob_right_pos, blob_top_pos, blob_bottom_pos)

        ball_positions = np.array([top_right, middle_left, middle_center, middle_right, bottom_left, bottom_right])

        print(ball_positions)

        object_points = np.array([
            [30, 25, 0],    # top right
            [-30, 0, 0],     # middle left
            [0, 0, 10],     # middle center
            [30, 0, 0],     # middle right
            [-30, -25, 0],    # bottom left
            [30, -25, 0]    # bottom right
        ], dtype=np.float32)


        object_points = np.array([
            [-30, 25, 0],    # top right
            [30, 0, 0],     # middle left
            [0, 0, 10],     # middle center
            [-30, 0, 0],     # middle right
            [30, -25, 0],    # bottom left
            [-30, -25, 0]    # bottom right
        ], dtype=np.float32)

        # if prev_translation_vector is not None:
        #     success, rotation_vector, translation_vector = cv2.solvePnP(
        #         object_points,
        #         ball_positions,
        #         cam_matrix,
        #         dist,
        #         rvec=prev_rotation_vector,
        #         tvec=prev_translation_vector,
        #         useExtrinsicGuess=True,
        #         flags=cv2.SOLVEPNP_ITERATIVE,
        #     )
        # else:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            ball_positions,
            cam_matrix,
            None,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        print(success)
        #cv2.SOLVEPNP_EPNP
        #cv2.SOLVEPNP_ITERATIVE
        #cv2.SOLVEPNP_IPPE
        # Convert rotation vector to rotation matrix
        # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # normal_vector_camera = rotation_matrix @ np.array([0, 0, 1])
        # if normal_vector_camera[2] > 0:
        #     rotation_vector = -rotation_vector
        #     translation_vector = -translation_vector

        # if prev_rotation_vector is not None:
        #     if np.dot(rotation_vector.flatten(), prev_rotation_vector.flatten()) < 0:
        #         # If the dot product is negative, flip the current rotation vector
        #         rotation_vector = -rotation_vector

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector) 

        prev_rotation_vector = rotation_vector
        prev_translation_vector = translation_vector

        print("Rotation Matrix:\n", rotation_vector)
        print("Translation Vector:\n", translation_vector)

        # Assuming you have an image loaded as 'img'
        img_with_axes = draw_axes_with_text(
            frame,
            cam_matrix,
            dist,
            rotation_vector,
            translation_vector,
            rotation_matrix[2, :],
        )

        cv2.imshow('Image', img_with_axes)
    cv2.imshow('Gray Image', gray_image)
    cv2.imshow('Detected White', thresh)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit the video stream
    if key == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()