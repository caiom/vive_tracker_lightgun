import cv2
import numpy as np
from typing import Optional, Tuple

def get_object_points_p2(blobs_position):
    sort_y = sorted(blobs_position, key=lambda x: x[1])

    sort_middle = sorted([sort_y[1], sort_y[2], sort_y[3]], key=lambda x: x[0])
    sort_bottom = sorted([sort_y[4], sort_y[5]], key=lambda x: x[0])

    top_right = sort_y[0]
    middle_left = sort_middle[0]
    middle_center = sort_middle[1]
    middle_right = sort_middle[2]
    bottom_left = sort_bottom[0]
    bottom_right = sort_bottom[1]

    return np.array([top_right, middle_left, middle_center, middle_right, bottom_left, bottom_right])

def get_object_points_p1(blobs_position):
    sort_y = sorted(blobs_position, key=lambda x: x[1])

    sort_middle = sorted([sort_y[1], sort_y[2]], key=lambda x: x[0])
    sort_bottom = sorted([sort_y[3], sort_y[4]], key=lambda x: x[0])

    top_right = sort_y[0]
    middle_left = sort_middle[0]
    middle_right = sort_middle[1]
    bottom_left = sort_bottom[0]
    bottom_right = sort_bottom[1]

    return np.array([top_right, middle_left, middle_right, bottom_left, bottom_right])

object_points_p2 = np.array([
    [-30, 25, 0],    # top right
    [30, 0, 0],     # middle left
    [0, 0, 10],     # middle center
    [-30, 0, 0],     # middle right
    [30, -25, 0],    # bottom left
    [-30, -25, 0]    # bottom right
], dtype=np.float32)

object_points_p1 = np.array([
    [-30, -25, 0],    # top right
    [-30, 0, 0],     # middle left
    [30, 0, 0],     # middle right
    [-30, 25, 0],    # bottom left
    [30, 25, 0]    # bottom right
], dtype=np.float32)

object_points = object_points_p1
get_object_points = get_object_points_p1
num_obj_points = object_points.shape[0]

class PoseEstimator:
    def __init__(self):
        """
        Initializes the PoseEstimator by setting up the camera and loading calibration data.
        """
        # Open the default camera (usually the webcam)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video stream.")

        # Try to set focus (if supported by the webcam)
        focus_value = 10  # Adjust based on camera capabilities
        self.cap.set(cv2.CAP_PROP_FOCUS, focus_value)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -12)
        self.cap.set(cv2.CAP_PROP_GAIN, 100)
        self.cap.set(cv2.CAP_PROP_FPS, 120.0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Load calibration data
        self.cam_matrix = np.load("new_cam_matrix.npy")
        self.dist_coeffs = np.load("distortion.npy")
        self.mapx = np.load("mapx.npy")
        self.mapy = np.load("mapy.npy")

        # Initialize previous rotation and translation vectors
        self.prev_rotation_vector = None
        self.prev_translation_vector = None

    def __del__(self):
        """
        Releases the camera resource when the instance is destroyed.
        """
        if self.cap.isOpened():
            self.cap.release()

    @staticmethod
    def calculate_center_of_blob(image: np.ndarray, max_val: float) -> Optional[Tuple[float, float]]:
        """
        Calculate the center of the white blob in a 2D uint8 NumPy array.

        Parameters:
        - image (np.ndarray): 2D array of type uint8 representing the image.
        - max_val (float): The maximum pixel value to consider.

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

        if total_weight == 0:
            return None

        # Compute the center coordinates and add 0.5 for precision
        x_center = (weighted_sum_x / total_weight) + 0.5
        y_center = (weighted_sum_y / total_weight) + 0.5

        return (x_center, y_center)

    def get_pose(self) -> Optional[np.ndarray]:
        """
        Captures a frame from the camera, processes it to detect blobs, and computes the pose.

        Returns:
        - np.ndarray: A 4x4 matrix combining rotation and translation if pose is detected.
        - None: If a valid pose cannot be determined.
        """
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            return None

        # Remap the frame using the loaded calibration data
        frame = cv2.remap(frame, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)

        # Convert to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, thresh = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels_im = cv2.connectedComponents(thresh)

        # Collect blob statistics
        blob_stats = []
        for label in range(1, num_labels):  # Start from 1 to exclude the background
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
                    blob_stats.append({
                        'label': label,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'area': area,
                        "perc_25": perc_25
                    })

        # If more than num_obj_points blobs, keep the top num_obj_points by area
        if len(blob_stats) > num_obj_points:
            blob_stats = sorted(blob_stats, key=lambda x: x["area"], reverse=True)[:num_obj_points]

        blobs_position = []

        for blob in blob_stats:
            x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']

            # Expand the bounding rectangle
            padding = 4
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, gray_image.shape[1] - 1)
            y2 = min(y + h + padding, gray_image.shape[0] - 1)

            # Crop the region of interest
            cropped_grad = gray_image[y1:y2, x1:x2]
            cropped_grad[cropped_grad > blob['perc_25']] = blob['perc_25']

            # Calculate the center of the blob
            center = self.calculate_center_of_blob(cropped_grad, blob['perc_25'])
            if center is not None:
                blob_x, blob_y = center
                blob_x += x1
                blob_y += y1
                blobs_position.append((blob_x, blob_y))

        # Proceed only if exactly 6 blobs are detected
        if len(blobs_position) == num_obj_points:

            # Assemble the image points in the required order
            ball_positions = get_object_points(blobs_position)

            # Solve PnP to find rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points,
                ball_positions,
                self.cam_matrix,
                None,
                flags=cv2.SOLVEPNP_SQPNP
            )

            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Create a 4x4 transformation matrix
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = rotation_matrix
                pose_matrix[:3, 3] = translation_vector.flatten()

                # Update previous rotation and translation vectors
                self.prev_rotation_vector = rotation_vector
                self.prev_translation_vector = translation_vector

                return pose_matrix

        # Return None if pose cannot be determined
        return None
