import threading
import cv2
import numpy as np
from typing import Optional, Tuple
from icamera import ICamera
import time
import concurrent.futures

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

def get_object_points_p3(blobs_position):
    sort_y = sorted(blobs_position, key=lambda x: x[1])

    sort_middle = sorted([sort_y[1], sort_y[2], sort_y[3]], key=lambda x: x[0])
    sort_bottom = sorted([sort_y[4], sort_y[5], sort_y[6]], key=lambda x: x[0])

    top_right = sort_y[0]
    middle_left = sort_middle[0]
    middle_center = sort_middle[1]
    middle_right = sort_middle[2]
    bottom_left = sort_bottom[0]
    bottom_center = sort_bottom[1]
    bottom_right = sort_bottom[2]

    return np.array([top_right, middle_left, middle_center, middle_right, bottom_left, bottom_center, bottom_right])

def get_object_points_p4(blobs_position):
    sort_y = sorted(blobs_position, key=lambda x: x[1])

    sort_middle = sorted([sort_y[1], sort_y[2], sort_y[3]], key=lambda x: x[0])

    top_center = sort_y[0]
    middle_left = sort_middle[0]
    middle_center = sort_middle[1]
    middle_right = sort_middle[2]
    bottom = sort_y[-1]

    return np.array([top_center, middle_left, middle_center, middle_right, bottom])

def get_object_points_p5(blobs_position):
    sort_y = sorted(blobs_position, key=lambda x: x[1])

    sort_middle = sorted(sort_y[1:6], key=lambda x: x[0])

    top_center = sort_y[0]
    middle_left = sort_middle[0]
    middle_left_center = sort_middle[1]
    middle_center = sort_middle[2]
    middle_right_center = sort_middle[3]
    middle_right = sort_middle[4]
    bottom_top = sort_y[-2]
    bottom = sort_y[-1]

    return np.array([top_center, 
                     middle_left, 
                     middle_left_center, 
                     middle_center, 
                     middle_right_center,
                     middle_right, 
                     bottom_top, 
                     bottom])

object_points_p4 = np.array([
    [0, 0, 0],    # top center
    [-45, 45.5, 0],     # middle left
    [0, 45.5, 0],     # middle center
    [45, 45.5, 0],     # middle right
    [0, 100, 0],    # bottom
], dtype=np.float32)

object_points_p2 = np.array([
    [-30, -25, 0],    # top right
    [-30, 0, 0],     # middle left
    [0, 0, -10],     # middle center
    [30, 0, 0],     # middle right
    [-30, 25, 0],    # bottom left
    [30, 25, 0]    # bottom right
], dtype=np.float32)

object_points_p1 = np.array([
    [-30, -25, 0],    # top right
    [-30, 0, 0],     # middle left
    [30, 0, 0],     # middle right
    [-30, 25, 0],    # bottom left
    [30, 25, 0]    # bottom right
], dtype=np.float32)

# object_points_p1 = np.array([
#     [-60, -50, 0],    # top right
#     [-60, 0, 0],     # middle left
#     [60, 0, 0],     # middle right
#     [-60, 50, 0],    # bottom left
#     [60, 50, 0]    # bottom right
# ], dtype=np.float32)

object_points_p3 = np.array([
    [-30, -25, 0],    # top right
    [-30, 0, 0],     # middle left
    [0, 0, 0],     # middle center
    [30, 0, 0],     # middle right
    [-30, 25, 0],    # bottom left
    [0, 25, 0],    # bottom center
    [30, 25, 0]    # bottom right
], dtype=np.float32)

object_points_p5 = np.array([
    [0, 0, 0],    # top center
    [-60, 45.5, 0],     # middle left
    [-30, 45.5, 0],     # middle left-center
    [0, 45.5, -10],     # middle center
    [30, 45.5, 0],     # middle right center
    [60, 45.5, 0],     # middle right
    [0, 72.75, 0],     # bottom top
    [0, 100, 0],    # bottom
], dtype=np.float32)

object_points_p5_large = np.array([
    [0, 0, 0],    # top center
    [-60, 43.5, 0],     # middle left
    [-30, 43.5, 0],     # middle left-center
    [0, 43.5, -10],     # middle center
    [30, 43.5, 0],     # middle right center
    [60, 43.5, 0],     # middle right
    [0, 70.75, 0],     # bottom top
    [0, 98, 0],    # bottom
], dtype=np.float32)

object_points = object_points_p5_large
y_offset = 35.0
object_points[:, 1] += y_offset

get_object_points = get_object_points_p5
num_obj_points = object_points.shape[0]

class PoseEstimator:
    def __init__(self):
        """
        Initializes the PoseEstimator by setting up the camera and loading calibration data.
        """
        # Open the default camera (usually the webcam)
        self.cam = ICamera()

        # Load calibration data
        base_path = "calib_images_icam_8mm_2\\"
        self.cam_matrix = np.load(base_path + "new_cam_matrix.npy")
        self.dist_coeffs = np.load(base_path + "distortion.npy")
        self.mapx = np.load(base_path + "mapx.npy")
        self.mapy = np.load(base_path + "mapy.npy")

        # Initialize previous rotation and translation vectors
        self.prev_rotation_vector = None
        self.prev_translation_vector = None

        # Pose storage
        self._pose_matrix = None
        self._pose_lock = threading.Lock()

        # Thread control
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._pose_thread, daemon=True)

        # Initialize ThreadPoolExecutor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)  # Adjust as needed

        # Start the background thread
        self._thread.start()

    def __del__(self):
        """
        Releases the camera resource when the instance is destroyed.
        """
        self.cam.cleanup()

    def cleanup(self):
        """
        Stops the background thread and releases the camera resource.
        """
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.executor.shutdown(wait=True)  # Shutdown the executor
        self.cam.cleanup()

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

        # img[img > max_val] = max_val

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
    
    def _process_blob(self, blob: dict, gray_image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Processes a single blob to calculate its center.

        Parameters:
        - blob (dict): A dictionary containing blob statistics.
        - gray_image (np.ndarray): The grayscale image.

        Returns:
        - tuple: (x_center, y_center) coordinates of the blob's center.
        - None: If the center cannot be determined.
        """
        x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']
        perc_25 = blob['perc_25']

        # Expand the bounding rectangle
        padding = 2
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, gray_image.shape[1] - 1)
        y2 = min(y + h + padding, gray_image.shape[0] - 1)

        # Crop the region of interest
        cropped_grad = gray_image[y1:y2, x1:x2].copy()  # To avoid modifying the original image
        background_noise = np.percentile(cropped_grad[cropped_grad < perc_25], 10)
        cropped_grad[cropped_grad < background_noise] = 0
        cropped_grad[cropped_grad >= background_noise] = cropped_grad[cropped_grad >= background_noise] - background_noise
        cropped_grad[cropped_grad > perc_25] = perc_25

        # Calculate the center of the blob
        center = self.calculate_center_of_blob(cropped_grad, perc_25)
        if center is not None:
            blob_x, blob_y = center
            blob_x += x1
            blob_y += y1
            return (blob_x, blob_y)

        return None
    
    def _pose_thread(self):
        """
        The background thread that continuously captures frames and computes the pose.
        """
        while not self._stop_event.is_set():
            pose = self._compute_pose()
            if pose is not None:
                with self._pose_lock:
                    self._pose_matrix = pose

    def _compute_pose(self) -> Optional[np.ndarray]:
        """
        Captures a frame, processes it, and computes the pose.

        Returns:
        - np.ndarray: A 4x4 matrix combining rotation and translation if pose is detected.
        - None: If a valid pose cannot be determined.
        """
        frame = self.cam.grab()
        if frame is None:
            return None
        
        sbuffer = time.perf_counter()
        frame = cv2.flip(frame, 1)
        
        # Remap the frame using the loaded calibration data
        frame = cv2.remap(frame, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)

        # sresize = time.perf_counter()
        rframe = cv2.resize(frame, (640,400), interpolation = cv2.INTER_LINEAR)
        # print(f"Resize time: {time.perf_counter() - sresize}")

        # Convert to grayscale
        gray_image = frame

        # Apply binary thresholding
        _, thresh = cv2.threshold(rframe, 80, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        # Collect blob statistics
        blob_stats = []
        for label in range(1, num_labels):
            # Extract statistics directly from 'stats'
            x, y, w, h, area = stats[label]

            if area > 4:  # Filter out small blobs
                # Define the ROI for the current blob
                roi_labels = labels_im[y:y+h, x:x+w]
                roi_gray = rframe[y:y+h, x:x+w]

                # Create a mask for the current blob within the ROI
                blob_mask = roi_labels == label

                # Calculate the 50th percentile within the blob mask
                perc_25 = np.percentile(roi_gray[blob_mask], 50)

                blob_stats.append({
                    'label': label,
                    'x': x*3,
                    'y': y*3,
                    'w': w*3,
                    'h': h*3,
                    'area': area*4,
                    'perc_25': perc_25
                })

        # If more than num_obj_points blobs, keep the top num_obj_points by area
        if len(blob_stats) > num_obj_points:
            blob_stats = sorted(blob_stats, key=lambda x: x["area"], reverse=True)[:num_obj_points]

        blobs_position = []

        for blob in blob_stats:
            x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']
            perc_25 = blob['perc_25']

            # Expand the bounding rectangle
            padding = 2
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, gray_image.shape[1] - 1)
            y2 = min(y + h + padding, gray_image.shape[0] - 1)

            # Crop the region of interest
            cropped_grad = gray_image[y1:y2, x1:x2]
            cropped_grad = cropped_grad.copy()  # To avoid modifying the original image
            background_noise = np.percentile(cropped_grad[cropped_grad < blob['perc_25']], 10)
            perc_25 -= background_noise
            cropped_grad[cropped_grad < background_noise] = 0
            cropped_grad[cropped_grad >= background_noise] = cropped_grad[cropped_grad >= background_noise] - background_noise
            cropped_grad[cropped_grad > perc_25] = perc_25

            # Calculate the center of the blob
            center = self.calculate_center_of_blob(cropped_grad, perc_25)
            if center is not None:
                blob_x, blob_y = center
                blob_x += x1
                blob_y += y1
                blobs_position.append((blob_x, blob_y))

        # blobs_position = []

        # # Use ThreadPoolExecutor to process blobs in parallel
        # futures = [
        #     self.executor.submit(self._process_blob, blob, gray_image)
        #     for blob in blob_stats
        # ]

        # for future in concurrent.futures.as_completed(futures):
        #     result = future.result()
        #     if result is not None:
        #         blobs_position.append(result)

        # Proceed only if the expected number of blobs are detected
        if len(blobs_position) == num_obj_points:
            # Replace 'get_object_points' and 'object_points' with your actual implementation
            # For example:
            blobs_position = get_object_points(blobs_position)  # 3D points in object space
            image_points = np.array(blobs_position, dtype=np.float32)

            # Solve PnP to find rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                object_points,
                image_points,
                self.cam_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )

            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Create a 4x4 transformation matrix
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = rotation_matrix
                pose_matrix[:3, 3] = translation_vector.flatten()

                print(f"Pose compute time: {time.perf_counter() - sbuffer}")

                return pose_matrix

        # Return None if pose cannot be determined
        return None

    def get_pose(self) -> Optional[np.ndarray]:
        """
        Returns the most recent valid pose computed by the background thread.

        Returns:
        - np.ndarray: A 4x4 matrix combining rotation and translation if a pose is available.
        - None: If no valid pose has been computed yet.
        """
        with self._pose_lock:
            return self._pose_matrix.copy() if self._pose_matrix is not None else None
