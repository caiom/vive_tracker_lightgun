import threading
import cv2
import numpy as np
from typing import Optional, Tuple
from icamera import ICamera
import time
import multiprocessing
from queue import Empty
from numba import njit

cv2.setNumThreads(2)

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
get_object_points = get_object_points_p5
num_obj_points = object_points.shape[0]

@njit
def process_cropped_region_numba(region: np.ndarray, background_noise: float, perc_25: float) -> (float, float):
    rows, cols = region.shape

    total_weight = 0.0
    weighted_sum_x = 0.0
    weighted_sum_y = 0.0

    # Step 1: Process the region in-place
    for y in range(rows):
        for x in range(cols):
            val = region[y, x]
            if val < background_noise:
                val = 0
            else:
                val = val - background_noise
                if val > perc_25:
                    val = perc_25
            val = float(val)
            weight = val / perc_25
            total_weight += weight
            weighted_sum_x += x * weight
            weighted_sum_y += y * weight

    if total_weight == 0.0:
        return -1.0, -1.0

    x_center = (weighted_sum_x / total_weight) + 0.5
    y_center = (weighted_sum_y / total_weight) + 0.5
    return x_center, y_center

def pose_worker(pose_queue: multiprocessing.Queue, stop_event):
    """
    Worker function to run in a separate process.
    Initializes the camera and computes poses, sending them back via a queue.
    """

    import psutil
    import os

    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    p = psutil.Process(os.getpid())
    p.cpu_affinity([0, 1])  # Pin to cores 0 and 1

    cam = ICamera()

    # Load calibration data
    base_path = "calib_images_icam_8mm_2\\"
    cam_matrix = np.load(base_path + "new_cam_matrix.npy")
    dist_coeffs = np.load(base_path + "distortion.npy")
    mapx = np.load(base_path + "mapx.npy")
    mapy = np.load(base_path + "mapy.npy")

    while not stop_event.is_set():
        pose = compute_pose(cam, cam_matrix, dist_coeffs, mapx, mapy)
        if pose is not None:
            try:
                # Keep only the latest pose
                if not pose_queue.empty():
                    try:
                        pose_queue.get_nowait()
                    except Empty:
                        pass
                pose_queue.put(pose)
            except multiprocessing.queues.Full:
                pass  # If queue is full, skip

    cam.cleanup()

def compute_pose(cam, cam_matrix, dist_coeffs, mapx, mapy):
    """
    Capture frame, process it, and compute the pose.
    This function mirrors the _compute_pose method from your original class.
    """
    stime = time.perf_counter()
    frame = cam.grab()
    if frame is None:
        return None
    
    egrab = time.perf_counter()
    print(f"Grab time: {egrab - stime}")
    
    frame = np.reshape(frame, (frame.shape[0], frame.shape[1]))

    # frame = cv2.flip(frame, 1)
    # Uncomment and adjust if remapping is needed
    # frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    rframe = cv2.resize(frame, (640, 400), interpolation=cv2.INTER_LINEAR)
    eresize = time.perf_counter()
    print(f"Resize time: {eresize - egrab}")

    # return None

    # Apply binary thresholding
    _, thresh = cv2.threshold(rframe, 80, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    econnected = time.perf_counter()
    print(f"CC time: {econnected - eresize}")
    # Collect blob statistics
    blob_stats = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area > 4:  # Filter out small blobs
            roi_labels = labels_im[y:y+h, x:x+w]
            roi_gray = rframe[y:y+h, x:x+w]
            blob_mask = roi_labels == label
            perc_25 = np.mean(roi_gray[blob_mask])
            blob_stats.append({
                'label': label,
                'x': x*3,
                'y': y*3,
                'w': w*3,
                'h': h*3,
                'area': area*4,
                'perc_25': perc_25
            })

    if len(blob_stats) < num_obj_points:
        return None
    # If more than num_obj_points blobs, keep the top num_obj_points by area
    if len(blob_stats) > num_obj_points:
        blob_stats = sorted(blob_stats, key=lambda x: x["area"], reverse=True)[:num_obj_points]

    blobs_position = []

    estats = time.perf_counter()
    print(f"Stats time: {estats - econnected}")

    for blob in blob_stats:
        center = calculate_center_of_blob(blob, frame)
        if center is not None:
            blobs_position.append(center)

    ecblob = time.perf_counter()
    print(f"Stats time: {ecblob - estats}")

    if len(blobs_position) == num_obj_points:
        blobs_position = get_object_points(blobs_position)  # Define this function accordingly
        image_points = np.array(blobs_position, dtype=np.float32)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )

        print(f"Total pose time: {time.perf_counter() - stime}")

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = translation_vector.flatten()
            return pose_matrix

    return None

def calculate_center_of_blob(blob: dict, gray_image: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate the center of the blob based on the blob statistics and grayscale image.
    """
    x, y, w, h, perc_25 = blob['x'], blob['y'], blob['w'], blob['h'], blob['perc_25']

    padding = 2
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray_image.shape[1] - 1)
    y2 = min(y + h + padding, gray_image.shape[0] - 1)

    cropped_grad = gray_image[y1:y2, x1:x2]
    # background_noise = np.percentile(cropped_grad[cropped_grad < perc_25], 10)

    background_noise = np.mean(cropped_grad[cropped_grad < perc_25])
    perc_25 -= background_noise

    if perc_25 <= 0:
        return None

    center = process_cropped_region_numba(cropped_grad, background_noise, perc_25)
    if center[0] != -1.0:
        blob_x, blob_y = center
        blob_x += x1
        blob_y += y1
        blob_x = (gray_image.shape[1]-1) - blob_x
        return (blob_x, blob_y)

    return None

class PoseEstimator:
    def __init__(self):
        """
        Initializes the PoseEstimator by setting up the camera and loading calibration data.
        """
        self.pose_queue = multiprocessing.Queue(maxsize=1)
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=pose_worker,
            args=(self.pose_queue, self.stop_event),
            daemon=True
        )
        self.process.start()
        self._pose_lock = threading.Lock()
        self._pose_matrix = None

    def cleanup(self):
        """
        Stops the background process and cleans up resources.
        """
        self.stop_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
        while not self.pose_queue.empty():
            self.pose_queue.get()
        self.pose_queue.close()

    def get_pose(self) -> Optional[np.ndarray]:
        """
        Retrieves the latest pose matrix from the background process.
        
        Returns:
        - np.ndarray: A 4x4 matrix combining rotation and translation if available.
        - None: If no valid pose has been computed yet.
        """
        with self._pose_lock:
            while not self.pose_queue.empty():
                self._pose_matrix = self.pose_queue.get()
            return self._pose_matrix.copy() if self._pose_matrix is not None else None
    
    def __del__(self):
        """
        Ensures that resources are cleaned up when the instance is destroyed.
        """
        self.cleanup()
