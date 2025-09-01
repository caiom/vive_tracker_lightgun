import threading
import cv2
import numpy as np
from typing import Optional, Tuple
from icamera import ICamera
import time
import multiprocessing
from queue import Empty
from numba import njit
from numba.core import types
from numba.typed import Dict

cv2.setNumThreads(2)

def get_object_points_p8(blobs_position):
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

object_points_p8_large = np.array([
    [0, 0, 0],    # top center
    [-60, 43.5, 0],     # middle left
    [-30, 43.5, 0],     # middle left-center
    [0, 43.5, -10],     # middle center
    [30, 43.5, 0],     # middle right center
    [60, 43.5, 0],     # middle right
    [0, 70.75, 0],     # bottom top
    [0, 98, 0],    # bottom
], dtype=np.float32)


def get_object_points_p5(blobs_position):
    sort_y = sorted(blobs_position, key=lambda x: x[1])

    sort_middle = sorted(sort_y[1:4], key=lambda x: x[0])

    top_center = sort_y[0]
    middle_left = sort_middle[0]
    middle_center = sort_middle[1]
    middle_right = sort_middle[2]
    bottom = sort_y[-1]

    return np.array([top_center, 
                     middle_left, 
                     middle_center, 
                     middle_right, 
                     bottom])


object_points_p5_large = np.array([
    [0, 0, 0],    # top center
    [-60, 43.5, 0],     # middle left
    [0, 43.5, -10],     # middle center
    [60, 43.5, 0],     # middle right
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
            val = float(region[y, x])
            if val < background_noise:
                val = 0.0
            else:
                val = val - background_noise
                if val > perc_25:
                    val = perc_25

            weight = val / perc_25
            total_weight += weight
            weighted_sum_x += x * weight
            weighted_sum_y += y * weight

    if total_weight == 0.0:
        return -1.0, -1.0

    return (weighted_sum_x / total_weight), (weighted_sum_y / total_weight)

def pose_worker(shared_arr, has_new, last_valid, arr_lock, stop_event):
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
    base_path = "calib_images_icam_8mm_3\\"
    cam_matrix = np.load(base_path + "cam_matrix.npy")
    dist_coeffs = np.load(base_path + "distortion.npy")
    mapx = np.load(base_path + "mapx.npy")
    mapy = np.load(base_path + "mapy.npy")


    arr = np.frombuffer(shared_arr.get_obj(), dtype='d')  # 'd' = double precision float
    arr = arr.reshape((4, 4))

    while not stop_event.is_set():
        pose = compute_pose(cam, cam_matrix, dist_coeffs, mapx, mapy)
        with arr_lock:
            if pose is not None:
                arr[:] = pose[:]
                last_valid.value = True
                has_new.value = True
            else:
                last_valid.value = False
    cam.cleanup()

def fast_percentile(array, perc=50):
    k = int(array.size * (perc/100.0))
    return np.partition(array.flatten(), k)[k]

def compute_pose(cam, cam_matrix, dist_coeffs, mapx, mapy):
    """
    Capture frame, process it, and compute the pose.
    This function mirrors the _compute_pose method from your original class.
    """
    print("Hi")
    stime = time.perf_counter()
    frame = cam.grab()
    if frame is None:
        print("no frame")
        return None
    
    egrab = time.perf_counter()
    print(f"Grab time: {egrab - stime}")
    
    frame = np.reshape(frame, (frame.shape[0], frame.shape[1]))

    # frame = cv2.flip(frame, 1)
    # Uncomment and adjust if remapping is needed
    # frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    resize_factor = 3

    rframe = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor), interpolation=cv2.INTER_LINEAR)
    eresize = time.perf_counter()
    print(f"Resize time: {eresize - egrab}")

    # # return None

    # # Apply binary thresholding
    val, thresh = cv2.threshold(rframe, 80, 255, cv2.THRESH_BINARY)

    # print(f"otsu: {val}")

    # # Find connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    econnected = time.perf_counter()
    print(f"Connected time: {econnected - eresize}")


    print(f"num blabels{num_labels}")

    # Collect blob statistics
    blob_stats = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        padding = max(3, int(0.2*w))

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, rframe.shape[1] - 1)
        y2 = min(y + h + padding, rframe.shape[0] - 1)

        if area > 9:  # Filter out small blobs
            # roi_labels = labels_im[y1:y2, x1:x2]
            roi_gray = rframe[y1:y2, x1:x2]
            # blob_mask = roi_labels == label

            otsu, blob_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


            kernel = np.ones((3, 3), np.uint8)
            mask_nuc = cv2.erode(blob_mask.astype(np.uint8), kernel, iterations=1).astype(bool)  # interior
            mask_bg  = ~cv2.dilate(blob_mask.astype(np.uint8), kernel, iterations=1).astype(bool)  # safe background

            if np.sum(mask_nuc) == 0:
                print("nuc 0")
                continue
            if np.sum(mask_bg) == 0:
                print("bg 0")
                continue
            perc_50 = fast_percentile(roi_gray[mask_nuc], 1)
            # perc_50 = np.min(roi_gray[mask_nuc])
            
            # perc_50 = np.percentile(roi_gray[blob_mask], 25)
            # nmask = np.logical_not(blob_mask)

            # print(perc_50)

            # if nmask.size == 0:
            #     print(f"Wrong mask of blob {area} - {num_labels} - {label}")
            #     perc_10 = 0
            # else:
            #     # perc_10 = fast_percentile(roi_gray[mask_bg], 75)
            #     # perc_10 = np.max(roi_gray[mask_bg])
            #     perc_10 = np.percentile(roi_gray[nmask], 15)
            #     # print(perc_10)
            #     # print("-----")

            perc_10 = fast_percentile(roi_gray[mask_bg], 99)
            # perc_10 = np.max(roi_gray[mask_bg])

            # print(perc_50, perc_10)
            # print(w)
            blob_stats.append({
                'x1': x1*resize_factor,
                'y1': y1*resize_factor,
                'x2': x2*resize_factor,
                'y2': y2*resize_factor,
                'perc_50': perc_50,
                'perc_10': perc_10,
                'area': area,
                'otsu': otsu,
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
    print(f"Center of blob time: {ecblob - estats}")

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

        if (time.perf_counter() - egrab) > 0.006:
            print(f"Warning {time.perf_counter() - egrab}")
        print(f"Total pose time: {time.perf_counter() - stime}")
        print(f"Processing time: {time.perf_counter() - egrab}")

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = translation_vector.flatten()

            print(translation_vector.flatten())
            return pose_matrix

    return None

def calculate_center_of_blob(blob: dict, gray_image: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate the center of the blob based on the blob statistics and grayscale image.
    """
    x1, y1, x2, y2, perc_50, background_noise, otsu = blob['x1'], blob['y1'], blob['x2'], blob['y2'], blob['perc_50'], blob['perc_10'], blob['otsu']

    cropped_grad = gray_image[y1:y2, x1:x2]

    bin_blob = cropped_grad > otsu
    weights = bin_blob.astype(np.float32)

    h,w = bin_blob.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m00 = bin_blob.sum()

    if m00 < 10:
        return None
    cx = (weights * xx).sum() / m00
    cy = (weights * yy).sum() / m00

    blob_x, blob_y = cx, cy
    blob_x += x1
    blob_y += y1
    blob_x = gray_image.shape[1] - blob_x
    return (blob_x, blob_y)

    # perc_50 -= background_noise

    # if perc_50 <= 0:
    #     return None

    # center = process_cropped_region_numba(cropped_grad, background_noise, perc_50)
    # if center[0] != -1.0:
    #     blob_x, blob_y = center
    #     blob_x += x1
    #     blob_y += y1
    #     blob_x = gray_image.shape[1] - blob_x
    #     return (blob_x, blob_y)

    return None

class PoseEstimator:
    def __init__(self):
        """
        Initializes the PoseEstimator by setting up the camera and loading calibration data.
        """
        shape = (4, 4)
        dtype = 'd'  # 'd' = float64 in multiprocessing.Array
        total_elements = shape[0] * shape[1]

        self.last_valid = multiprocessing.Value('b', False)
        self.has_new = multiprocessing.Value('b', False)
        self.shared_arr = multiprocessing.Array(dtype, total_elements)
        self.lock = multiprocessing.Lock()

        self.pose_queue = multiprocessing.Queue(maxsize=100)
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=pose_worker,
            args=(self.shared_arr, self.has_new, self.last_valid, self.lock, self.stop_event),
            daemon=True
        )
        self.process.start()
        self._pose_matrix = np.frombuffer(self.shared_arr.get_obj(), dtype=dtype)
        self._pose_matrix = np.reshape(self._pose_matrix, (4, 4))

    def cleanup(self):
        """
        Stops the background process and cleans up resources.
        """
        self.stop_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()

    def get_pose(self) -> Optional[np.ndarray]:
        """
        Retrieves the latest pose matrix from the background process.
        
        Returns:
        - np.ndarray: A 4x4 matrix combining rotation and translation if available.
        - None: If no valid pose has been computed yet.
        """
        with self.lock:
            pose_matrix = self._pose_matrix.copy()
            valid = self.last_valid.value
            is_new = self.has_new.value
            if is_new:
                self.has_new.value = False

        return pose_matrix, valid, is_new
    
    def __del__(self):
        """
        Ensures that resources are cleaned up when the instance is destroyed.
        """
        self.cleanup()

if __name__ == "__main__":
    pose = PoseEstimator()

    # time.sleep(180)
    
    st = time.time()

    

    while (time.time() - st) < 180:
        # pose_matrix = pose.get_pose()
        time.sleep(1)

    pose.cleanup()
