import cv2
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict
import time

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

object_points = object_points_p5_large
get_object_points = get_object_points_p5
num_obj_points = object_points.shape[0]

@njit
def process_cropped_region_numba(region: np.ndarray, background_noise: float, perc_25: float):
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

def calculate_center_of_blob(blob: dict, gray_image: np.ndarray):
    """
    Calculate the center of the blob based on the blob statistics and grayscale image.
    """
    x, y, w, h, perc_50, background_noise = blob['x'], blob['y'], blob['w'], blob['h'], blob['perc_50'], blob['perc_10']

    padding = 2
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray_image.shape[1] - 1)
    y2 = min(y + h + padding, gray_image.shape[0] - 1)

    cropped_grad = gray_image[y1:y2, x1:x2]

    perc_50 -= background_noise

    if perc_50 <= 0:
        return None

    center = process_cropped_region_numba(cropped_grad, background_noise, perc_50)
    if center[0] != -1.0:
        blob_x, blob_y = center
        blob_x += x1
        blob_y += y1
        blob_x = (gray_image.shape[1]-1) - blob_x
        return (blob_x, blob_y)

    return None

@njit
def get_blobs_stats(img, mapping, blob_features): # Function is compiled to machine code when called the first time

    current_label = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img[y, x]

            if val > 0:
                if val in mapping:
                    label = mapping[val]
                else:
                    label = current_label
                    mapping[val] = current_label
                    current_label += 1

                if blob_features[label, 0] > x:
                    blob_features[label, 0] = x
                if blob_features[label, 1] < x:
                    blob_features[label, 1] = x
                if blob_features[label, 2] > y:
                    blob_features[label, 2] = y
                if blob_features[label, 3] < y:
                    blob_features[label, 3] = y
                blob_features[label, 4] += 1

    return current_label

blob_features = np.zeros((50, 5), dtype=np.int64)
d = Dict.empty(
    key_type=types.int64,
    value_type=types.int64,
)

time.sleep(20)
img = cv2.imread("sample_frame_0.png", cv2.IMREAD_GRAYSCALE)
base_path = "calib_images_icam_8mm_2\\"
cam_matrix = np.load(base_path + "new_cam_matrix.npy")
dist_coeffs = np.load(base_path + "distortion.npy")

# img = np.zeros((1920, 1200), dtype=np.uint8)

binary_img_gpu = cv2.cuda_GpuMat((640, 400), cv2.CV_8U)
resized_img_gpu = cv2.cuda_GpuMat((640, 400), cv2.CV_8U)
label_img_gpu = cv2.cuda_GpuMat((640, 400), cv2.CV_32S)
base_img_gpu = cv2.cuda_GpuMat((1920, 1200), cv2.CV_8U)
st = cv2.cuda.Stream()
# stime = time.perf_counter()

def fast_percentile(array, perc=50):

    k = int(array.size * (perc/100.0))
    return np.partition(array.flatten(), k)[k]

proc_times = []
gpu_times = []
for i in range(10000):

    if i == 5:
        stotal = time.perf_counter()
    stime = time.perf_counter()
    img = img.copy()
    sgpu = time.perf_counter()

    rframe = cv2.resize(img, (640, 400), interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(rframe, 80, 255, cv2.THRESH_BINARY)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # base_img_gpu.upload(img, stream=st)

    # d = Dict.empty(
    # key_type=types.int64,
    # value_type=types.int64,
    # )

    # cv2.cuda.resize(base_img_gpu, (640,400), interpolation = cv2.INTER_LINEAR, dst=resized_img_gpu, stream=st)
    # cv2.cuda.threshold(resized_img_gpu, 80, 255, cv2.THRESH_BINARY, dst=binary_img_gpu, stream=st)
    # cv2.cuda.connectedComponents(binary_img_gpu, labels=label_img_gpu)
    
    # labels_im = label_img_gpu.download()
    # rframe = resized_img_gpu.download()
    egpu = time.perf_counter() - sgpu
    gpu_times.append(egpu)
    # # perc = fast_percentile(rframe, 50)
    # # perc = fast_percentile(rframe, 10)
    # # blob_features[:] = 0
    # blob_features = np.zeros((50, 5), dtype=np.int64)
    # blob_features[:, 0] = 5000
    # blob_features[:, 2] = 5000
    # # blob_features = [[5000, 0, 5000, 0, 0] for _ in range(20)]
    # # d.clear()
    # num_labels = get_blobs_stats(labels_im, d, blob_features)
    # blob_features = blob_features[:num_labels]
    # print(labels)
    # print(blob_features)

    blob_stats = []
    for label in range(1, num_labels):
        
        x, y, w, h, area = stats[label]
        # x, x2, y, y2, area = blob_features[label]
        # h = (y2 - y) + 1
        # w = (x2 - x) + 1
        if area > 9:  # Filter out small blobs
            roi_labels = labels_im[y:y+h, x:x+w]
            roi_gray = rframe[y:y+h, x:x+w]
            blob_mask = roi_labels > 0
            perc_50 = fast_percentile(roi_gray[blob_mask], 50)
            perc_10 = fast_percentile(roi_gray[~blob_mask], 10)
            blob_stats.append({
                'x': x*3,
                'y': y*3,
                'w': w*3,
                'h': h*3,
                'perc_50': perc_50,
                'perc_10': perc_10,
            })

    # If more than num_obj_points blobs, keep the top num_obj_points by area
    if len(blob_stats) > num_obj_points:
        blob_stats = sorted(blob_stats, key=lambda x: x["area"], reverse=True)[:num_obj_points]

    blobs_position = []

    estats = time.perf_counter()
    # print(f"Stats time: {estats - econnected}")

    for blob in blob_stats:
        center = calculate_center_of_blob(blob, img)
        if center is not None:
            blobs_position.append(center)

    # ecblob = time.perf_counter()
    # print(f"Stats time: {ecblob - estats}")

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

        proc_time = time.perf_counter() - stime
        if i!=0:
            proc_times.append(proc_time)
        if proc_time > 0.0055:
            print(f"Warning {proc_time}")
        time.sleep(max(0.006 - proc_time, 0.0))
        # print(f"Total pose time: {time.perf_counter() - stime}")

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = translation_vector.flatten()
    # print(f"Processing time: {time.perf_counter()-stime}")

# print(blob_features)
print(f"Total Processing time: {time.perf_counter()-stotal}")
proc_times = np.array(proc_times)
print(np.mean(proc_times))
print(np.std(proc_times))
print(np.percentile(proc_times, 99))
print(np.max(proc_times))

print(np.mean(gpu_times))
print(np.std(gpu_times))
print(np.percentile(gpu_times, 99))
print(np.max(gpu_times))