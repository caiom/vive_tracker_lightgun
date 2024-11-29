import cv2 
import numpy as np 
import os 
import glob 
import pathlib as pl
  
  
# Define the dimensions of checkerboard 
CHECKERBOARD = (7, 10) 
  
  
# stop the iteration when specified 
# accuracy, epsilon, is reached or 
# specified number of iterations are completed. 
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
  
  
# Vector for 3D points 
threedpoints = [] 
  
# Vector for 2D points 
twodpoints = [] 
  
  
#  3D points real world coordinates 
objectp3d = np.zeros((1, CHECKERBOARD[0]  
                      * CHECKERBOARD[1],  
                      3), np.float32) 
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 15
prev_img_shape = None
  
  
# Extracting path of individual image stored 
# in a given directory. Since no path is 
# specified, it will take current directory 
# jpg files alone 
base_path = pl.Path("C:\\Users\\v3n0w\\Downloads\\Camera\\vive_tracker_lightgun\\calib_images_icam_8mm")
images = list(base_path.glob("*.png"))
  
for filename in images: 
    image = cv2.imread(filename) 
    image = cv2.flip(image, 1)
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    ret, corners = cv2.findChessboardCorners( 
                    grayColor, CHECKERBOARD,  
                    cv2.CALIB_CB_ADAPTIVE_THRESH  
                    + cv2.CALIB_CB_FAST_CHECK + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE) 
  
    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv2.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) 
  
        twodpoints.append(corners2) 
  
        # Draw and display the corners 
        image = cv2.drawChessboardCorners(image,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
  
    # cv2.imshow('img', image)
    # cv2.waitKey(0)  
  
cv2.destroyAllWindows() 
  
h, w = image.shape[:2] 
image_size = (w, h)

ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    matrix, distortion, image_size, alpha=0, newImgSize=image_size
)

print(matrix)
print(new_camera_matrix) 

mapx, mapy = cv2.initUndistortRectifyMap(
    matrix, distortion, None, new_camera_matrix, image_size, cv2.CV_32FC1
)

np.save(str(base_path / "mapx.npy"), mapx)
np.save(str(base_path / "mapy.npy"), mapy)


for filename in images: 
    image = cv2.imread(filename) 
    # Load a test image
    # Correct the image distortion
    undistorted_image = cv2.undistort(image, matrix, distortion, None, new_camera_matrix)
    # Display the original and corrected image side by side
    combined_image = np.hstack((image, undistorted_image))
    cv2.imshow('Original vs Undistorted', combined_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
  
  
# Displaying required output 
print(" Camera matrix:") 
print(matrix) 
np.save(str(base_path / "cam_matrix.npy"), matrix)
np.save(str(base_path / "new_cam_matrix.npy"), new_camera_matrix)
np.save(str(base_path / "distortion.npy"), distortion)
  
print("\n Distortion coefficient:") 
print(distortion) 
  
print("\n Rotation Vectors:") 
print(r_vecs) 
  
print("\n Translation Vectors:") 
print(t_vecs) 