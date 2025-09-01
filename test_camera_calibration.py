import cv2 
import numpy as np 
import os 
import glob 
import pathlib as pl



base_path = "calib_images_icam_8mm_2\\"
new_cam_matrix = np.load(base_path + "new_cam_matrix.npy")
cam_matrix = np.load(base_path + "cam_matrix.npy")
dist_coeffs = np.load(base_path + "distortion.npy")
mapx = np.load(base_path + "mapx.npy")
mapy = np.load(base_path + "mapy.npy")



base_path = pl.Path(".")
images = list(base_path.glob("tsample*.png"))


print(images)
frame_number = 0
for img_path in images:
    frame = cv2.imread(img_path)
    color_img = frame.copy() 
    grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # grayColor = cv2.remap(grayColor, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    # gray_image = frame[:, :, 0]
  
  
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


    ret, corners = cv2.findChessboardCorners( 
                    grayColor, CHECKERBOARD,  
                    cv2.CALIB_CB_ADAPTIVE_THRESH  
                    + cv2.CALIB_CB_FAST_CHECK + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE) 
    if ret == True: 
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv2.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) 
  
        twodpoints.append(corners2) 
  
        # Draw and display the corners 
        image = cv2.drawChessboardCorners(color_img,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
        

        # 6. Calcular a pose da câmera (rotação e translação) usando solvePnP
        retval, rvec, tvec = cv2.solvePnP(objectp3d, corners2, cam_matrix, dist_coeffs)
        projected_points, _ = cv2.projectPoints(objectp3d, rvec, tvec, cam_matrix, dist_coeffs)


        # retval, rvec, tvec = cv2.solvePnP(objectp3d, corners2, new_cam_matrix, None)
        # projected_points, _ = cv2.projectPoints(objectp3d, rvec, tvec, new_cam_matrix, None)
    
        # 8. Calcular o erro de reprojeção (distância entre os pontos detectados e os projetados)
        error = cv2.norm(corners, projected_points, cv2.NORM_L2) / len(projected_points)

        dists = []
        for img_pt, proj_pt in zip(corners, projected_points):

            dist = np.sqrt((img_pt[0, 0] - proj_pt[0,0]) ** 2 + (img_pt[0, 1] - proj_pt[0,1]) ** 2)
            dists.append(dist)
        
        print(np.mean(dists), np.median(dists), np.max(dists))
        print(f'Erro de reprojeção: {error} pixels')


        img_with_points = color_img.copy()
        for pt in projected_points:
            center = np.round(pt[0]).astype(np.int32)
            cv2.circle(img_with_points, center, 5, (0, 255, 0), 2)  # Pontos projetados em verde
        for pt in corners:
            center = np.round(pt[0]).astype(np.int32)
            cv2.circle(img_with_points, center, 5, (0, 0, 255), 2)  # Pontos detectados em vermelho
  
    cv2.imshow('img', image)


    file_name = f'corner_sample_frame_{frame_number}.png'
    frame_number += 1
    
    # Save the current frame
    cv2.imwrite(file_name, image)
    print(f"Frame saved as {file_name}.")
    cv2.waitKey(0)  
  
  
# Extracting path of individual image stored 
# in a given directory. Since no path is 
# specified, it will take current directory 
# jpg files alone 
# base_path = pl.Path("calib_images_icam_8mm_2")
# images = list(base_path.glob("*.png"))
  
# for filename in images: 
#     image = cv2.imread(filename) 
#     image = cv2.flip(image, 1)
#     grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
#     # Find the chess board corners 
#     # If desired number of corners are 
#     # found in the image then ret = true 
#     ret, corners = cv2.findChessboardCorners( 
#                     grayColor, CHECKERBOARD,  
#                     cv2.CALIB_CB_ADAPTIVE_THRESH  
#                     + cv2.CALIB_CB_FAST_CHECK + 
#                     cv2.CALIB_CB_NORMALIZE_IMAGE) 
  
#     # If desired number of corners can be detected then, 
#     # refine the pixel coordinates and display 
#     # them on the images of checker board 
#     if ret == True: 
#         threedpoints.append(objectp3d) 
  
#         # Refining pixel coordinates 
#         # for given 2d points. 
#         corners2 = cv2.cornerSubPix( 
#             grayColor, corners, (11, 11), (-1, -1), criteria) 
  
#         twodpoints.append(corners2) 
  
#         # Draw and display the corners 
#         image = cv2.drawChessboardCorners(image,  
#                                           CHECKERBOARD,  
#                                           corners2, ret) 
  
#     cv2.imshow('img', image)
#     cv2.waitKey(0)  
  
# cv2.destroyAllWindows() 
  
# h, w = image.shape[:2] 
# image_size = (w, h)

# ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
#     threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
#     matrix, distortion, image_size, alpha=0, newImgSize=image_size
# )

# print(matrix)
# print(new_camera_matrix) 

# mapx, mapy = cv2.initUndistortRectifyMap(
#     matrix, distortion, None, new_camera_matrix, image_size, cv2.CV_32FC1
# )

# np.save(str(base_path / "mapx.npy"), mapx)
# np.save(str(base_path / "mapy.npy"), mapy)


# for filename in images: 
#     image = cv2.imread(filename) 
#     # Load a test image
#     # Correct the image distortion
#     undistorted_image = cv2.undistort(image, matrix, distortion, None, new_camera_matrix)
#     # Display the original and corrected image side by side
#     combined_image = np.hstack((image, undistorted_image))
#     cv2.imshow('Original vs Undistorted', combined_image)
#     cv2.waitKey(0)
#     # cv2.destroyAllWindows()
  
  
# # Displaying required output 
# print(" Camera matrix:") 
# print(matrix) 
# np.save(str(base_path / "cam_matrix.npy"), matrix)
# np.save(str(base_path / "new_cam_matrix.npy"), new_camera_matrix)
# np.save(str(base_path / "distortion.npy"), distortion)
  
# print("\n Distortion coefficient:") 
# print(distortion) 
  
# print("\n Rotation Vectors:") 
# print(r_vecs) 
  
# print("\n Translation Vectors:") 
# print(t_vecs) 