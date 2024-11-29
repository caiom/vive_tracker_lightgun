import cv2
import numpy as np
import time

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
# width = 1280
# height = 720
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -9) 
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FPS, 120.0)
cap.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0) 

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
# Loop to continuously grab frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # sremap = time.time()
    # frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    # print(f"Time to remap: {time.time()-sremap}")
    # frame = cv2.undistort(frame, cam_matrix, dist, None, cam_matrix)


    # If the frame was not grabbed correctly, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the resulting frame
    cv2.imshow('HSV Color Picker', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Define a file name (e.g., chessboard.png)
        file_name = f'sample_{frame_number}.png'
        frame_number += 1
        
        # Save the current frame
        cv2.imwrite(file_name, frame)
        print(f"Frame saved as {file_name}")

    # Press 'q' to quit the video stream
    if key == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()