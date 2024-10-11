import cv2
import numpy as np
import time


def show_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the HSV value of the pixel at (x, y)
        hsv_value = hsv_image[y, x]

        print(f"HSV: {hsv_value}")
        
        # # Display the HSV values on the image
        # image_copy = frame.copy()
        # cv2.putText(image_copy, f"HSV: {hsv_value}", (x, y - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # # Show the updated image with the HSV value
        # cv2.imshow("HSV Color Picker", image_copy)

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
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Check if the resolution was set correctly
current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {int(current_width)}x{int(current_height)}")


real_ball_diameter = 19.9

cam_matrix = np.load("cam_matrix.npy")
dist = np.load("distortion.npy")

fx = cam_matrix[0, 0]
fy = cam_matrix[1, 1]
cx = cam_matrix[0, 2]
cy = cam_matrix[1, 2]

focal_length = (fx + fy) / 2.0

cv2.namedWindow("HSV Color Picker")
cv2.setMouseCallback("HSV Color Picker", show_hsv_value)


# Loop to continuously grab frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.undistort(frame, cam_matrix, dist, None, cam_matrix)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    # If the frame was not grabbed correctly, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    lower_red = np.array([165, 100, 150])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)


    st = time.time()
    circles = cv2.HoughCircles(blurred_mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                           param1=50, param2=30, minRadius=10, maxRadius=100)
    print(f"Hough time: {time.time()-st}")
    
    # If some circles are detected, draw them
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        max_i = -1
        max_r = 0
        for i, (x, y, r) in enumerate(circles):
            # Draw the circle in the original image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # Optionally draw the center of the circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            if r > max_r:
                max_i = i
                max_r = r

        if max_i != -1:
            u, v, r = circles[max_i]
            image_ball_diameter = r * 2
            distance_to_ball = (focal_length * real_ball_diameter) / image_ball_diameter
            Z = distance_to_ball
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            print(f"u: {u}, v: {v}, Real-world coordinates: X = {X}, Y = {Y}, Z = {Z}")

    # Find contours of the red ball
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming we found at least one contour
    # if contours:
    #     # Find the largest contour (assuming it's the ball)
    #     contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    #     for i in range(3):
        
    #         # Fit a circle around the largest contour
    #         ((x, y), radius) = cv2.minEnclosingCircle(contours_sorted[i])
            
    #         # Get the diameter of the ball in the image (in pixels)
    #         image_ball_diameter = radius * 2

    #         # Calculate the distance to the ball using the pinhole camera model
    #         distance_to_ball = (focal_length * real_ball_diameter) / image_ball_diameter

    #         print(f"Distance to the ball {i}: {distance_to_ball} units")

    #         # Display the result (optional)
    #         cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    # Set transparency (alpha value between 0 and 1)
    alpha = 0.5

    # Blend the mask and the original image using alpha transparency
    blended_image = cv2.addWeighted(mask_3_channel, alpha, frame, 1 - alpha, 0)
    # Display the resulting frame
    cv2.imshow('HSV Color Picker', blended_image)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit the video stream
    if key == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()