import cv2

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
cap.set(cv2.CAP_PROP_FPS, 120.0)
cap.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

# Check if the resolution was set correctly
current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {int(current_width)}x{int(current_height)}")

chessboard_size = (7, 10)
frame_number=1

# Loop to continuously grab frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not grabbed correctly, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # # Convert the frame to grayscale (required for corner detection)
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Try to find the chessboard corners in the grayscale image
    # ret, corners = cv2.findChessboardCorners(gray_frame, chessboard_size, None)

    # # If corners are found, draw them on the frame
    # if ret:
    #     print("Found chessboard")
    #     # Refine the corner positions (optional, but improves accuracy)
    #     corners = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1),
    #                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
    #     # Draw the corners on the original frame
    #     cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Define a file name (e.g., chessboard.png)
        file_name = f'chessboard_frame{frame_number}.png'
        frame_number += 1
        
        # Save the current frame
        cv2.imwrite(file_name, frame)
        print(f"Frame saved as {file_name}")
    # Display the resulting frame
    # cv2.imshow('Webcam Video', frame)

    # current_focus = cap.get(cv2.CAP_PROP_FOCUS)
    # print(f"Current focus: {current_focus}")

    # Press 'q' to quit the video stream
    if key == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()