import cv2
import numpy as np
import mvsdk

class ICamera:
    def __init__(self, exposure=0.3):
        """
        Initialize the camera, set parameters, and prepare for frame grabbing.
        """
        # Enumerate connected camera devices
        dev_list = mvsdk.CameraEnumerateDevice()
        if not dev_list:
            raise Exception("No camera was found!")

        dev_info = dev_list[0]

        # Initialize the first camera found
        try:
            self.h_camera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as e:
            raise Exception(f"CameraInit Failed({e.error_code}): {e.message}")

        # Retrieve camera capabilities
        self.cap = mvsdk.CameraGetCapability(self.h_camera)

        # Configure camera settings
        mvsdk.CameraSetIspOutFormat(self.h_camera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        mvsdk.CameraSetTriggerMode(self.h_camera, 0)
        mvsdk.CameraSetAeState(self.h_camera, 0)
        mvsdk.CameraSetExposureTime(self.h_camera, exposure * 1000)  # Exposure time in microseconds
        mvsdk.CameraSetAnalogGain(self.h_camera, 2)
        mvsdk.CameraSetAnalogGainX(self.h_camera, 2.0)

        # Start camera streaming
        mvsdk.CameraPlay(self.h_camera)

        # Allocate memory for frame buffer
        frame_buffer_size = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax
        self.p_frame_buffer = mvsdk.CameraAlignMalloc(frame_buffer_size, 16)

    def grab(self):
        """
        Capture a single frame from the camera.

        Returns:
            frame (np.ndarray): The captured frame as a NumPy array in grayscale.
                                 Returns None if capturing fails or times out.
        """
        try:
            # Acquire image buffer with a timeout of 200ms
            p_raw_data, frame_head = mvsdk.CameraGetImageBuffer(self.h_camera, 200)

            # Process the raw image data
            mvsdk.CameraImageProcess(self.h_camera, p_raw_data, self.p_frame_buffer, frame_head)

            # Release the image buffer back to the camera
            mvsdk.CameraReleaseImageBuffer(self.h_camera, p_raw_data)

            # Optionally flip the image if needed (1 for horizontal flip)
            mvsdk.CameraFlipFrameBuffer(self.p_frame_buffer, frame_head, 1)

            # Convert the processed buffer to a NumPy array
            frame_data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self.p_frame_buffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((frame_head.iHeight, frame_head.iWidth, 1))  # Grayscale

            return frame

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")
            return None

    def cleanup(self):
        """
        Release camera resources and clean up allocated memory.
        """
        mvsdk.CameraUnInit(self.h_camera)
        mvsdk.CameraAlignFree(self.p_frame_buffer)

# Example Usage
if __name__ == "__main__":
    camera = ICamera(30)
    frame_number = 0
    try:
        while True:
            frame = camera.grab()
            if frame is not None:
                cv2.imshow("Camera Frame", frame)
            
            # Exit loop when 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                file_name = f'chessboard_frame_{frame_number}.png'
                frame_number += 1
                
                # Save the current frame
                cv2.imwrite(file_name, frame)
                print(f"Frame saved as {file_name}.")
    finally:
        camera.cleanup()
        cv2.destroyAllWindows()
