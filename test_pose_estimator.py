from pose_estimator import PoseEstimator
import time

# Initialize the PoseEstimator
pose_estimator = PoseEstimator()

try:
    while True:
        # Get the current pose
        sp = time.time()
        pose = pose_estimator.get_pose()
        print(f"Pose time: {time.time()-sp}")

        if pose is not None:
            # Process the pose matrix as needed
            # For example, print it or use it in further computations
            print("Pose Matrix:\n", pose)
        else:
            # Handle the case where the pose could not be determined
            print("Pose not detected.")

        # Implement any additional logic or break conditions as needed

except KeyboardInterrupt:
    # Gracefully handle termination (e.g., via Ctrl+C)
    pass

finally:
    # Ensure resources are released
    del pose_estimator