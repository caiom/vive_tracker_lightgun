from pose_estimator import PoseEstimator
import time
import numpy as np

# Initialize the PoseEstimator
pose_estimator = PoseEstimator()

poses = []

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
            poses.append(pose)
        else:
            # Handle the case where the pose could not be determined
            print("Pose not detected.")

        # Implement any additional logic or break conditions as needed

except KeyboardInterrupt:
    # Gracefully handle termination (e.g., via Ctrl+C)
    poses = np.stack(poses)
    np.save("short_p5_small_9_exp_pad_2_0_gain.npy", poses)

finally:
    # Ensure resources are released
    del pose_estimator