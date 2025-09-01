from pose_estimator_icam import PoseEstimator
import time
import numpy as np


if __name__ == '__main__':

    # Initialize the PoseEstimator
    pose_estimator = PoseEstimator()

    poses = []

    try:
        while True:
            # Get the current pose
            sp = time.time()
            pose, valid, new = pose_estimator.get_pose()

            if not new:
                time.sleep(0.0001)
                continue
            print(f"Pose time: {time.time()-sp}")

            if valid:
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
        # np.save("standard_5_points_bin_blob.npy", poses)

    finally:
        # Ensure resources are released
        pose_estimator.cleanup()
        del pose_estimator