import cv2
import numpy as np

from ube.data_stream.cube_tracker import CubeTracker
from ube.data_stream.data_stream_config import DataStreamConfig
from ube.data_stream.utils.constants import CubeID

# Initialize the camera
cap = cv2.VideoCapture(0)
config = DataStreamConfig()
tracker = CubeTracker(id=CubeID.CUBE_0, config=DataStreamConfig())


def draw_pose_on_frame(frame, pose, camera_matrix, dist_coeffs, axis_length=0.1):
    """
    Draws the pose on the frame.

    Args:
        frame (np.ndarray): The image frame.
        pose (np.ndarray): The pose array [time, x, y, z, rx, ry, rz].
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        axis_length (float): The length of the axes to draw.
    """
    # Extract translation and rotation vectors from the pose
    tvec = pose[1:4].reshape((3, 1))
    rvec = pose[4:7].reshape((3, 1))

    # Draw the axes on the frame
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    pose = tracker.get_pose(frame)
    print(pose)

    # Example usage
    camera_matrix = config.camera_matrix
    dist_coeffs = config.dist_coeffs

    if pose is not None:
        # Draw the pose on the frame
        frame = draw_pose_on_frame(frame, pose, camera_matrix, dist_coeffs)

    # Display the frame
    cv2.imshow("Pose Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
