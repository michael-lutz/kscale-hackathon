import cv2
import numpy as np
from ube.data_stream.cube_tracker import CubeTracker
from ube.data_stream.data_stream_config import DataStreamConfig
from ube.data_stream.utils.constants import CubeID

# Initialize the camera
cap = cv2.VideoCapture(0)
config = DataStreamConfig()
tracker = CubeTracker(
    id=CubeID.CUBE_0,
    config=DataStreamConfig(
        dist_coeffs=np.array([0.08320202, -0.01125216, -0.00116249, -0.00470709, -0.23390565])
    ),
)

# Define the ArUco dictionary and board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.GridBoard(
    size=(1, 1), markerLength=0.04, markerSeparation=0.01, dictionary=aruco_dict
)


def draw_pose_on_frame(frame, rvec, tvec, camera_matrix, dist_coeffs, axis_length=0.1):
    """
    Draws the pose on the frame.

    Args:
        frame (np.ndarray): The image frame.
        rvec (np.ndarray): The rotation vector.
        tvec (np.ndarray): The translation vector.
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        axis_length (float): The length of the axes to draw.
    """
    # Draw the axes on the frame
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict)

    if ids is not None:
        # Refine detected markers
        cv2.aruco.refineDetectedMarkers(frame, board, corners, ids, rejectedImgPoints)

        # Estimate pose of the board
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board, config.camera_matrix, config.dist_coeffs
        )

        if retval:
            # Draw the pose on the frame
            frame = draw_pose_on_frame(frame, rvec, tvec, config.camera_matrix, config.dist_coeffs)

    # Display the frame
    cv2.imshow("Pose Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
