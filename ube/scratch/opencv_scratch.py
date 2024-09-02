import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Marker size in meters
marker_size = 0.02

# Camera matrix and distortion coefficients (using default values)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)


def ensure_positive_z(rvec, tvec):
    # Ensure Z is always positive
    if tvec[0][0][2] < 0:
        tvec[0][0] = -tvec[0][0]
        rvec[0][0] = -rvec[0][0]
    return rvec, tvec


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    print(ids)
    if ids is not None and 0 in ids:
        # Find index of marker with ID 0
        index = np.where(ids == 0)[0][0]

        # Estimate pose for marker 0
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[index], marker_size, camera_matrix, dist_coeffs
        )

        # Ensure Z is positive
        rvec, tvec = ensure_positive_z(rvec, tvec)

        # Draw axis for marker 0
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # Print position and rotation vector
        print(f"Position (x, y, z): {tvec[0][0]}")
        print(f"Rotation Vector: {rvec[0][0]}")
        print("---")

    # Display the frame
    cv2.imshow("Marker 0 Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
