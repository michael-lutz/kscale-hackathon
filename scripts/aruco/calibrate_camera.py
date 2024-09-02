"""
This script is used to calibrate a camera using images of a chessboard pattern.

Instructions:
0. Print the chessboard pattern on a white paper and ensure each square is 1 inch by 1 inch.
1. Run the script: `python calibrate_camera.py`
2. Press 'c' to capture an image, 'q' to quit
3. After capturing 20 images, the script will automatically calibrate and print the results. Ensure
   the chessboard is in focus and the images are clear. Get as many possible angles and backgrounds
   as possible.
4. Update the config with the camera matrix and distortion coefficients.

"""

import cv2
import numpy as np
import glob
import os


def calibrate_camera(images_path, chessboard_size=(8, 5), square_size=1.0):
    """Calibrates the camera using images of a chessboard pattern.

    Args:
        images_path (str): Path to the directory containing calibration images.
        chessboard_size (tuple): Number of internal corners per a chessboard row and column (columns, rows).
        square_size (float): Size of a square on the chessboard in real-world units (e.g., centimeters).

    Returns:
        ret: RMS re-projection error.
        camera_matrix: Camera matrix.
        dist_coeffs: Distortion coefficients.
        rvecs: Rotation vectors.
        tvecs: Translation vectors.
    """
    # Termination criteria for corner subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0, 0, 0), (1, 0, 0), (2, 0, 0) ...,(6,5,0)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(images_path + "/*.jpg")
    gray = None  # Initialize gray to None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if gray is None:
        raise ValueError("No valid images found for calibration.")

    # Camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def capture_calibration_images(output_dir, num_images=20, chessboard_size=(9, 6)):
    """Captures calibration images using a connected camera when a key is pressed.

    Args:
        output_dir (str): Directory to save captured images.
        num_images (int): Number of images to capture.
        chessboard_size (tuple): Number of internal corners per a chessboard row and column (columns, rows).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    count = 0

    print("Press 'c' to capture an image, 'q' to quit")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                count += 1
                img_name = os.path.join(output_dir, f"calib_{count}.jpg")
                cv2.imwrite(img_name, frame)
                print(f"Captured {img_name}")

                # Draw and display the corners
                frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(500)
            else:
                print("Chessboard not found. Try again.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < num_images:
        print(f"Capture ended early. Captured {count} images.")


if __name__ == "__main__":
    images_path = "./calibration_images"  # Directory with calibration images
    chessboard_size = (8, 5)  # Number of internal corners in the chessboard (cols, rows)
    square_size = 2.54  # Real-world size of chessboard squares in cm

    capture_calibration_images(images_path, num_images=20, chessboard_size=chessboard_size)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        images_path, chessboard_size, square_size
    )

    print("Camera calibration successful!")
    print(f"RMS re-projection error: {ret}")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)
