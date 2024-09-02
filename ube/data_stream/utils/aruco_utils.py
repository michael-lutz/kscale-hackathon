"""Utility functions for ArUco marker detection via opencv"""
from typing import List, Tuple
import cv2
import numpy as np

def detect_aruco_markers(
    frame: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    aruco_params: cv2.aruco.DetectorParameters,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size: float
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Detect ArUco markers in an image.

    Args:
        frame (np.ndarray): The input image.
        aruco_dict (cv2.aruco_Dictionary): The ArUco dictionary.
        aruco_params (cv2.aruco_DetectorParameters): The detector
            parameters.
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        marker_size (float): The size of the markers.

    Returns:
        List[Tuple[int, np.ndarray, np.ndarray]]: A list of detected
        markers with their IDs, positions, and orientations.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None:
        return []

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_size,
        camera_matrix,
        dist_coeffs
    )
    
    return [(ids[i][0], tvecs[i][0], rvecs[i][0]) for i in range(len(ids))]

def draw_cube_axes(
    frame: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float
) -> np.ndarray:
    """
    Draw the axes of a cube on the image.

    Args:
        frame (np.ndarray): The input image.
        position (np.ndarray): The position of the cube.
        orientation (np.ndarray): The orientation of the cube.
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        axis_length (float): The length of the axes.

    Returns:
        np.ndarray: The image with the cube axes drawn on it.
    """
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, orientation, position, axis_length)
    return frame
