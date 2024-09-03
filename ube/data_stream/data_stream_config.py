"""Configuration for the data stream"""

from typing import Dict
import attrs
import cv2
import numpy as np

from ube.data_stream.utils.constants import CubeID, MarkerPosition


@attrs.define
class DataStreamConfig:
    """
    Configuration for the data stream

    Properties:
        marker_size: The size of the markers in meters
        cube_side_length: The side length of the cube in meters
        aruco_dict: The ArUco dictionary to use for marker detection
        aruco_params: The detector parameters to use for marker detection
        camera_matrix: The camera matrix to use for marker detection
        dist_coeffs: The distortion coefficients to use for marker detection
    """

    marker_size: float = attrs.field(default=0.2)
    """The size of the markers in meters"""

    cube_side_length: float = attrs.field(default=0.25)
    """The side length of the cube in meters"""

    aruco_dict: cv2.aruco.Dictionary = attrs.field(
        default=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    """The ArUco dictionary to use for marker detection"""

    aruco_params: cv2.aruco.DetectorParameters = attrs.field(default=cv2.aruco.DetectorParameters())
    """The detector parameters to use for marker detection"""

    camera_matrix: np.ndarray = attrs.field(
        default=np.array([[1450, 0, 950], [0, 1450, 515], [0, 0, 1]], dtype=np.float32)
    )
    """The camera matrix to use for marker detection"""

    dist_coeffs: np.ndarray = attrs.field(default=np.zeros((4, 1), dtype=np.float32))
    """The distortion coefficients to use for marker detection"""

    cube_marker_map: Dict[CubeID, Dict[int, MarkerPosition]] = {
        CubeID.CUBE_0: {
            0: MarkerPosition.BACK,
            1: MarkerPosition.RIGHT,
            2: MarkerPosition.FRONT,
            3: MarkerPosition.LEFT,
            4: MarkerPosition.TOP,
        },
        CubeID.CUBE_1: {
            5: MarkerPosition.BACK,
            6: MarkerPosition.RIGHT,
            7: MarkerPosition.FRONT,
            8: MarkerPosition.LEFT,
            9: MarkerPosition.TOP,
        },
    }
    """The mapping of cube ids to their marker ids and positions"""

    fresh_pose_time_threshold: float = 100
    """The maximum age of poses to consider, in ms"""

    ewma_alpha: float = 0.6
    """The alpha value for the EWMA filter"""
