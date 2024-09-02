"""This module contains the CubeTracker class, which is receives a stream of images and
returns the pose of the cube in the image."""

import time
import cv2
import numpy as np
import attrs

from ube.data_stream.data_scoring import score_poses
from ube.data_stream.data_stream_config import DataStreamConfig
from ube.data_stream.pose_timeseries import PoseTimeseries
from ube.data_stream.utils.constants import CubeID, MarkerPosition


@attrs.define
class CubeTracker:
    """Tracks the cube in the image"""

    id: CubeID = attrs.field()
    """The ID of the cube"""

    config: DataStreamConfig = attrs.field()
    """The configuration of the data stream"""

    pose_time_series: PoseTimeseries = attrs.field()
    """The timeseries of poses of the cube"""

    def get_pose(self, frame: cv2.typing.MatLike) -> np.ndarray:
        """
        Returns the position and orientation of the cube in the image

        Args:
            image (cv2.typing.MatLike): The input image containing the cube

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the cube
        """
        # Loading new position observations
        corners, aruco_ids, _ = cv2.aruco.detectMarkers(
            frame, self.config.aruco_dict, parameters=self.config.aruco_params
        )

        current_time = time.time()
        for aruco_id in aruco_ids:
            if aruco_id not in self.config.cube_marker_map[self.id].keys():
                continue

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[aruco_id],
                self.config.marker_size,
                self.config.camera_matrix,
                self.config.dist_coeffs,
            )

            pose_vec = self.get_cube_pose_vector(aruco_id, current_time, tvec, rvec)
            self.pose_time_series.add_pose(id=aruco_id, pose=pose_vec)

        # Computing the pose of the cube
        fresh_ids = self.pose_time_series.fresh_pose_ids(
            time_threshold=self.config.fresh_pose_time_threshold
        )
        fresh_poses = self.pose_time_series.get_fresh_poses(ids=fresh_ids)[:, 1:]
        score_vector = score_poses(self.pose_time_series, fresh_ids)

        calculated_pose = fresh_poses

    def get_marker_to_center_vector(self, marker_pos: MarkerPosition) -> np.ndarray:
        """
        Returns the vector from the marker to the center of the cube

        Args:
            marker_pos (MarkerPosition): The position of the marker
            rvec (np.ndarray): The rotation vector
        """
        half_size = self.config.cube_side_length / 2
        marker_to_center = {
            MarkerPosition.BACK: np.array([0, 0, half_size]),
            MarkerPosition.RIGHT: np.array([-half_size, 0, 0]),
            MarkerPosition.FRONT: np.array([0, 0, -half_size]),
            MarkerPosition.LEFT: np.array([half_size, 0, 0]),
            MarkerPosition.TOP: np.array([0, -half_size, 0]),
        }

        return marker_to_center[marker_pos]

    def get_cube_pose_vector(
        self, id: int, current_time: float, tvec: np.ndarray, rvec: np.ndarray
    ) -> np.ndarray:
        """
        Returns the pose vector of the cube, using the cube's dimensions and the marker's position
        to find the pose of the cube about its center.

        Args:
            id (int): The marker ID
            current_time (float): The current time
            tvec (np.ndarray): The translation vector
            rvec (np.ndarray): The rotation vector

        Returns:
            np.ndarray: The pose vector of the cube
        """
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        marker_pos = self.config.cube_marker_map[self.id][id]

        # Define the vector from the marker to the center of the cube
        half_size = self.config.cube_side_length / 2
        marker_to_center = {
            MarkerPosition.BACK: np.array([0, 0, half_size]),
            MarkerPosition.RIGHT: np.array([-half_size, 0, 0]),
            MarkerPosition.FRONT: np.array([0, 0, -half_size]),
            MarkerPosition.LEFT: np.array([half_size, 0, 0]),
            MarkerPosition.TOP: np.array([0, -half_size, 0]),
        }

        # Rotate the marker_to_center vector and add accordingly
        rotated_vector = R.dot(marker_to_center[marker_pos])
        cube_center = tvec + rotated_vector
        pose_vector = np.concatenate([[current_time], cube_center, rvec])

        return pose_vector

    def score_adjusted_tvec(self, tvecs: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        """
        Returns the adjusted translation vector of the cube

        Args:
            tvecs (np.ndarray): The translation vectors with dimensions (n, 3)
            score_vector (np.ndarray): The normalized score vector with dimensions (n)

        Returns:
            np.ndarray: The adjusted translation vector with dimensions (3)
        """
        assert np.isclose(np.sum(score_vector), 1)

        return tvecs.T @ score_vector
