"""This module contains the CubeTracker class, which is receives a stream of images and
returns the pose of the cube in the image."""

import time
from typing import Optional
import cv2
import numpy as np
import attrs

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

    pose_time_series: PoseTimeseries = attrs.field(factory=PoseTimeseries, init=False)
    """The timeseries of poses of the cube"""

    def get_pose(self, frame: cv2.typing.MatLike) -> Optional[np.ndarray]:
        """
        Returns the position and orientation of the cube in the image

        Args:
            image (cv2.typing.MatLike): The input image containing the cube

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the pose of the cube
        """
        # loading new position observations
        corners, aruco_ids, _ = cv2.aruco.detectMarkers(
            frame, self.config.aruco_dict, parameters=self.config.aruco_params
        )

        if aruco_ids is None:
            return None

        # adding the new observations to the pose time series
        current_time = time.time()
        for index, aruco_id in enumerate(aruco_ids):
            aruco_id = aruco_id[0]
            if aruco_id not in list(self.config.cube_marker_map[self.id].keys()):
                continue
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[index],
                self.config.marker_size,
                self.config.camera_matrix,
                self.config.dist_coeffs,
            )
            rvec = rvec[0][0]
            tvec = tvec[0][0]
            pose_vec = self.get_cube_pose_vector(aruco_id, current_time, tvec, rvec)
            self.pose_time_series.add_pose(id=aruco_id, pose=pose_vec)

        # Computing the pose of the cube after adjusting for quality
        fresh_ids = self.pose_time_series.fresh_pose_ids(
            time_threshold=self.config.fresh_pose_time_threshold
        )
        t_fresh_poses = self.pose_time_series.get_fresh_poses(ids=fresh_ids)
        if t_fresh_poses is None or len(t_fresh_poses) == 0:
            return None

        return t_fresh_poses[0]

        # fresh_poses = t_fresh_poses[:, 1:]
        # score_vector = score_poses(self.pose_time_series, fresh_ids)
        # tvec = self.score_adjusted_tvec(fresh_poses[:, :3], score_vector)
        # rvec = self.score_adjusted_rvec(fresh_poses[:, 3:], score_vector)
        ## TODO: apply kalman filter based on un-normalized score values
        # calculated_pose = np.concatenate([[current_time], tvec, rvec], axis=0)

        # return calculated_pose

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
        # getting the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(rvec)
        marker_pos = self.config.cube_marker_map[self.id][id]
        # rotating the marker_to_center vector and add accordingly
        half_size = self.config.cube_side_length / 2
        marker_to_center = {
            MarkerPosition.BACK: np.array([0, 0, half_size]),
            MarkerPosition.RIGHT: np.array([-half_size, 0, 0]),
            MarkerPosition.FRONT: np.array([0, 0, -half_size]),
            MarkerPosition.LEFT: np.array([half_size, 0, 0]),
            MarkerPosition.TOP: np.array([0, -half_size, 0]),
        }
        rotated_vector = R.dot(marker_to_center[marker_pos])
        cube_center = tvec  # + rotated_vector # TODO: readd...
        print("OLD", tvec)
        print("NEW", cube_center)
        pose_vector = np.concatenate([[current_time], cube_center, rvec], axis=0)

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

    def score_adjusted_rvec(self, rvecs: np.ndarray, score_vector: np.ndarray) -> np.ndarray:
        """
        Returns the adjusted rotation vector of the cube

        Args:
            rvecs (np.ndarray): The rotation vectors with dimensions (n, 3)
            score_vector (np.ndarray): The normalized score vector with dimensions (n)

        Returns:
            np.ndarray: The adjusted rotation vector with dimensions (3)
        """
        assert np.isclose(np.sum(score_vector), 1)
        rotation_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
        weighted_rotation_matrices = [w * R for w, R in zip(score_vector, rotation_matrices)]
        adjusted_rotation_matrix = np.sum(weighted_rotation_matrices, axis=0)
        averaged_rvec, _ = cv2.Rodrigues(adjusted_rotation_matrix)

        return averaged_rvec.flatten()
