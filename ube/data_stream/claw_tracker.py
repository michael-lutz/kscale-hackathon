"""
The full stream of claw wrist positions, orientations, and gripper widths.
"""

import time
from typing import Optional
import attrs
import cv2
import numpy as np

from ube.data_stream.cube_tracker import CubeTracker
from ube.data_stream.data_stream_config import DataStreamConfig
from ube.data_stream.tracker_base import Tracker
from ube.data_stream.utils.constants import CubeID


@attrs.define
class ClawTracker(Tracker):
    """Tracks the claw wrist positions, orientations, and gripper widths"""

    config: DataStreamConfig = attrs.field()
    """The configuration of the data stream"""

    left_cube_tracker: CubeTracker = attrs.field(init=False)
    """The timeseries of poses of the left cube"""

    right_cube_tracker: CubeTracker = attrs.field(init=False)
    """The timeseries of poses of the right claw"""

    def __attrs_post_init__(self):
        """Initialize cube trackers using the provided config"""
        self.left_cube_tracker = CubeTracker(id=CubeID.CUBE_0, config=self.config)
        self.right_cube_tracker = CubeTracker(id=CubeID.CUBE_1, config=self.config)

    def get_pose(self, frame: cv2.typing.MatLike) -> Optional[np.ndarray]:
        """
        Returns the position and orientation of the cube in the image

        Args:
            image (cv2.typing.MatLike): The input image containing the cube

        Returns:
            np.ndarray: A 8-dimensional vector representing: (t, x, y, z, rx, ry, rz, w)
        """
        # get poses of the left and right cubes
        left_cube_pose = self.left_cube_tracker.get_pose(frame)
        right_cube_pose = self.right_cube_tracker.get_pose(frame)

        if left_cube_pose is None or right_cube_pose is None:
            return None

        print("Left cube pose:", left_cube_pose)
        print("Right cube pose:", right_cube_pose)
        # get the wrist and gripper width of the left and right cubes
        wrist_pose = (left_cube_pose[1:4] + right_cube_pose[1:4]) / 2
        gripper_width = np.linalg.norm(right_cube_pose[1:4] - left_cube_pose[1:4])

        # find the orientation by taking the average of the two orientations
        # need to convert to rotation matrix first
        R_l = cv2.Rodrigues(left_cube_pose[4:7])[0]
        R_r = cv2.Rodrigues(right_cube_pose[4:7])[0]
        R_wrist = (R_l + R_r) / 2
        wrist_orientation = cv2.Rodrigues(R_wrist)[0].flatten()

        current_time = time.time()
        return np.concatenate([[current_time], wrist_pose, wrist_orientation, [gripper_width]])
