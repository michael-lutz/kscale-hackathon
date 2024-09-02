"""Quick timeseries representation of poses"""

import time
import attrs
import numpy as np

from collections import deque
from typing import Dict, Iterator, List, Optional


@attrs.define
class PoseTimeseries:
    """A timeseries of observed poses from ArUco markers"""

    poses: Dict[int, deque[np.ndarray]] = attrs.field(factory=dict)
    """The poses in the timeseries. Each pose is a 7-dimensional vector in the form of
    [time, x, y, z, rx, ry, rz]. Assume that the ids are consecutive starting from 0.
    
    NOTE: The predicted pose has protected id of -1.
    """

    max_poses: int = attrs.field(default=100)
    """The maximum number of poses to store per marker"""

    def add_pose(self, id: int, pose: np.ndarray) -> None:
        """
        Add a pose to the deque and cycle accordingly

        Args:
            pose (np.ndarray): The pose to add to the deque.
            id (int): The id of the marker to add the pose to.
        """
        if id not in self.poses:
            self.poses[id] = deque(maxlen=self.max_poses)

        self.poses[id].append(pose)

    def get_poses(self, ids: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Get the poses from the timeseries in order of ids provided.

        Args:
            ids (Optional[List[int]]): The ids of the markers to get the poses for.
                If None, all markers will be returned.

        Returns:
            List[np.ndarray]: List of n arrays with dimensions (t, 7) where n is the number of ids
                specified and t is the number of poses for each id.
        """
        if ids is None:
            ids = list(self.poses.keys())

        poses = []
        for id in ids:
            pose_arr = np.array(self.poses[id])
            poses.append(pose_arr)

        return poses

    def get_fresh_poses(self, ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the most recent pose matrix from the timeseries in order of ids provided.

        Args:
            ids (Optional[List[int]]): The ids of the markers to get the latest poses for.
                If None, all markers will be returned.

        Returns:
            np.ndarray: Array with dimensions (n, 7) where n is the number of ids specified.
        """
        if ids is None:
            ids = list(self.poses.keys())

        recent_poses = []
        for id in ids:
            recent_poses.append(self.poses[id][-1])
        return np.array(recent_poses)

    def fresh_pose_ids(self, time_threshold: float) -> List[int]:
        """
        Get the ids of non-stale markers

        Args:
            time_threshold (float): The maximum age of poses to consider, in ms.

        Returns:
            List[int]: The ids of the non-stale markers.
        """
        current_time = time.time()
        recent_pose_ids = []
        oldest_acceptable_time = current_time - time_threshold / 1000

        for pose_id, pose_deque in self.poses.items():
            if pose_deque[-1][0] > oldest_acceptable_time:
                recent_pose_ids.append(pose_id)

        return recent_pose_ids

    def __getitem__(self, id: int) -> deque[np.ndarray]:
        return self.poses[id]

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self) -> Iterator[deque[np.ndarray]]:
        for pose in self.poses.values():
            yield pose
