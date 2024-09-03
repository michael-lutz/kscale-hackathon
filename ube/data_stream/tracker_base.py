"""
Base class for all trackers
"""

from typing import Optional
import attrs

from abc import ABC, abstractmethod

import cv2
import numpy as np


@attrs.define
class Tracker(ABC):
    """
    Base class for all trackers
    """

    @abstractmethod
    def get_pose(self, frame: cv2.typing.MatLike) -> Optional[np.ndarray]:
        """
        Returns the position and orientation of the cube in the image

        Args:
            image (cv2.typing.MatLike): The input image containing the cube
        """
