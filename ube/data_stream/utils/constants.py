"""Constants for the data stream"""

import enum


class CubeID(enum.Enum):
    """Cube IDs, currently limited to 4"""

    CUBE_0 = enum.auto()
    CUBE_1 = enum.auto()
    CUBE_2 = enum.auto()
    CUBE_3 = enum.auto()


class MarkerPosition(enum.Enum):
    """Marker Positions (from the perspective of the camera)"""

    BACK = enum.auto()
    """The side of the cube that is closest to the camera"""
    RIGHT = enum.auto()
    """The side of the cube that is to the right of the camera"""
    FRONT = enum.auto()
    """The side of the cube that is furthest from the camera"""
    LEFT = enum.auto()
    """The side of the cube that is to the left of the camera"""
    TOP = enum.auto()
    """
    The top of the cube, such that the marker points towards the camera if the cube is angled
    upwards
    """
