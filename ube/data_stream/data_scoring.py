"""Contains the CubeTracker class and its tracking logic"""

from typing import Any, List
import numpy as np

from ube.data_stream.pose_timeseries import PoseTimeseries


def check_valid_input_for_scoring(func):
    """Decorator to check if the input is valid"""

    def wrapper(pose_time_series: Any, *args, **kwargs):
        if pose_time_series is None or not isinstance(pose_time_series, PoseTimeseries):
            raise ValueError("Must pass in pose timeseries to score.")
        if not pose_time_series.poses:
            raise ValueError("Must pass in pose timeseries with poses to score.")
        return func(pose_time_series, *args, **kwargs)

    return wrapper


def override_agreement_scoring(func):
    """Decorator to override the agreement scoring"""

    def wrapper(pose_time_series: PoseTimeseries, pose_ids: List[int], *args, **kwargs):
        if pose_time_series.get_fresh_poses(pose_ids).shape[0] == 1:
            return np.ones(len(pose_ids))
        return func(pose_time_series, pose_ids, *args, **kwargs)

    return wrapper


@check_valid_input_for_scoring
@override_agreement_scoring
def score_euclidian_distance_agreement(
    pose_time_series: PoseTimeseries, ids_to_score: List[int]
) -> np.ndarray:
    """
    Returns an array of scores representing agremeent between poses.

    This is ultimately calculated by first finding a pairwise agreement matrix, which is an
    NxN matrix A where A[i, j] is the exponential decay of the euclidean distance between
    pose i and pose j. The score is the sum of the agreement matrix, normalized by the
    number of poses. As such, the score ranges from 0 to 1, with 1 indicating perfect
    agreement between all poses.


    Args:
        pose_time_series (PoseTimeseries): A timeseries of poses, where each pose is an array
            in the form of [time, x, y, z, rx, ry, rz]. This is expected to be a dictionary
            mapping marker IDs to deques of pose arrays.
        ids_to_score (List[int]): A list of marker IDs to score.
        time_threshold (float): The maximum age of poses to consider, in ms.

    Returns:
        np.ndarray: The euclidian distance agreement score vector
    """

    poses = pose_time_series.get_fresh_poses(ids_to_score)
    poses = poses[:, 1:3]  # isolating for position
    # Calculating the euclidean distance matrix
    distance_matrix = np.linalg.norm(poses[:, np.newaxis, :] - poses[np.newaxis, :, :], axis=2)
    # Doing an exponential decay to weight more similar poses more heavily (and a natural way to
    # ensure that the score is between 0 and 1)
    agreement_matrix = np.exp(-distance_matrix)
    # Subtracting the diagonal to remove self-similarity and normalize to 0-1
    np.fill_diagonal(agreement_matrix, 0)
    score_vector = np.sum(agreement_matrix, axis=1)
    score_vector /= len(poses) - 1
    return score_vector


@check_valid_input_for_scoring
@override_agreement_scoring
def score_cosine_similarity_agreement(
    pose_time_series: PoseTimeseries, ids_to_score: List[int]
) -> np.ndarray:
    """
    Returns an array of scores representing agremeent between poses.

    This is ultimately calculated by first finding a pairwise agreement matrix, which is an
    NxN matrix A where A[i, j] is the cosine similarity between pose i and pose j. The score
    is the sum of the agreement matrix, normalized by the number of poses. As such, the score
    ranges from 0 to 1, with 1 indicating perfect agreement between all poses.

    Args:
        pose_time_series (PoseTimeseries): A timeseries of poses, where each pose is an array
            in the form of [time, x, y, z, rx, ry, rz]. This is expected to be a dictionary
            mapping marker IDs to deques of pose arrays.
        ids_to_score (List[int]): A list of marker IDs to score.
        time_threshold (float): The maximum age of poses to consider, in ms.

    Returns:
        np.ndarray: The cosine similarity agreement score vector
    """
    poses = pose_time_series.get_fresh_poses(ids_to_score)
    poses = poses[:, 3:]  # isolating for rotation only
    similarity_matrix = np.dot(poses, poses.T)
    similarity_matrix = similarity_matrix / (np.linalg.norm(poses, axis=1) ** 2)
    # Subtracting the diagonal to remove self-similarity and normalize to 0-1
    np.fill_diagonal(similarity_matrix, 0)
    score_vector = np.sum(similarity_matrix, axis=1)
    score_vector /= len(poses) - 1
    return score_vector


@check_valid_input_for_scoring
def score_noisiness(pose_time_series: PoseTimeseries, ids_to_score: List[int]) -> np.ndarray:
    """
    Returns an array of scores representing a measure of noisyness in poses over time.

    Args:
        poses_time_series (PoseTimeseries): A timeseries of poses, where each pose is an array
            in the form of [time, x, y, z, rx, ry, rz]. This is expected to be a dictionary
            mapping marker IDs to deques of pose arrays.
        ids_to_score (List[int]): A list of marker IDs to score.

    Returns:
        np.ndarray: The variance score vector
    """
    poses = pose_time_series.get_poses(ids_to_score)
    poses = [
        pose[:, 1:] for pose in poses
    ]  # isolating for position and orientation (removing time)

    mean_absolute_changes = []
    for pose in poses:
        if pose.shape[0] > 1:
            mean_absolute_change = np.mean(np.abs(np.diff(pose, axis=0)))
        else:
            mean_absolute_change = 0
        mean_absolute_changes.append(mean_absolute_change)

    score_vec = np.array(mean_absolute_changes)
    score_vec = np.exp(-score_vec)
    return score_vec


def score_poses(pose_time_series: PoseTimeseries, ids_to_score: List[int]) -> np.ndarray:
    """
    Aggregates and normalizes scores defined above for a set of ids

    Args:
        pose_time_series (PoseTimeseries): A timeseries of poses, where each pose is an array
            in the form of [time, x, y, z, rx, ry, rz]. This is expected to be a dictionary
            mapping marker IDs to deques of pose arrays.
        ids_to_score (List[int]): A list of marker IDs to score.

    Returns:
        np.ndarray: The aggregated and normalized score vector with dimension n, where n is the
            number of ids to score.
    """
    euclidean_distance_scores = score_euclidian_distance_agreement(pose_time_series, ids_to_score)
    cosine_similarity_scores = score_cosine_similarity_agreement(pose_time_series, ids_to_score)
    noisiness_scores = score_noisiness(pose_time_series, ids_to_score)

    # Aggregating the scores
    aggregated_scores = np.sum(
        [euclidean_distance_scores, cosine_similarity_scores, noisiness_scores], axis=0
    )
    aggregated_scores = aggregated_scores / np.sum(aggregated_scores)
    return aggregated_scores
