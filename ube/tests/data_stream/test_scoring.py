import numpy as np
import pytest
from ube.data_stream.data_scoring import (
    score_euclidian_distance_agreement,
    score_cosine_similarity_agreement,
    score_noisiness,
)
from ube.data_stream.pose_timeseries import PoseTimeseries


def test_score_euclidian_distance_agreement():
    # Test case 1: Perfect agreement (all markers at the same position)
    poses1 = {
        0: [np.array([0, 0, 0, 0, 0, 0, 0])],
        1: [np.array([0, 0, 0, 0, 0, 0, 0])],
        2: [np.array([0, 0, 0, 0, 0, 0, 0])],
    }
    pose_time_series1 = PoseTimeseries(poses=poses1)
    scores1 = score_euclidian_distance_agreement(pose_time_series1, [0, 1, 2])
    assert np.allclose(scores1, np.ones(3))

    # Test case 2: Some disagreement
    poses2 = {
        0: [np.array([0, 0, 0, 0, 0, 0, 0])],
        1: [np.array([0, 1, 1, 0, 0, 0, 0])],
        2: [np.array([0, 2, 2, 0, 0, 0, 0])],
    }
    pose_time_series2 = PoseTimeseries(poses=poses2)
    scores2 = score_euclidian_distance_agreement(pose_time_series2, [0, 1, 2])
    assert np.all(scores2 < 1)
    assert np.all(scores2 > 0)
    assert scores2[1] > scores2[0] == scores2[2]

    # Test case 3: Some heavy agreement
    poses3 = {
        0: [np.array([0, 0, 0, 0, 0, 0, 0])],
        1: [np.array([0, 0, 0, 0, 0, 0, 0])],
        2: [np.array([0, 2, 2, 0, 0, 0, 0])],
    }
    pose_time_series3 = PoseTimeseries(poses=poses3)
    scores3 = score_euclidian_distance_agreement(pose_time_series3, [0, 1, 2])
    assert np.all(scores3 < 1)
    assert np.all(scores3 > 0)
    assert scores3[1] == scores3[0] > scores3[2]

    # Test case 4: Single marker
    poses4 = {0: [np.array([0, 1, 2, 3, 4, 5, 6])]}
    pose_time_series4 = PoseTimeseries(poses=poses4)
    scores4 = score_euclidian_distance_agreement(pose_time_series4, [0])
    assert np.allclose(scores4, np.ones(1))

    # Test case 5: Empty input
    poses5 = {}
    pose_time_series5 = PoseTimeseries(poses=poses5)
    with pytest.raises(ValueError):
        score_euclidian_distance_agreement(pose_time_series5, [])


def test_score_cosine_similarity_agreement():
    # Test case 1: Perfect agreement (all rotations identical)
    poses1 = {
        0: [np.array([0, 0, 0, 0, 1, 0, 0])],
        1: [np.array([0, 0, 0, 0, 1, 0, 0])],
        2: [np.array([0, 0, 0, 0, 1, 0, 0])],
    }
    pose_time_series1 = PoseTimeseries(poses=poses1)
    scores1 = score_cosine_similarity_agreement(pose_time_series1, [0, 1, 2])
    assert np.allclose(scores1, np.ones(3))

    # Test case 2: Some disagreement
    poses2 = {
        0: [np.array([0, 0, 0, 0, 1, 0, 1])],
        1: [np.array([0, 0, 0, 0, 1, 1, 0])],
        2: [np.array([0, 0, 0, 0, 0, 1, 1])],
    }
    pose_time_series2 = PoseTimeseries(poses=poses2)
    scores2 = score_cosine_similarity_agreement(pose_time_series2, [0, 1, 2])
    assert np.all(scores2 < 1)
    assert np.all(scores2 > 0)
    assert np.allclose(scores2, scores2[0])  # All scores should be equal due to symmetry

    # Test case 3: Opposite rotations
    poses3 = {0: [np.array([0, 0, 0, 0, 1, 0, 0])], 1: [np.array([0, 0, 0, 0, 0, 1, 0])]}
    pose_time_series3 = PoseTimeseries(poses=poses3)
    scores3 = score_cosine_similarity_agreement(pose_time_series3, [0, 1])
    assert np.allclose(scores3, np.zeros(2))

    # Test case 4: Single pose
    poses4 = {0: [np.array([0, 1, 0, 0, 0, 0, 0])]}
    pose_time_series4 = PoseTimeseries(poses=poses4)
    scores4 = score_cosine_similarity_agreement(pose_time_series4, [0])
    assert np.allclose(scores4, np.ones(1))

    # Test case 5: Empty input
    poses5 = {}
    pose_time_series5 = PoseTimeseries(poses=poses5)
    with pytest.raises(ValueError):
        score_cosine_similarity_agreement(pose_time_series5, [])


def test_score_noisiness():
    # Test case 1: No noise (constant poses)
    poses1 = {
        0: [np.array([t / 10, 0, 0, 0, 0, 0, 0]) for t in range(10)],
    }
    pose_time_series1 = PoseTimeseries(poses=poses1)
    scores1 = score_noisiness(pose_time_series1, [0])
    assert np.allclose(scores1, np.ones(1))

    # Test case 2: Increasing noise
    poses2 = {
        0: [
            np.array([t / 10, 0, 0, 0, 0, 0, 0]) + np.random.normal(0, 0.1 * t / 10, 7)
            for t in range(10)
        ],
    }
    pose_time_series2 = PoseTimeseries(poses=poses2)
    scores2 = score_noisiness(pose_time_series2, [0])
    assert scores2[0] > 0

    # Test case 3: Different noise levels for multiple markers
    poses3 = {
        0: [np.array([t / 10, 0, 0, 0, 0, 0, 0]) + np.random.normal(0, 0.1, 7) for t in range(10)],
        1: [np.array([t / 10, 0, 0, 0, 0, 0, 0]) + np.random.normal(0, 0.2, 7) for t in range(10)],
        2: [np.array([t / 10, 0, 0, 0, 0, 0, 0]) + np.random.normal(0, 0.3, 7) for t in range(10)],
    }
    pose_time_series3 = PoseTimeseries(poses=poses3)
    scores3 = score_noisiness(pose_time_series3, [0, 1, 2])
    assert scores3[0] > scores3[1] > scores3[2]

    # Test case 4: Single frame (should return zero noisiness)
    poses4 = {
        0: [np.array([0, 1, 2, 3, 4, 5, 6])],
    }
    pose_time_series4 = PoseTimeseries(poses=poses4)
    scores4 = score_noisiness(pose_time_series4, [0])
    assert np.allclose(scores4, np.ones(1))

    # Test case 5: Empty input
    poses5 = {}
    pose_time_series5 = PoseTimeseries(poses=poses5)
    with pytest.raises(ValueError):
        score_noisiness(pose_time_series5, [])

    # Test case 6: Noisiness in rotation vs position
    poses6 = {
        0: [
            np.array([t / 10, 0, 0, 0, 1, 0, 0])
            + np.concatenate([[0], np.random.normal(0, 0.1, 3), np.zeros(3)])
            for t in range(10)
        ],
        1: [
            np.array([t / 10, 0, 0, 0, 1, 0, 0])
            + np.concatenate([[0], np.zeros(3), np.random.normal(0, 0.1, 3)])
            for t in range(10)
        ],
    }
    pose_time_series6 = PoseTimeseries(poses=poses6)
    scores6 = score_noisiness(pose_time_series6, [0, 1])
    assert scores6[0] != scores6[1]  # Scores should be different for position vs rotation noise
