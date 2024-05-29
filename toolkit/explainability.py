from tslearn.metrics import dtw
from sktime.distances import erp_distance
import numpy as np
from toolkit.result_plot import get_segments


def calculate_scores(target: np.ndarray, reconstructed: np.ndarray):
    """
    Computes the DTW distance between the target and reconstructed time series.
    :param target: the target time series: shape (n_timesteps, n_features)
    :param reconstructed: the reconstructed time series (same shape as target)
    """

    dtw_score = dtw(target, reconstructed)
    euclidean_score = np.sqrt(((target - reconstructed) ** 2).sum())
    erp_score = 0
    for i in range(target.shape[1]):
        erp_score += erp_distance(target[:, i], reconstructed[:, i])
    erp_score /= target.shape[1]
    return dtw_score, euclidean_score, erp_score


def explainable_scores(target: np.ndarray, reconstructed: np.ndarray, labels: np.ndarray) -> dict:
    anomaly_segments = get_segments(labels)
    dtw_scores = []
    euclidean_scores = []
    erp_scores = []
    for segment in anomaly_segments:
        if segment[0] == segment[1]:
            segment = (segment[0], segment[1] + 1)
        current_target = target[segment[0]:segment[1], :]
        current_reconstructed = reconstructed[segment[0]:segment[1], :]
        dtw_score, euclidean_score, erp_score = calculate_scores(current_target, current_reconstructed)
        dtw_scores.append(dtw_score)
        euclidean_scores.append(euclidean_score)
        erp_scores.append(erp_score)

    score = {
        "dtw": float(np.mean(dtw_scores)),
        "euclidean": float(np.mean(euclidean_scores)),
        "erp": float(np.mean(erp_scores))
    }
    return score


if __name__ == '__main__':
    a_0 = np.sin(np.arange(100))
    b_0 = np.sin(np.arange(100)) + np.random.normal(0, 0.1, 100)
    a = a_0.reshape(20, 5)
    b = b_0.reshape(20, 5)
    dtw_score, euclidean_score, erp_score = explainable_scores(a, b)
    print(f"DTW score: {dtw_score}")
    print(f"Euclidean score: {euclidean_score}")
    print(f"ERP score: {erp_score}")
