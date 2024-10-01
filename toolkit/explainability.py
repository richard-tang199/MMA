from tslearn.metrics import dtw
from sktime.distances import erp_distance
import numpy as np
from toolkit.result_plot import get_segments
import stumpy
from dtaidistance.subsequence.dtw import subsequence_alignment
import random

random.seed(42)


def calculate_scores(target: np.ndarray, reconstructed: np.ndarray):
    """
    Computes the DTW distance between the target and reconstructed time series.
    :param target: the target time series: shape (n_timesteps, n_features)
    :param reconstructed: the reconstructed time series (same shape as target)
    """

    dtw_score = 0
    euclidean_score = 0
    erp_score = 0
    for i in range(target.shape[1]):
        erp_score += erp_distance(target[:, i], reconstructed[:, i])
        euclidean_score += np.sqrt(((target[:, i] - reconstructed[:, i]) ** 2).sum())
        dtw_score += dtw(target[:, i], reconstructed[:, i])
    erp_score /= target.shape[1]
    # dtw_score = dtw(target, reconstructed)
    return dtw_score, euclidean_score, erp_score


def global_scores(train_data: np.ndarray, reconstructed: np.ndarray):
    """
    @param train_data: length, dimensions
    @param reconstructed: length, dimensions
    @return:
    """
    train_data = train_data.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    nn_dtw = 0
    nn_euclidean = 0

    for i in range(train_data.shape[1]):
        query = reconstructed[:, i]
        series = train_data[:, i]
        dtw_length = subsequence_alignment(query=query, series=series, use_c=True, penalty=0).matching
        dtw_length = dtw_length * len(query)
        nn_dtw += np.min(dtw_length)
        euclidean = stumpy.mass(Q=query, T=series, normalize=False, p=2.0)
        nn_euclidean += np.min(euclidean)
    return nn_dtw, nn_euclidean


def explainable_scores(target: np.ndarray, reconstructed: np.ndarray,
                       train_dta: np.ndarray, labels: np.ndarray, main_period: int) -> dict:
    """
    @param target: raw test data
    @param reconstructed: reconstruct data
    @param train_data: raw train data
    @param labels: inject anomaly labels
    @return:
    """
    anomaly_segments = get_segments(labels)

    dtw_scores = []
    euclidean_scores = []
    erp_scores = []
    r_dtw_scores = []
    r_euclidean_scores = []
    r_erp_scores = []

    nn_dtw_scores = []
    nn_euclidean_scores = []
    r_nn_dtw_scores = []
    r_nn_euclidean_scores = []

    for segment in anomaly_segments:
        current_segment_length = segment[1] - segment[0]
        if current_segment_length < main_period:
            length_diff = main_period - current_segment_length
            segme
            nt = (segment[0] - length_diff // 2, segment[1] + length_diff // 2)
            if segment[0] < 0:
                segment = (0, main_period)
            if segment[1] > target.shape[0]:
                segment = (target.shape[0] - main_period, target.shape[0])

        # randomly select a segment from the reconstructed data
        random_start = random.randint(0, reconstructed.shape[0] - segment[1] + segment[0])
        random_end = random_start + segment[1] - segment[0]
        random_target = target[random_start:random_end, :]
        random_reconstructed = reconstructed[random_start:random_end, :]

        # anomaly part
        current_target = target[segment[0]:segment[1], :]
        current_reconstructed = reconstructed[segment[0]:segment[1], :]

        dtw_score, euclidean_score, erp_score = calculate_scores(current_target, current_reconstructed)
        r_dtw_score, r_euclidean_score, r_erp_score = calculate_scores(random_target, random_reconstructed)

        nn_dtw, nn_euclidean = global_scores(train_data=train_data,
                                             reconstructed=current_reconstructed)
        r_nn_dtw, r_nn_euclidean = global_scores(train_data=train_data,
                                                 reconstructed=random_reconstructed)

        dtw_scores.append(dtw_score)
        euclidean_scores.append(euclidean_score)
        erp_scores.append(erp_score)
        nn_dtw_scores.append(nn_dtw)
        nn_euclidean_scores.append(nn_euclidean)

        r_dtw_scores.append(r_dtw_score)
        r_euclidean_scores.append(r_euclidean_score)
        r_erp_scores.append(r_erp_score)
        r_nn_dtw_scores.append(r_nn_dtw)
        r_nn_euclidean_scores.append(r_nn_euclidean)

    score = {
        "ED_L": float(np.mean(euclidean_scores)),
        "DTW_L": float(np.mean(dtw_scores)),
        "ED_G": float(np.mean(nn_euclidean_scores)),
        "DTW_G": float(np.mean(nn_dtw_scores)),
        "r_ED_L": float(np.mean(r_euclidean_scores)),
        "r_DTW_L": float(np.mean(r_dtw_scores)),
        "r_ED_G": float(np.mean(r_nn_euclidean_scores)),
        "r_DTW_G": float(np.mean(r_nn_dtw_scores))
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
