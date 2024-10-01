# -*-coding:utf-8-*-
import numpy as np
from toolkit.spot.dspot import dspot
from toolkit.spot.pot import pot
from toolkit.spot.spot import spot
import time
import signal


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout!")


def find_threshold(training_anomaly_scores: np.ndarray, testing_anomaly_scores: np.ndarray, test_labels: np.ndarray,
                   subsequence_length: int):
    """
    Find the threshold for anomaly detection based on the training data.
    Args:
        @param training_anomaly_scores: Anomaly scores of training data.
        @param testing_anomaly_scores: Anomaly scores of testing data.
        @param test_labels: Labels of testing data.
        @param subsequence_length: The init ial length for the POT algorithm.
    """
    std_result = np.zeros(testing_anomaly_scores.shape)
    mean, std = np.mean(training_anomaly_scores), np.std(training_anomaly_scores)
    std_threshold = mean + 3 * std
    std_result[testing_anomaly_scores > std_threshold] = 1
    std_threshold = np.array([std_threshold] * len(testing_anomaly_scores))

    mad_result = np.zeros(testing_anomaly_scores.shape)
    median = np.median(training_anomaly_scores)
    mad = 1.4826 * np.median(np.abs(training_anomaly_scores - median))
    mad_threshold = median + 3 * mad
    mad_result[testing_anomaly_scores > mad_threshold] = 1
    mad_threshold = np.array([mad_threshold] * len(testing_anomaly_scores))

    iqr_result = np.zeros(testing_anomaly_scores.shape)
    q1, q3 = np.percentile(training_anomaly_scores, [25, 75])
    iqr = q3 - q1
    iqr_threshold = q3 + 1.5 * iqr
    iqr_result[testing_anomaly_scores > iqr_threshold] = 1
    iqr_threshold = np.array([iqr_threshold] * len(testing_anomaly_scores))

    top_n_result = np.zeros(testing_anomaly_scores.shape)
    top_n = np.sum(test_labels) / len(test_labels)
    sorted_index = np.argsort(testing_anomaly_scores)
    top_n_index = sorted_index[-int(top_n * len(testing_anomaly_scores)):]
    top_n_result[top_n_index] = 1
    top_n_threshold = testing_anomaly_scores[top_n_index[0]]
    top_n_threshold = np.array([top_n_threshold] * len(testing_anomaly_scores))

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    try:
        pot_result = np.zeros(testing_anomaly_scores.shape)
        pot_threshold, _ = pot(training_anomaly_scores, risk=0.01)
        pot_result[testing_anomaly_scores > pot_threshold] = 1
        pot_threshold = np.array([pot_threshold] * len(testing_anomaly_scores))
    except TimeoutError:
        pot_result = std_result
        pot_threshold = std_threshold
    finally:
        signal.alarm(0)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    try:
        dspot_result = np.zeros(testing_anomaly_scores.shape)
        init_length = 5 * subsequence_length
        try:
            dspot_threshold, anomaly_index = spot(testing_anomaly_scores, init_length, 0.01)
            dspot_result[anomaly_index] = 1
        except ValueError:
            dspot_result = std_result
            dspot_threshold = std_threshold
    except TimeoutError:
        dspot_result = std_result
        dspot_threshold = std_threshold
    finally:
        signal.alarm(0)

    return {
        "std": [std_threshold, std_result],
        "mad": [mad_threshold, mad_result],
        "iqr": [iqr_threshold, iqr_result],
        "pot": [pot_threshold, pot_result],
        "dspot": [dspot_threshold, dspot_result],
        "top_n": [top_n_threshold, top_n_result],
        "anomaly_scores": testing_anomaly_scores
    }
