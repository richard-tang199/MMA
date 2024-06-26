import numpy as np
from .utils.metrics import metricor
from .utils.utility import get_list_anomaly
from .analysis.robustness_eval import generate_curve


def get_range_vus_roc(score, labels):
    grader = metricor()
    slidingWindow = int(np.mean(get_list_anomaly(labels)))
    R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels, score, 2 * slidingWindow)
    metrics = {'R_AUC_ROC': R_AUC_ROC, 'R_AUC_PR': R_AUC_PR, 'VUS_ROC': VUS_ROC, 'VUS_PR': VUS_PR}

    return metrics
