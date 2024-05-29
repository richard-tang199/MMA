import numpy as np
from sklearn import metrics
from dataclasses import dataclass
from evaluation.vus.metrics import get_range_vus_roc
from evaluation.affiliation.generics import convert_vector_to_events
from evaluation.affiliation.metrics import pr_from_events


def convert_score_to_label(anomaly_score: np.ndarray, thresholds: np.ndarray):
    """
    @param anomaly_score: np.ndarray
    @param threshold: list of float
    @return: label_list
    """
    label_list = []
    for threshold in thresholds:
        label_list.append(np.where(anomaly_score > threshold, 1, 0))
    return label_list


@dataclass
class EvalResult:
    best_f1_wo_pa: float
    f1_pa_k: float
    best_precision_wo_pa: float
    best_recall_wo_pa: float
    best_threshold_wo_pa: float
    affiliation_precision: float
    affiliation_recall: float
    affiliation_f1: float
    auc_prc: float
    auc_roc: float
    R_AUC_ROC: float
    R_AUC_PR: float
    VUS_ROC: float
    VUS_PR: float
    pa_rate: list
    best_f1_with_pa: list = None
    best_precision_with_pa: list = None
    best_recall_with_pa: list = None
    best_threshold_with_pa: list = None


@dataclass
class EfficiencyResult:
    test_time: float
    flops: float
    params: float


def pak(scores, targets, thres, k=20):
    """

    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """
    scores = np.array(scores)
    thres = np.array(thres)

    predicts = scores > thres
    actuals = targets > 0.01

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts


def evaluate(scores, targets, pa=True, k_list=(0, 25, 50, 75)) -> EvalResult:
    """
    @param pa: whether use point adjustment
    @param targets: ground truth
    @param scores: anomaly score be in [0,1]
    @param interval:
    @type k_list: tuple PA%K threshold: 0 equals to conventional point adjust and 100 equals to original predictions
    @return: EvalResult
    """
    assert len(scores) == len(targets)
    try:
        scores = np.asarray(scores)
        targets = np.asarray(targets)
    except TypeError:
        scores = np.asarray(scores.cpu())
        targets = np.asarray(targets.cpu())

    precision, recall, threshold = metrics.precision_recall_curve(targets, scores)
    threshold = threshold.astype(np.float64)
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)
    auc_prc = metrics.auc(recall, precision)

    # get volume under surface
    vus_metrics = get_range_vus_roc(score=scores, labels=targets)

    # get k point adjustment value
    # go through the k list [15, 30, 50]
    best_f1_pa_list = []
    best_recall_pa_list = []
    best_precision_pa_list = []
    best_threshold_pa_list = []
    f1_pa_k = None

    # determine interval for point adjustment (1000 points)
    # interval (1, threshold_length)
    interval = 10 * (len(threshold) // 1000)
    if interval == 0:
        interval = 10

    if len(threshold) < 10:
        interval = 1

    # TODO: optimize the affilication metrics
    # get affiliation metrics
    if len(scores) // interval < 1:
        ths = threshold
    else:
        ths = [threshold[interval * i] for i in range(len(threshold) // interval)]

    predict_label_list = convert_score_to_label(scores, ths)
    Trange = (0, len(targets))
    events_pred_list = [convert_vector_to_events(predict_label) for predict_label in predict_label_list]
    events_ground_truth = convert_vector_to_events(targets)
    affiliation_metrics_list = [pr_from_events(events_pred, events_ground_truth, Trange)
                                for events_pred in events_pred_list]
    affiliation_precision = np.array([affiliation_metrics["Affiliation_Precision"]
                                      for affiliation_metrics in affiliation_metrics_list])
    affiliation_recall = np.array([affiliation_metrics["Affiliation_Recall"]
                                   for affiliation_metrics in affiliation_metrics_list])
    affiliation_f1 = (2 * affiliation_precision * affiliation_recall /
                      (affiliation_precision + affiliation_recall + 1e-12))

    interval = 10 * len(threshold) // 1000
    if interval == 0:
        interval = 10

    if len(threshold) < 10:
        interval = 1

    if pa:
        # point k adjustment
        for k in k_list:
            # find best F1 score with varying thresholds
            if len(scores) // interval < 1:
                ths = threshold
            else:
                ths = [threshold[interval * i] for i in range(len(threshold) // interval)]
            pa_f1_scores = [metrics.f1_score(targets, pak(scores, targets, th, k)) for th in ths]
            pa_f1_scores = np.asarray(pa_f1_scores)

            best_f1_pa_list.append(np.max(pa_f1_scores))
            pa_scores = pak(scores, targets, ths[np.argmax(pa_f1_scores)], k)
            best_threshold_pa_list.append(ths[np.argmax(pa_f1_scores)])
            best_precision_pa_list.append(metrics.precision_score(targets, pa_scores))
            best_recall_pa_list.append(metrics.recall_score(targets, pa_scores))

        f1_pa_k = 0
        best_f1_pa_list.append(np.max(f1_score))
        for i, best_f1_pa in enumerate(best_f1_pa_list):
            if i == 0 or i == len(best_f1_pa_list) - 1:
                f1_pa_k += best_f1_pa
            else:
                f1_pa_k += 2 * best_f1_pa
        f1_pa_k /= (2 * len(best_f1_pa_list) - 2)

    results = EvalResult(
        best_f1_wo_pa=np.max(f1_score),
        f1_pa_k=f1_pa_k,
        best_precision_wo_pa=precision[np.argmax(f1_score)],
        best_recall_wo_pa=recall[np.argmax(f1_score)],
        best_threshold_wo_pa=threshold[np.argmax(f1_score)],
        affiliation_f1=np.max(affiliation_f1),
        affiliation_precision=affiliation_precision[np.argmax(affiliation_f1)],
        affiliation_recall=affiliation_recall[np.argmax(affiliation_f1)],
        auc_roc=metrics.roc_auc_score(targets, scores),
        auc_prc=auc_prc,
        R_AUC_ROC=vus_metrics["R_AUC_ROC"],
        R_AUC_PR=vus_metrics["R_AUC_PR"],
        VUS_ROC=vus_metrics["VUS_ROC"],
        VUS_PR=vus_metrics["VUS_PR"],
        pa_rate=k_list,
        best_f1_with_pa=best_f1_pa_list,
        best_precision_with_pa=best_precision_pa_list,
        best_recall_with_pa=best_recall_pa_list,
        best_threshold_with_pa=best_threshold_pa_list
    )

    return results


if __name__ == "__main__":
    y_test = np.zeros(10000)
    y_test[10:20] = 1
    y_test[50:60] = 1
    anomaly_scores = np.random.randint(0, 10000, size=10000)
    anomaly_scores = anomaly_scores / 10000
    anomaly_scores[12:23] = 0.9
    anomaly_scores[40:70] = 0.98
    eval_result = evaluate(scores=anomaly_scores, targets=y_test)
    print("finished")
