import math
import os.path
import sys
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from toolkit.result_plot import recon_plot
import argparse
from toolkit.load_dataset import load_dataset, load_pollute_dataset
from toolkit.get_anomaly_score import AnomalyScoreCalculator
from evaluation.evaluate import evaluate, EfficiencyResult
import json
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.models.feature import Window
from TSB_UAD.models.sand import SAND
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="Series2Graph")
parser.add_argument('--group', type=str, default="1", help='group number')
parser.add_argument("--learning_rate", type=float, default=2e-3, help="learning rate")
parser.add_argument('--data_name', type=str, default='UCR', help='dataset name')
parser.add_argument("--window_length", type=int, default=100, help="window length")
parser.add_argument('--num_epochs', type=int, default=30, help="number of epochs")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--eval_gap", type=int, default=10, help="evaluation gap")
parser.add_argument('--figure_length', type=int, default=60, help="number of workers for dataloader")
parser.add_argument('--figure_width', type=int, default=20, help="number of workers for dataloader")
parser.add_argument('--anomaly_mode', type=str, default="error", help="anomaly mode")  # "error" or "dynamic"
parser.add_argument('--mode', type=str, default="normal", help="normal or robust verification")
parser.add_argument("--anomaly_ratio", type=float, default=0)

if __name__ == "__main__":
    sys.path.append("other_models")
    args = parser.parse_args()
    print(f"\n model name: {args.model_name}, data name: {args.data_name}_{args.group}")
    now = datetime.now().strftime("%m-%d-%H-%M")
    model_name = args.model_name
    data_name = args.data_name
    args.smoother_window_size = None
    group = args.group
    args.figure_length, args.figure_width = 160, 20
    args.group = args.group.zfill(3)

    raw_train_data, raw_test_data, raw_test_labels = load_dataset(data_name, args.group)
    raw_train_data = raw_train_data.squeeze()
    raw_test_data = raw_test_data.squeeze()
    args.window_length = find_length(raw_train_data)

    if args.mode != "normal":
        raw_train_data, raw_test_data, raw_train_labels, raw_test_labels = load_pollute_dataset(
            data_name=args.data_name,
            group=args.group,
            mode=args.mode,
            ratio=args.anomaly_ratio
        )
    else:
        args.anomaly_ratio = 0
        raw_train_labels = None

    output_dir = (f"output/{model_name}/{data_name}/"
                  f"{args.mode}_{args.anomaly_ratio}/{data_name}_{group}/window_length_{args.window_length}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "SAND":
        model = SAND(pattern_length=args.window_length, subsequence_length=4 * args.window_length)
        data = np.concatenate((raw_train_data, raw_test_data), axis=0).astype(np.float64)

        init_length = len(raw_train_data)

        start = time.time()
        model.fit(X=data,
                  online=False,
                  alpha=0.5,
                  init_length=init_length,
                  overlaping_rate=int(1.5 * args.window_length),
                  )
        duration = time.time() - start

        score = model.decision_scores_
        test_anomaly_score = score[-len(raw_test_data):]
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(test_anomaly_score.reshape(-1, 1)).ravel()
        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        recon_train = None
        recon_test = None
        flops = 0
        params = 0
        train_data = raw_train_data
        test_data = raw_test_data
        threshold = test_result.best_threshold_wo_pa
        train_anomaly_score = None
    elif model_name == "Series2Graph":
        from TSB_UAD.models.series2graph import Series2Graph
        model = Series2Graph(pattern_length=args.window_length)
        data = np.concatenate((raw_train_data, raw_test_data), axis=0).astype(np.float64)

        start = time.time()
        model.fit(data)
        query_length = 2 * args.window_length
        model.score(query_length=query_length, dataset=data)
        duration = time.time() - start
        score = model.decision_scores_
        score = np.array([score[0]] * math.ceil(query_length // 2) + list(score) + [score[-1]] * (query_length // 2))
        test_anomaly_score = score[-len(raw_test_data):]
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(test_anomaly_score.reshape(-1, 1)).ravel()
        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        recon_train = None
        recon_test = None
        flops = 0
        params = 0
        train_data = raw_train_data
        test_data = raw_test_data
        threshold = test_result.best_threshold_wo_pa
        train_anomaly_score = None

    train_data = train_data[:, np.newaxis]
    test_data = test_data[:, np.newaxis]

    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        json.dump(test_result.__dict__, f, indent=4)

    efficiency_result = EfficiencyResult(test_time=duration, flops=flops, params=params)
    efficiency_result = efficiency_result.__dict__
    with open(os.path.join(output_dir, f"efficiency_result.json"), "w") as f:
        json.dump(efficiency_result, f, indent=4)

    figure_save_path = os.path.join(output_dir, f"result.png")
    if recon_train is not None and recon_test is not None:
        gap = (recon_train.shape[0] + recon_test.shape[0]) // 80
    else:
        gap = 400

    recon_plot(
        save_path=figure_save_path,
        gap=gap,
        figure_length=args.figure_length,
        figure_width=args.figure_width,
        font_size=4,
        test_data=test_data,
        test_label=raw_test_labels,
        recon_test_data=recon_test,
        test_anomaly_score=test_anomaly_score,
        train_data=train_data,
        train_label=raw_train_labels,
        recon_train_data=recon_train,
        train_anomaly_score=train_anomaly_score,
        threshold=threshold,
        plot_diff=True
    )

    np.save(os.path.join(output_dir, f"raw_test_data.npy"), test_data)
    np.save(os.path.join(output_dir, f"raw_test_labels.npy"), raw_test_labels)
    np.save(os.path.join(output_dir, f"test_anomaly_score.npy"), test_anomaly_score)
