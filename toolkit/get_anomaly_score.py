import numpy as np
from evaluation.evaluate import evaluate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
from scipy import signal
import pandas as pd
from toolkit.load_dataset import load_dataset
import os
import json
from dataclasses import dataclass
from toolkit.result_plot import recon_plot

constant_std = 1e-6


@dataclass
class AnomalyScoreOutput:
    train_score_all: np.ndarray  # sequence_length
    test_score_all: np.ndarray  # sequence_length
    train_channel_score: np.ndarray = None  # sequence_length, num_channels
    test_channel_score: np.ndarray = None  # sequence_length, num_channels


def moving_average(score_t, window=3):
    # return length = len(score_t) - window + 1
    ret = np.cumsum(score_t, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


class AnomalyScoreCalculator:
    def __init__(self, mode: str, if_normalize=True, long_window=1024, short_window=5, average_window=None):
        assert mode in ['error', 'dynamic', 'dynamic_kernel'], 'Invalid mode'
        self.mode = mode
        self.if_normalize = if_normalize
        self.long_window = long_window
        self.short_window = short_window
        self.average_window = average_window

    def calculate_anomaly_score(self,
                                raw_train_data: np.ndarray,
                                raw_test_data: np.ndarray,
                                recon_train_data: np.ndarray,
                                recon_test_data: np.ndarray):
        """
        @type recon_train_data: object
        """
        if self.mode == 'error':
            output = self.get_error_score(raw_train_data, raw_test_data,
                                          recon_train_data, recon_test_data)
            return output
        elif self.mode == 'dynamic':
            output = self.get_dynamic_scores(raw_train_data,
                                             raw_test_data,
                                             recon_train_data,
                                             recon_test_data,
                                             long_window=self.long_window,
                                             short_window=self.short_window)
        else:
            raise NotImplementedError('other dynamic kernel not implemented')

        return output

    def get_error_score(self,
                        raw_train_data: np.ndarray,
                        raw_test_data: np.ndarray,
                        recon_train_data: np.ndarray,
                        recon_test_data: np.ndarray):
        """
        @param recon_test_data: sequence_length, num_channels
        @param raw_test_data: sequence_length, num_channels
        @param raw_train_data: sequence_length, num_channels
        @param recon_train_data: sequence_length, num_channels
        @return: train_score_normalized, test_score_normalized
        """
        if len(recon_train_data.shape) == 1:
            recon_train_data = recon_train_data[:, np.newaxis]

        if len(recon_test_data.shape) == 1:
            recon_test_data = recon_test_data[:, np.newaxis]

        train_diff = np.abs(raw_train_data - recon_train_data)  # sequence_length, num_channels
        test_diff = np.abs(raw_test_data - recon_test_data)  # sequence_length, num_channels


        if self.if_normalize:
            mean_channel_diff_train = np.mean(train_diff, axis=0)
            train_channel_normalized = train_diff - mean_channel_diff_train
            test_channel_normalized = test_diff - mean_channel_diff_train
        else:
            train_channel_normalized = train_diff
            test_channel_normalized = test_diff

        train_score_normalized = np.sqrt(np.mean(train_channel_normalized ** 2, axis=1))
        test_score_normalized = np.sqrt(np.mean(test_channel_normalized ** 2, axis=1))

        if self.average_window is not None:
            weight = np.ones(self.average_window) / self.average_window
            train_score_normalized = np.convolve(train_score_normalized, weight, mode='same')
            test_score_normalized = np.convolve(test_score_normalized, weight, mode='same')

        return AnomalyScoreOutput(
            train_score_all=train_score_normalized,
            test_score_all=test_score_normalized,
            train_channel_score=train_channel_normalized,
            test_channel_score=test_channel_normalized
        )

    def get_dynamic_scores(self,
                           raw_train_data: np.ndarray,
                           raw_test_data: np.ndarray,
                           recon_train_data: np.ndarray,
                           recon_test_data: np.ndarray,
                           long_window=2000,
                           short_window=5):

        error_tc_train = np.abs(raw_train_data - recon_train_data)  # sequence_length, num_channels
        error_tc_test = np.abs(raw_test_data - recon_test_data)  # sequence_length, num_channels
        n_cols = error_tc_train.shape[1]

        score_tc_test_dyn = np.stack(
            [self._get_dynamic_score_t(error_tc_train[:, col],
                                       error_tc_test[:, col],
                                       long_window,
                                       short_window) for col in range(n_cols)], axis=-1
        )
        score_tc_train_dyn = np.stack(
            [self._get_dynamic_score_t(None,
                                       error_tc_train[:, col],
                                       long_window,
                                       short_window) for col in range(n_cols)], axis=-1
        )
        score_t_train_dyn = np.sum(score_tc_train_dyn, axis=1)
        score_t_test_dyn = np.sum(score_tc_test_dyn, axis=1)

        score_t_train_dyn = MinMaxScaler().fit_transform(score_t_train_dyn.reshape(-1, 1)).reshape(-1)
        score_t_test_dyn = MinMaxScaler().fit_transform(score_t_test_dyn.reshape(-1, 1)).reshape(-1)

        return AnomalyScoreOutput(
            train_score_all=score_t_train_dyn,
            test_score_all=score_t_test_dyn,
            train_channel_score=score_tc_train_dyn,
            test_channel_score=score_tc_test_dyn
        )

    def get_gaussian_kernel_scores(self, error_tc_train, error_tc_test, error_t_train, error_t_test, kernel_sigma=1,
                                   long_window=2000,
                                   short_window=3):
        # if error_tc is available, it will be used rather than error_t
        score_t_dyn, score_tc_dyn = self.get_dynamic_scores(error_tc_train=error_tc_train,
                                                            error_tc_test=error_tc_test,
                                                            error_t_train=error_t_train,
                                                            error_t_test=error_t_test,
                                                            long_window=long_window,
                                                            short_window=short_window)[:2]
        gaussian_kernel = signal.gaussian(kernel_sigma * 8, std=kernel_sigma)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        if score_tc_dyn is None:
            score_tc_dyn_gauss_conv = None
            score_t_dyn_gauss_conv = signal.convolve(score_t_dyn, gaussian_kernel, mode="same")
        else:
            n_cols = score_tc_dyn.shape[1]
            score_tc_dyn_gauss_conv = np.stack([signal.convolve(score_tc_dyn[:, col], gaussian_kernel, mode="same")
                                                for col in range(n_cols)], axis=-1)
            score_t_dyn_gauss_conv = np.sum(score_tc_dyn_gauss_conv, axis=1)

        return score_t_dyn_gauss_conv, score_tc_dyn_gauss_conv

    @staticmethod
    def _get_dynamic_score_t(error_t_train, error_t_test, long_window, short_window):
        # n_t: sequence length
        n_t = error_t_test.shape[0]

        # assuming that length of scores is always greater than short_window
        short_term_means = np.concatenate((error_t_test[:short_window - 1], moving_average(error_t_test, short_window)))
        if long_window >= n_t:
            long_win = n_t - 1
        else:
            long_win = long_window

        if error_t_train is None:
            init_score_t_test = np.zeros(long_win - 1)
            means_test_t = moving_average(error_t_test, long_win)
            stds_test_t = np.array(pd.Series(error_t_test).rolling(window=long_win).std().values)[long_win - 1:]
            stds_test_t[stds_test_t == 0] = constant_std
            distribution = norm(0, 1)
            score_t_test_dyn = -distribution.logsf((short_term_means[(long_win - 1):] - means_test_t) / stds_test_t)
            score_t_test_dyn = np.concatenate([init_score_t_test, score_t_test_dyn])
        else:
            if len(error_t_train) < long_win - 1:
                full_ts = np.concatenate([np.zeros(long_win - 1 - len(error_t_train)), error_t_train, error_t_test],
                                         axis=0)
            else:
                full_ts = np.concatenate([error_t_train[-long_win + 1:], error_t_test], axis=0)
            means_test_t = moving_average(full_ts, long_win)
            stds_test_t = np.array(pd.Series(full_ts).rolling(window=long_win).std().values)[long_win - 1:]
            stds_test_t[stds_test_t == 0] = constant_std
            distribution = norm(0, 1)
            score_t_test_dyn = -distribution.logsf((short_term_means - means_test_t) / stds_test_t)

        return score_t_test_dyn


if __name__ == "__main__":
    data_name = "ASD"
    train_data, test_data, labels = load_dataset("ASD", group=3)
    epoch = 90
    mode = "dynamic"  # "error" or "dynamic"
    load_dir = "output\PatchDetector\ASD_3\window_len_1024-d_model_64-patch_len_32-mode_common_channel-03-11-21-26"

    recon_train_data = np.load(os.path.join(load_dir, f"recon_train_{epoch}.npy"))
    recon_test_data = np.load(os.path.join(load_dir, f"recon_test_{epoch}.npy"))
    raw_score = np.load(os.path.join(load_dir, f"test_anomaly_score_{epoch}.npy"))

    train_data = train_data[:len(recon_train_data), :]
    test_data = test_data[:len(recon_test_data), :]
    labels = labels[:len(recon_test_data)]

    save_dir = f"anomalyScore_analysis_output/{data_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # calculate anomaly scores
    anomaly_score_calculator = AnomalyScoreCalculator(mode=mode, if_normalize=True, long_window=4000, short_window=3)
    anomaly_score_output = (anomaly_score_calculator.
                            calculate_anomaly_score(train_data, test_data, recon_train_data, recon_test_data))
    test_score_normalized = anomaly_score_output.test_score_all

    evaluate_result = evaluate(test_score_normalized, labels)
    evaluate_result_raw = evaluate(raw_score, labels)

    eval_result_save_path = os.path.join(save_dir, f"test_result_{mode}_{epoch}.json")
    with open(eval_result_save_path, "w") as file:
        json.dump(evaluate_result.__dict__, file, indent=4)

    eva_result_raw_save_path = os.path.join(save_dir, f"test_result_raw_{epoch}.json")
    with open(eva_result_raw_save_path, "w") as file:
        json.dump(evaluate_result_raw.__dict__, file, indent=4)

    # plot
    threshold = evaluate_result.best_threshold_wo_pa
    raw_threshold = evaluate_result_raw.best_threshold_wo_pa

    recon_plot(
        save_path=os.path.join(save_dir, f"recon_plot_{mode}_{epoch}.png"),
        gap=400,
        test_data=test_data,
        train_data=train_data,
        figure_length=20,
        figure_width=15,
        font_size=3,
        threshold=threshold,
        recon_train_data=recon_train_data,
        recon_test_data=recon_test_data,
        train_anomaly_score=anomaly_score_output.train_score_all,
        test_anomaly_score=anomaly_score_output.test_score_all,
        test_label=labels
    )

    recon_plot(
        save_path=os.path.join(save_dir, f"recon_plot_raw_{epoch}.png"),
        gap=400,
        test_data=test_data,
        train_data=train_data,
        figure_length=20,
        figure_width=15,
        font_size=3,
        threshold=raw_threshold,
        recon_train_data=recon_train_data,
        recon_test_data=recon_test_data,
        test_anomaly_score=raw_score,
        test_label=labels
    )
