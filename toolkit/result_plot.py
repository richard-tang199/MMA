import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import tqdm

matplotlib.use('agg')


def get_segments(label_data: np.ndarray) -> list:
    segments = []
    start = None
    for i in range(len(label_data)):
        if label_data[i] == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(label_data) - 1))
    return segments


def recon_plot(save_path: str,
               gap: float,
               test_data: np.ndarray,
               train_data: np.ndarray,
               threshold: float = None,
               figure_length: int = None,
               figure_width: int = None,
               font_size: int = None,
               recon_test_data: np.ndarray = None,
               recon_train_data: np.ndarray = None,
               test_anomaly_score: np.ndarray = None,
               train_anomaly_score: np.ndarray = None,
               test_label: np.ndarray = None,
               train_label: np.ndarray = None,
               plot_diff: bool = False):
    """
    @type threshold: object
    @param font_size:
    @param figure_wid
    @param figure_length:
    @param gap: axis gap
    @param save_path: save path/ png file name
    @param test_data: sequence_length, num_channels
    @param recon_test_data: sequence_length, num_channels
    @param train_data: sequence_length, num_channels
    @param recon_train_data: sequence_length, num_channels
    @param test_anomaly_score: sequence_length
    @param train_anomaly_score: sequence_length
    @param test_label: sequence_length
    @param train_label: sequence_length
    @return: None
    """

    sequence_length, dim_size = test_data.shape[0] + train_data.shape[0], train_data.shape[1]
    if figure_length is None or figure_width is None:
        if sequence_length >= 2000 and dim_size >= 8:
            figure_length, figure_width = sequence_length // 100, int(dim_size * 5)
        else:
            figure_length, figure_width = 20, 15

    if font_size is None:
        font_size = min(figure_length, figure_width) // dim_size

    # add a dimension for anomaly_score plotting
    n_dim = dim_size + 1

    ratio = train_data.shape[0] / test_data.shape[0]
    fig, axs = plt.subplots(
        nrows=n_dim, ncols=2,
        sharey=False, sharex=False,
        figsize=(figure_length, figure_width),
        tight_layout=True,
        gridspec_kw={'width_ratios': [ratio, 1]}
    )

    train_anomaly_segments = None
    test_anomaly_segments = None

    if train_label is not None:
        train_anomaly_segments = get_segments(train_label)

    if test_label is not None:
        test_anomaly_segments = get_segments(test_label)

    diff_train = []
    diff_test = []

    for dim in tqdm.trange(1, dim_size + 1):
        dim_train = train_data[:, dim - 1]
        dim_test = test_data[:, dim - 1]

        # plot train data
        axs[dim - 1][0].text(0.5, 0.8, f"train_data {dim}", fontsize=font_size)
        axs[dim - 1][0].plot(dim_train, label="raw")

        if recon_train_data is not None:
            axs[dim - 1][0].plot(recon_train_data[:, dim - 1], label="recon")
            diff_train_dim = abs(dim_train - recon_train_data[:, dim - 1]) * 0.8
            diff_train.append(diff_train_dim)
            if plot_diff:
                axs[dim - 1][0].plot(diff_train_dim, label="diff")

        # plot train label
        if train_anomaly_segments is not None:
            for seg in train_anomaly_segments:
                if seg[0] == seg[1]:
                    axs[dim - 1][0].axvline(x=seg[0], color='red', alpha=0.1)
                else:
                    axs[dim - 1][0].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

        axs[dim - 1][0].legend(fontsize=font_size)
        axs[dim - 1][0].xaxis.set_major_locator(MultipleLocator(gap))
        axs[dim - 1][0].set_xlim(0, train_data.shape[0])

        # plot test data
        axs[dim - 1][1].text(0.5, 0.8, f"test_data {dim}", fontsize=font_size)
        axs[dim - 1][1].plot(dim_test, label="raw")

        if recon_test_data is not None:
            axs[dim - 1][1].plot(recon_test_data[:, dim - 1], label="recon")
            diff_test_dim = abs(dim_test - recon_test_data[:, dim - 1]) * 0.8
            diff_test.append(diff_test_dim)
            if plot_diff:
                axs[dim - 1][1].plot(diff_test_dim, label="diff")

        if test_anomaly_segments is not None:
            for seg in test_anomaly_segments:
                if seg[0] == seg[1] or seg[1] - seg[0] == 1:
                    axs[dim - 1][1].axvline(x=seg[0], color='red', alpha=1, linewidth=2)
                else:
                    axs[dim - 1][1].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

        axs[dim - 1][1].legend(fontsize=font_size)
        axs[dim - 1][1].xaxis.set_major_locator(MultipleLocator(gap))
        axs[dim - 1][1].set_xlim(0, test_data.shape[0])

        y_max, y_min = (max(dim_train.max(), dim_test.max(), 1) + 0.1,
                        min(dim_train.min(), dim_test.min(), 0) - 0.2)

        axs[dim - 1][0].set_ylim(y_min, y_max)
        axs[dim - 1][1].set_ylim(y_min, y_max)

    # plot train anomaly score
    if train_anomaly_score is None and recon_train_data is not None:
        train_anomaly_score = np.array(diff_train).mean(0)
        # train_anomaly_score = (train_anomaly_score - train_anomaly_score.min()) / (
        #         train_anomaly_score.max() - train_anomaly_score.min())

    if train_anomaly_score is not None:
        axs[-1][0].plot(train_anomaly_score)

    if train_anomaly_segments is not None:
        for seg in train_anomaly_segments:
            if seg[0] == seg[1]:
                axs[-1][0].axvline(x=seg[0], color='red', alpha=1, linewidth=2)
            else:
                axs[-1][0].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

    axs[-1][0].xaxis.set_major_locator(MultipleLocator(gap))
    axs[-1][0].set_xlim(0, train_data.shape[0])

    # plot test anomaly score
    if test_anomaly_score is None and recon_test_data is not None:
        test_anomaly_score = np.array(diff_test).mean(0)
        # test_anomaly_score = (test_anomaly_score - test_anomaly_score.min()) / (
        #         test_anomaly_score.max() - test_anomaly_score.min())

    if test_anomaly_score is not None:
        axs[-1][1].plot(test_anomaly_score)

    if test_anomaly_segments is not None:
        for seg in test_anomaly_segments:
            if seg[0] == seg[1] or seg[1] - seg[0] == 1:
                axs[-1][1].axvline(x=seg[0], color='red', alpha=0.1)
            else:
                axs[-1][1].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

    if threshold is not None:
        axs[-1][1].axhline(y=threshold, color='red', alpha=0.5)

    axs[-1][1].xaxis.set_major_locator(MultipleLocator(gap))
    axs[-1][1].set_xlim(0, test_data.shape[0])

    plt.savefig(save_path, format="png", dpi=100)
    plt.close()


def score_plot(save_path: str,
               gap: float,
               test_data: np.ndarray,
               train_data: np.ndarray,
               threshold: float = None,
               figure_length: int = None,
               figure_width: int = None,
               font_size: int = None,
               recon_test_data: np.ndarray = None,
               recon_train_data: np.ndarray = None,
               test_anomaly_score: np.ndarray = None,
               train_anomaly_score: np.ndarray = None,
               test_label: np.ndarray = None,
               train_label: np.ndarray = None,
               plot_diff: bool = False):
    """
    @type threshold: object
    @param font_size:
    @param figure_width:
    @param figure_length:
    @param gap: axis gap
    @param save_path: save path/ png file name
    @param test_data: sequence_length, num_channels
    @param recon_test_data: sequence_length, num_channels
    @param train_data: sequence_length, num_channels
    @param recon_train_data: sequence_length, num_channels
    @param test_anomaly_score: sequence_length, num_channels
    @param train_anomaly_score: sequence_length, num_channels
    @param test_label: sequence_length
    @param train_label: sequence_length
    @return: None
    """

    sequence_length, dim_size = test_data.shape[0] + train_data.shape[0], train_data.shape[1]
    if figure_length is None or figure_width is None:
        if sequence_length >= 2000 and dim_size >= 8:
            figure_length, figure_width = sequence_length // 100, int(dim_size * 5)
        else:
            figure_length, figure_width = 20, 15

    if font_size is None:
        font_size = min(figure_length, figure_width) // dim_size

    # add a dimension for anomaly_score plotting
    n_dim = dim_size + 1

    ratio = train_data.shape[0] / test_data.shape[0]
    fig, axs = plt.subplots(
        nrows=n_dim, ncols=2,
        sharey=False, sharex=False,
        figsize=(figure_length, figure_width),
        tight_layout=True,
        gridspec_kw={'width_ratios': [ratio, 1]}
    )

    train_anomaly_segments = None
    test_anomaly_segments = None

    if train_label is not None:
        train_anomaly_segments = get_segments(train_label)

    if test_label is not None:
        test_anomaly_segments = get_segments(test_label)

    diff_train = []
    diff_test = []

    for dim in tqdm.trange(1, dim_size + 1):
        dim_train = train_data[:, dim - 1]
        dim_test = test_data[:, dim - 1]
        dim_train_anomaly_score = train_anomaly_score[:, dim - 1]
        dim_test_anomaly_score = test_anomaly_score[:, dim - 1]

        # plot train data
        axs[dim - 1][0].text(0.5, 0.8, f"train_data {dim}", fontsize=font_size)
        axs[dim - 1][0].plot(dim_train, label="raw")
        axs[dim - 1][0].plot(dim_train_anomaly_score, label="anomaly_score", color='black', linestyle='--')

        if recon_train_data is not None:
            axs[dim - 1][0].plot(recon_train_data[:, dim - 1], label="recon")
            diff_train_dim = abs(dim_train - recon_train_data[:, dim - 1]) * 0.8 - 0.2
            diff_train.append(diff_train_dim)
            if plot_diff:
                axs[dim - 1][0].plot(diff_train_dim, label="diff")

        # plot train label
        if train_anomaly_segments is not None:
            for seg in train_anomaly_segments:
                if seg[0] == seg[1]:
                    axs[dim - 1][0].axvline(x=seg[0], color='red', alpha=0.1)
                else:
                    axs[dim - 1][0].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

        axs[dim - 1][0].legend(fontsize=font_size)
        axs[dim - 1][0].xaxis.set_major_locator(MultipleLocator(gap))
        axs[dim - 1][0].set_xlim(0, train_data.shape[0])

        # plot test data
        axs[dim - 1][1].text(0.5, 0.8, f"test_data {dim}", fontsize=font_size)
        axs[dim - 1][1].plot(dim_test, label="raw")
        axs[dim - 1][1].plot(dim_test_anomaly_score, label="anomaly", color='black', linestyle='--')

        if recon_test_data is not None:
            axs[dim - 1][1].plot(recon_test_data[:, dim - 1], label="recon")
            diff_test_dim = abs(dim_test - recon_test_data[:, dim - 1]) * 0.8 - 0.2
            diff_test.append(diff_test_dim)
            if plot_diff:
                axs[dim - 1][1].plot(diff_test_dim, label="diff")

        if test_anomaly_segments is not None:
            for seg in test_anomaly_segments:
                if seg[0] == seg[1]:
                    axs[dim - 1][1].axvline(x=seg[0], color='red', alpha=0.1)
                else:
                    axs[dim - 1][1].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

        axs[dim - 1][1].legend(fontsize=font_size)
        axs[dim - 1][1].xaxis.set_major_locator(MultipleLocator(gap))
        axs[dim - 1][1].set_xlim(0, test_data.shape[0])

        y_max, y_min = (max(dim_train.max(), dim_test.max(), 1) + 0.1,
                        min(dim_train.min(), dim_test.min(), 0) - 0.2)

        axs[dim - 1][0].set_ylim(y_min, y_max)
        axs[dim - 1][1].set_ylim(y_min, y_max)

    # plot train anomaly score
    if train_anomaly_score is None and recon_train_data is not None:
        train_anomaly_score = np.array(diff_train).mean(0)
        # train_anomaly_score = (train_anomaly_score - train_anomaly_score.min()) / (
        #         train_anomaly_score.max() - train_anomaly_score.min())

    if train_anomaly_score is not None:
        axs[-1][0].plot(train_anomaly_score.mean(-1))

    if train_anomaly_segments is not None:
        for seg in train_anomaly_segments:
            if seg[0] == seg[1]:
                axs[-1][0].axvline(x=seg[0], color='red', alpha=0.1)
            else:
                axs[-1][0].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

    axs[-1][0].xaxis.set_major_locator(MultipleLocator(gap))
    axs[-1][0].set_xlim(0, train_data.shape[0])

    # plot test anomaly score
    if test_anomaly_score is None and recon_test_data is not None:
        test_anomaly_score = np.array(diff_test).mean(0)
        # test_anomaly_score = (test_anomaly_score - test_anomaly_score.min()) / (
        #         test_anomaly_score.max() - test_anomaly_score.min())

    if test_anomaly_score is not None:
        axs[-1][1].plot(test_anomaly_score.mean(-1))

    if test_anomaly_segments is not None:
        for seg in test_anomaly_segments:
            if seg[0] == seg[1]:
                axs[-1][1].axvline(x=seg[0], color='red', alpha=0.1)
            else:
                axs[-1][1].axvspan(seg[0], seg[1], facecolor='red', alpha=0.3)

    if threshold is not None:
        axs[-1][1].axhline(y=threshold, color='red', alpha=0.5)

    axs[-1][1].xaxis.set_major_locator(MultipleLocator(gap))
    axs[-1][1].set_xlim(0, test_data.shape[0])

    plt.savefig(save_path, format="png", dpi=100)
    plt.close()


if __name__ == "__main__":
    segments = get_segments(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]))
    print(segments)
    # raw_train_data, raw_test_data, labels = load_dataset(data_name="UCR", group="235")
    # recon_plot(save_path="sample.png", train_data=raw_train_data, test_data=raw_test_data, test_label=labels, gap=400)
    # print(segments)
    # print("finished")
