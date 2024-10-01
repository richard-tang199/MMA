import numpy as np
import pandas as pd
import os
import sys
import pickle
import tqdm

sys.path.append("../")


def load_dataset(data_name: str, group=None):
    # TODO: GIVE interval to use: few shot, few sample, all data
    """
    @type data_name: str
    @param group: asd group
    @return: train_data: (sequence_length, num_channels),
    test_data: (sequence_length, num_channels),
    labels: sequence_length
    """
    if data_name == "sate":
        # train_data: (sequence_length, num_channels), test_data: (sequence_length, num_channels),
        # labels: sequence_length

        data_dir = os.path.join("ano_dataset", data_name, group, "processed")
        train_data = np.load(os.path.join(data_dir, f"{group}_train.npy"))
        test_data = np.load(os.path.join(data_dir, f"{group}_test.npy"))
        labels = np.load(os.path.join(data_dir, f"{group}_labels.npy"))[:, 0]

    elif data_name == "ASD":
        data_dir = os.path.join("ano_dataset", "ASD", "processed")
        with open(os.path.join(data_dir, f"omi-{group}_train.pkl"), "rb") as file:
            train_data = pickle.load(file)

        with open(os.path.join(data_dir, f"omi-{group}_test.pkl"), "rb") as file:
            test_data = pickle.load(file)

        with open(os.path.join(data_dir, f"omi-{group}_test_label.pkl"), "rb") as file:
            labels = pickle.load(file)

    elif data_name == "SMD":
        data_dir = os.path.join("ano_dataset", "SMD", "downsampled")

        assert group in ['1-1', '1-6', '1-7', '2-1', '2-2', '2-7', '2-8', '3-3', '3-4', '3-6', '3-8', '3-11'], (
            "group not found")

        with open(os.path.join(data_dir, f"machine-{group}_train.pkl"), "rb") as file:
            train_data = pickle.load(file)

        with open(os.path.join(data_dir, f"machine-{group}_test.pkl"), "rb") as file:
            test_data = pickle.load(file)

        with open(os.path.join(data_dir, f"machine-{group}_test_label.pkl"), "rb") as file:
            labels = pickle.load(file)

    elif data_name == "synthetic":
        data_dir = os.path.join("ano_dataset", "synthetic", "processed")
        train_data = np.loadtxt(os.path.join(data_dir, "synthetic_train.csv"), delimiter=",")
        test_data = np.loadtxt(os.path.join(data_dir, "synthetic_test.csv"), delimiter=",")
        labels = np.loadtxt(os.path.join(data_dir, "synthetic_label.csv"), delimiter=",", dtype=int)

    elif data_name == "GECCO":
        data_dir = os.path.join("ano_dataset", "GECCO", "processed")
        train_data = np.load(os.path.join(data_dir, "NIPS_TS_Water_train.npy"))
        test_data = np.load(os.path.join(data_dir, "NIPS_TS_Water_test.npy"))
        labels = np.load(os.path.join(data_dir, "NIPS_TS_Water_test_label.npy"))

    elif data_name == "PSM":
        data_dir = os.path.join("ano_dataset", "PSM", "processed")
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=0, header=0).values
        test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=0, header=0).values
        labels = pd.read_csv(os.path.join(data_dir, "test_label.csv"), index_col=0, header=0).values
        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)
        labels = np.nan_to_num(labels)

    elif data_name == "Swan":
        data_dir = os.path.join("ano_dataset", "Swan", "processed")
        train_data = np.load(os.path.join(data_dir, "NIPS_TS_Swan_train.npy"))
        test_data = np.load(os.path.join(data_dir, "NIPS_TS_Swan_test.npy"))
        labels = np.load(os.path.join(data_dir, "NIPS_TS_Swan_test_label.npy"))

    elif data_name == "TELCO":
        data_dir = os.path.join("ano_dataset", "TELCO", "processed")
        train_data = np.load(os.path.join(data_dir, "train_data.npy"))[:8000, :]
        test_data = np.load(os.path.join(data_dir, "test_data.npy"))
        labels = np.load(os.path.join(data_dir, "test_label.npy"))

    elif data_name == "UCR":
        data_dir = os.path.join("ano_dataset", "UCR", "processed")
        train_data = np.load(os.path.join(data_dir, f"{group}_train.npy"))
        test_data = np.load(os.path.join(data_dir, f"{group}_test.npy"))
        train_data = train_data[:, np.newaxis]
        test_data = test_data[:, np.newaxis]
        labels = np.load(os.path.join(data_dir, f"{group}_test_label.npy"))
    else:
        print(f"******* {data_name} *******")
        raise NotImplementedError("Not implement dataset")

    return np.array(train_data, dtype=np.float32), np.array(test_data, dtype=np.float32), np.array(labels, dtype=int)


def load_pollute_dataset(data_name: str, group=None, mode="realistic", ratio: float = None):
    # todo add train label
    if data_name == "synthetic":
        data_dir = os.path.join("ano_dataset", data_name, "processed")
        test_data = np.loadtxt(os.path.join(data_dir, "synthetic_test.csv"), delimiter=",")
        labels = np.loadtxt(os.path.join(data_dir, "synthetic_label.csv"), delimiter=",", dtype=int)

        if mode == "realistic":
            ratio = int(ratio)
            data_dir = os.path.join("ano_dataset", "synthetic", "new")
            train_data = np.loadtxt(os.path.join(data_dir, f"synthetic_{ratio}_test.csv"), delimiter=",")
            train_label = np.loadtxt(os.path.join(data_dir, f"synthetic_{ratio}_label.csv"), delimiter=",", dtype=int)
        elif mode == "simulated":
            data_dir = os.path.join("ano_dataset", "synthetic", "pollute")
            train_data = np.loadtxt(os.path.join(data_dir, f"synthetic_train_{mode}_{ratio}.csv"), delimiter=",")
            train_label = np.loadtxt(os.path.join(data_dir, f"synthetic_train_label_{mode}_{ratio}.csv"), delimiter=",", dtype=int)

    elif data_name == "sate":
        data_dir = os.path.join("ano_dataset", data_name, group, "processed")
        test_data = np.load(os.path.join(data_dir, f"{group}_test.npy"))
        labels = np.load(os.path.join(data_dir, f"{group}_labels.npy"))[:, 0]

        if mode == "realistic":
            ratio = int(ratio)
            data_dir = os.path.join("ano_dataset", data_name, "new")
            train_data = np.load(os.path.join(data_dir, f"{group}_train_{mode}_{ratio}.npy"))
            train_label = np.load(os.path.join(data_dir, f"{group}_train_label_{mode}_{ratio}.npy"))

        elif mode == "simulated":
            data_dir = os.path.join("ano_dataset", data_name, group, "pollute")
            train_data = np.load(os.path.join(data_dir, f"{group}_train_{mode}_{ratio}.npy"))
            train_label = np.load(os.path.join(data_dir, f"{group}_train_label_{mode}_{ratio}.npy"))

    elif data_name == "ASD":
        assert group in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
        data_dir = os.path.join("ano_dataset", data_name, "processed")
        with open(os.path.join(data_dir, f"omi-{group}_test.pkl"), "rb") as file:
            test_data = pickle.load(file)

        with open(os.path.join(data_dir, f"omi-{group}_test_label.pkl"), "rb") as file:
            labels = pickle.load(file)

        if mode == "realistic":
            ratio = int(ratio)
            data_dir = os.path.join("ano_dataset", data_name, "new")
            train_data = np.load(os.path.join(data_dir, f"omi-{group}_train_{mode}_{ratio}.npy"))
            train_label = np.load(os.path.join(data_dir, f"omi-{group}_train_label_{mode}_{ratio}.npy"))
        elif mode == "simulated":
            data_dir = os.path.join("ano_dataset", data_name, "pollute")
            train_data = np.load(os.path.join(data_dir, f"omi-{group}_train_{mode}_{ratio}.npy"))
            train_label = np.load(os.path.join(data_dir, f"omi-{group}_train_label_{mode}_{ratio}.npy"))

    elif data_name == "UCR":
        assert group in ["006", "025", "048", "141", "145", "160", "173"]
        data_dir = os.path.join("ano_dataset", data_name, "processed")
        test_data = np.load(os.path.join(data_dir, f"{group}_test.npy"))
        labels = np.load(os.path.join(data_dir, f"{group}_test_label.npy"))

        if mode == "realistic":
            ratio = int(ratio)
            data_dir = os.path.join("ano_dataset", data_name, "new")
            train_data = np.load(os.path.join(data_dir, f"{data_name}-{group}_train_{mode}_{ratio}.npy"))
            train_label = np.load(os.path.join(data_dir, f"{data_name}-{group}_train_label_{mode}_{ratio}.npy"))

        elif mode == "simulated":
            data_dir = os.path.join("ano_dataset", data_name, "pollute")
            train_data = np.load(os.path.join(data_dir, f"{data_name}-{group}_train_{mode}_{ratio}.npy"))
            train_label = np.load(os.path.join(data_dir, f"{data_name}-{group}_train_label_{mode}_{ratio}.npy"))

    return (np.array(train_data, dtype=np.float32), np.array(test_data, dtype=np.float32),
            np.array(train_label, dtype=int), np.array(labels, dtype=int))

def load_explain_dataset(data_name: str, group=None):
    if data_name == "synthetic":
        data_dir = os.path.join("ano_dataset", data_name, "processed")
        train_data = np.loadtxt(os.path.join(data_dir, "synthetic_train.csv"), delimiter=",")
        test_data = np.loadtxt(os.path.join(data_dir, "synthetic_test.csv"), delimiter=",")
        raw_test_data = np.loadtxt(os.path.join(data_dir, "synthetic_raw_test.csv"), delimiter=",")
        explain_labels = np.loadtxt(os.path.join(data_dir, "synthetic_label.csv"), delimiter=",", dtype=int)

    elif data_name == "sate":
        data_dir = os.path.join("ano_dataset", data_name, group, "processed")
        train_data = np.load(os.path.join(data_dir, f"{group}_train.npy"))
        data_dir = os.path.join("ano_dataset", data_name, group, "explain")
        raw_test_data = np.load(os.path.join(data_dir, f"raw_test_data.npy"))
        test_data = np.load(os.path.join(data_dir, f"new_test_data.npy"))
        explain_labels = np.load(os.path.join(data_dir, f"explain_labels.npy"))

    elif data_name == "ASD":
        assert group in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
        data_dir = os.path.join("ano_dataset", data_name, "processed")
        with open(os.path.join(data_dir, f"omi-{group}_train.pkl"), "rb") as file:
            train_data = pickle.load(file)
        data_dir = os.path.join("ano_dataset", data_name, "pollute")
        test_data = np.load(os.path.join(data_dir, f"omi-{group}_train_realistic.npy"))
        explain_labels = np.load(os.path.join(data_dir, f"omi-{group}_train_label_realistic.npy"))
        raw_test_data = train_data

    elif data_name == "UCR":
        assert group in ["006", "025", "048", "141", "145", "160", "173"]
        data_dir = os.path.join("ano_dataset", "UCR", "processed")
        train_data = np.load(os.path.join(data_dir, f"{group}_train.npy"))
        data_dir = os.path.join("ano_dataset", "UCR", "explain", group )
        raw_test_data = np.load(os.path.join(data_dir, f"raw_test_data.npy"))
        test_data = np.load(os.path.join(data_dir, f"new_test_data.npy"))
        explain_labels = np.load(os.path.join(data_dir, f"explain_labels.npy"))

    else:
        raise NotImplementedError("Not implement dataset")

    return (np.array(train_data, dtype=np.float32),
            np.array(test_data, dtype=np.float32),
            np.array(raw_test_data, dtype=np.float32),
            np.array(explain_labels, dtype=int))


class SequenceWindowConversion:
    def __init__(self, window_size: int, stride_size: int = 1, mode="train"):
        """
        @param window_size: window size
        @param stride_size: moving size
        """

        self.windows = None
        self.pad_sequence_data = None
        self.raw_sequence_data = None
        self.pad_length = None
        self.is_converted = False
        self.window_size = window_size
        self.stride_size = stride_size
        self.mode = mode
        assert stride_size <= window_size, "window size must be larger than stride size"

    def sequence_to_windows(self, sequence_data) -> np.ndarray:
        """
        @param sequence_data: (length, channels)
        @return: windows: (num_window, window_size, channels)
        """
        self.is_converted = True
        self.raw_sequence_data = sequence_data
        raw_data_length, num_channels = self.raw_sequence_data.shape
        # pad the first patch
        pad_length: int = self.stride_size - (raw_data_length - self.window_size) % self.stride_size
        if self.stride_size == 1 or (raw_data_length - self.window_size) % self.stride_size == 0:
            pad_length = 0
        self.pad_length = pad_length
        # TODO: check the pad length
        # if self.mode == "test":
        #     assert self.pad_length <= 500, "stride size is not proper"
        self.pad_sequence_data = np.concatenate([np.zeros([pad_length, num_channels]), sequence_data], axis=0)
        data_length, num_channels = self.pad_sequence_data.shape

        start_idx_list = np.arange(0, data_length - self.window_size + 1, self.stride_size)
        end_idx_list = np.arange(self.window_size, data_length + 1, self.stride_size)
        windows = []

        for start_id, end_id in zip(start_idx_list, end_idx_list):
            windows.append(self.pad_sequence_data[start_id:end_id])

        self.windows = np.array(windows, dtype=np.float32)

        return self.windows

    def windows_to_sequence(self, windows: np.ndarray) -> np.ndarray:
        """
        convert the windows back to same length sequence, where the overlapping parts take the mean value
        @param windows: (num_window, window_size, channels)
        @return: sequence_data: (length, channels)
        """
        assert self.is_converted, "please first convert to windows"
        # initialize an empty array to store the sequence data
        sequence_data = np.zeros_like(self.pad_sequence_data)
        # get the number of windows, the window size, and the number of channels
        num_window, window_size, num_channels = windows.shape
        # get the length of the original sequence data
        length = sequence_data.shape[0]
        # loop through each window
        for i in range(num_window):
            # get the start and end index of the window in the sequence data
            start = i * self.stride_size
            end = start + window_size
            # if the end index exceeds the length, truncate the window
            if end > length:
                end = length
                window = windows[i, :end - start, :]
            else:
                window = windows[i]
            # add the window to the corresponding part of the sequence data
            sequence_data[start:end, :] += window
        # get the number of times each element in the sequence data is added
        counts = np.zeros_like(sequence_data)
        # loop through each window again
        for i in range(num_window):
            # get the start and end index of the window in the sequence data
            start = i * self.stride_size
            end = start + window_size
            # if the end index exceeds the length, truncate the window
            if end > length:
                end = length
            # increment the counts by one for each element in the window
            counts[start:end, :] += 1
        # divide the sequence data by the counts to get the mean value
        sequence_data /= counts
        # return the sequence data
        return sequence_data[self.pad_length:, :]


if __name__ == "__main__":
    # data_name = "Swan"
    # group = "1-1"
    # train_data, test_data, labels = load_dataset(data_name, group=group)
    # save_path = os.path.join("ano_dataset", data_name, "plot", f"{data_name}.png")
    # recon_plot(train_data=train_data, test_data=test_data, gap=600, figure_length=160, figure_width=80, font_size=10,
    #            test_label=labels, save_path=save_path)
    # print(train_data.shape, test_data.shape, labels.shape)
    import random

    window_size = 1024
    num_channels = 5
    for i in tqdm.trange(100):
        multiple = random.uniform(1.1, 100.0)
        stride_size = random.randint(1, window_size)
        size = int(window_size * multiple)
        train_data = np.random.randint(0, 10, size=(size, num_channels))
        window_converter = SequenceWindowConversion(window_size=window_size, stride_size=stride_size)
        data_windows = window_converter.sequence_to_windows(train_data)
        recon_data = window_converter.windows_to_sequence(data_windows)
        assert (recon_data == train_data).all(), "value differ"
        print('\nfinished test')
