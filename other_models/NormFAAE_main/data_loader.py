from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np


# Obtain time window sample
class SegDataLoader(Dataset):
    def __init__(self, data, win_size, step, label=None):
        self.data = data
        self.win_size = win_size
        self.step = step
        self.label = label

    def __len__(self):
        return max((self.data.shape[0] - self.win_size) // self.step + 1, 0)

    def __getitem__(self, index):
        index *= self.step
        if self.label is None:
            return np.float32(self.data[index:index + self.win_size])
        else:
            return np.float32(self.data[index:index + self.win_size]), \
                   np.float32(self.label[index:index + self.win_size])


# Delete features with unique value
def delete_unique(train, test):
    n_features = train.shape[1]
    for i in range(n_features - 1, -1, -1):
        feature_train = train[:, i]
        feature_test = test[:, i]
        c1 = np.unique(feature_train).shape[0]
        c2 = np.unique(feature_test).shape[0]
        if c1 == 1 and c2 == 1:
            train = np.delete(train, i, axis=-1)
            test = np.delete(test, i, axis=-1)

    n_features = train.shape[1]
    return train, test, n_features


# Obtain the mean, standard deviation, minimum, range, and variable type of the original dataset
def get_statistics(train):
    n_features = train.shape[1]
    min_ = np.zeros([n_features])
    mea_ = np.zeros([n_features])
    dis_ = np.zeros([n_features])
    std_ = np.zeros([n_features])
    con_ = np.zeros([n_features])
    for i in range(n_features):
        feature_i = train[:, i]
        min_[i] = feature_i.min()
        mea_[i] = feature_i.mean()
        if (feature_i.max() - feature_i.min()) != 0:
            dis_[i] = feature_i.max() - feature_i.min()
        else:
            dis_[i] = 1
        if feature_i.std() != 0:
            std_[i] = feature_i.std()
        else:
            std_[i] = 1
        if np.unique(feature_i).shape[0] > 3:
            con_[i] = 1
        else:
            con_[i] = 0
    return dis_, min_, mea_, std_, con_


# Get dataloader dataset
def get_dataset(data_name, data_path, batch_size, win_size, step):
    train = np.load(data_path + data_name + "_train.npy", allow_pickle=False)
    test = np.load(data_path + data_name + "_test.npy", allow_pickle=False)
    test_label = np.load(data_path + data_name + "_test_label.npy", allow_pickle=False)

    train, test, n_features = delete_unique(train, test)
    dis, min, mea, std, con = get_statistics(train)
    train_set = SegDataLoader(train, win_size, step)
    test_data = SegDataLoader(test, win_size, win_size, label=test_label)

    train_size = int(len(train_set) * 0.8)
    valid_size = len(train_set) - train_size
    train_data, valid_data = random_split(train_set, [train_size, valid_size])

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, valid_loader, test_loader, n_features, mea, std, dis, min, con
