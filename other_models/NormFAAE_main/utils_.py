import os
import csv
import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score



def try_gpu(i=0):
    try:
        return torch.device(f'cuda:{i}')
    except:
        return torch.device('cpu')


def creat_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def setup_seed(seed=0):
    torch.manual_seed(seed)  # Torch random module.
    np.random.seed(seed)  # Numpy random module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # Random seed of current GPU.
        torch.cuda.manual_seed_all(seed)  # Random seed of all GPU.


def grad_clipping(net, theta):
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EarlyStopping:
    def __init__(self, patience, data_name, model_path):
        self.patience = patience
        self.counter = 0
        self.val_loss_min = np.Inf
        self.best_score = None
        self.early_stop = False
        self.data_name = data_name
        self.model_path = model_path

    def __call__(self, val_loss, encoder, decoder, stander):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(encoder, decoder, stander)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(encoder, decoder, stander)
            self.counter = 0

    def save_checkpoint(self, encoder, decoder, stander):
        torch.save(encoder.state_dict(), self.model_path + self.data_name + '_encoder.pth')
        torch.save(decoder.state_dict(), self.model_path + self.data_name + '_decoder.pth')
        torch.save(stander.state_dict(), self.model_path + self.data_name + '_stander.pth')


def get_metrics(label, score, path, data_name):
    auc = roc_auc_score(label, score)

    events, i = [], 0
    while i < label.shape[0]:
        if label[i] == 1:
            start = i
            while i < label.shape[0] and label[i] == 1:
                i += 1
            end = i
            events.append((start, end))
        else:
            i += 1

    Fc1, F1_K = eval_result(label, score, events)
    with open(path + f"{data_name}.csv", 'a', newline='') as f:
        csv_file = csv.writer(f)
        csv_file.writerow([str(auc), str(Fc1), str(F1_K)])
    f.close()


def eval_result(label, test_scores, events):
    max_Fc1 = 0.0
    max_F1_K = 0.0
    for ratio in np.arange(0, 50.1, 0.1):
        threshold = np.percentile(test_scores, 100 - ratio)
        pred = (test_scores > threshold).astype(int)
        Fc1 = cal_Fc1(pred, events, label)
        if Fc1 > max_Fc1:
            max_Fc1 = Fc1

        F1_K = []
        for K in np.arange(0, 1.1, 0.1):
            F1_K.append(cal_F1_K(K, pred.copy(), events, label))
        AUC_F1_K = np.trapz(np.array(F1_K), np.arange(0, 1.1, 0.1))
        if AUC_F1_K > max_F1_K:
            max_F1_K = AUC_F1_K
    return max_Fc1, max_F1_K


def cal_Fc1(pred, events, label):
    tp = np.sum([pred[start:end].any() for start, end in events])
    fn = len(events) - tp
    rec_e = tp / (tp + fn)
    pre_t = precision_score(label, pred)
    if rec_e == 0 or pre_t == 0:
        return 0
    Fc1 = 2 * rec_e * pre_t / (rec_e + pre_t)
    return Fc1


def cal_F1_K(K, pred, events, label):
    for start, end in events:
        if np.sum(pred[start:end]) > K * (end - start):
            pred[start:end] = 1
    return f1_score(label, pred)
