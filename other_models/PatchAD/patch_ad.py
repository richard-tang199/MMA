import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from einops import rearrange, repeat
from PatchAD.patchad_model.models import PatchMLPAD
import tqdm

def my_kl_loss(p, q):
    # B N D
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    # B N
    return torch.sum(res, dim=-1)


def inter_intra_dist(p, q, w_de=True, train=1, temp=1):
    # B N D
    if train:
        if w_de:
            p_loss = torch.mean(my_kl_loss(p, q.detach() * temp)) + torch.mean(my_kl_loss(q.detach(), p * temp))
            q_loss = torch.mean(my_kl_loss(p.detach(), q * temp)) + torch.mean(my_kl_loss(q, p.detach() * temp))
        else:
            p_loss = -torch.mean(my_kl_loss(p, q.detach()))
            q_loss = -torch.mean(my_kl_loss(q, p.detach()))
    else:
        if w_de:
            p_loss = my_kl_loss(p, q.detach()) + my_kl_loss(q.detach(), p)
            q_loss = my_kl_loss(p.detach(), q) + my_kl_loss(q, p.detach())

        else:
            p_loss = -(my_kl_loss(p, q.detach()))
            q_loss = -(my_kl_loss(q, p.detach()))

    return p_loss, q_loss


def normalize_tensor(tensor):
    # tensor: B N D
    sum_tensor = torch.sum(tensor, dim=-1, keepdim=True)
    normalized_tensor = tensor / sum_tensor
    return normalized_tensor


def anomaly_score(patch_num_dist_list, patch_size_dist_list, win_size, train=1, temp=1, w_de=True):
    for i in range(len(patch_num_dist_list)):
        patch_num_dist = patch_num_dist_list[i]
        patch_size_dist = patch_size_dist_list[i]

        patch_num_dist = repeat(patch_num_dist, 'b n d -> b (n rp) d', rp=win_size // patch_num_dist.shape[1])
        patch_size_dist = repeat(patch_size_dist, 'b p d -> b (rp p) d', rp=win_size // patch_size_dist.shape[1])

        patch_num_dist = normalize_tensor(patch_num_dist)
        patch_size_dist = normalize_tensor(patch_size_dist)

        patch_num_loss, patch_size_loss = inter_intra_dist(patch_num_dist, patch_size_dist, w_de, train=train,
                                                           temp=temp)

        if i == 0:
            patch_num_loss_all = patch_num_loss
            patch_size_loss_all = patch_size_loss
        else:
            patch_num_loss_all += patch_num_loss
            patch_size_loss_all += patch_size_loss

    return patch_num_loss_all, patch_size_loss_all


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = val_loss
        score2 = val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        print('Save model')
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(torch.nn.Module):
    def __init__(self, epochs, window_size, channels):
        super(Solver, self).__init__()
        self.patch_size = [3, 5, 7]
        self.num_epochs = epochs
        self.lr = 0.0001
        self.win_size = window_size
        self.input_c = channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_mx = 0.1
        self.cont_beta = 0.5
        self.build_model()

    def build_model(self):
        self.model = PatchMLPAD(win_size=self.win_size, e_layer=3, patch_sizes=self.patch_size, dropout=0.1,
                                activation="relu", output_attention=True,
                                channel=self.input_c, d_model=60, cont_model=self.win_size, norm='n')

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        win_size = self.win_size
        loss_mse = nn.MSELoss()
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)

            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                            win_size=win_size, train=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            # loss3 = patch_size_loss + patch_num_loss

            p_loss = patch_size_loss
            q_loss = patch_num_loss

            loss_1.append((p_loss).item())
            loss_2.append((q_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def fit(self, train_loader):
        time_now = time.time()
        win_size = self.win_size
        train_steps = len(train_loader)

        mse_loss = nn.MSELoss()

        for epoch in tqdm.trange(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data,) in enumerate(train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                # input = input + torch.rand_like(input)*0.2

                patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(
                    input)

                loss = 0.

                cont_loss1, cont_loss2 = anomaly_score(patch_num_dist_list, patch_size_mx_list, win_size=win_size,
                                                       train=1, temp=1)
                cont_loss_1 = cont_loss1 - cont_loss2
                loss -= self.patch_mx * cont_loss_1

                cont_loss12, cont_loss22 = anomaly_score(patch_num_mx_list, patch_size_dist_list, win_size=win_size,
                                                         train=1, temp=1)
                cont_loss_2 = cont_loss12 - cont_loss22
                loss -= self.patch_mx * cont_loss_2

                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                                win_size=win_size, train=1, temp=1)
                patch_num_loss = patch_num_loss / len(patch_num_dist_list)
                patch_size_loss = patch_size_loss / len(patch_num_dist_list)

                loss3 = patch_num_loss - patch_size_loss

                loss -= loss3 * (1 - self.patch_mx)

                loss_mse = mse_loss(recx, input)
                # print(loss_mse)
                loss += loss_mse

                loss.backward()
                self.optimizer.step()

                # self.analysis(from_file=0)
                # self.model.train()

                if (i + 1) % 20 == 0:
                    print(f'MSE {loss_mse.item()} Loss {loss.item()}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    epo_left = speed * (len(train_loader))
                    print('Epoch time left: {:.4f}s'.format(epo_left))
                    # self.test(from_file=0)
                    # self.model.train()

    @torch.no_grad()
    def test(self, test_loader, from_file=0):
        if from_file:
            print('load model from file')
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.data_name) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 1  # + (self.patch_mx*10)
        win_size = self.win_size
        use_project_score = 0
        cont_beta = self.cont_beta
        mse_loss = nn.MSELoss(reduction='none')

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []

        test_data = []
        for i, (input_data,) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list, patch_size_dist_list, patch_num_mx_list, patch_size_mx_list, recx = self.model(input)

            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list, patch_size_mx_list,
                                                                win_size=win_size, train=0, temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list, patch_size_dist_list,
                                                                win_size=win_size, train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            loss3 = patch_size_loss - patch_num_loss

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # metric = torch.softmax((-patch_num_loss ), dim=-1)
            mse_loss_ = mse_loss(recx, input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            # metric1 = -patch_num_loss
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1 - cont_beta)
            attens_energy.append(metric)

        attens_energy = torch.cat(attens_energy, dim=0).cpu().numpy()
        attens_energy = attens_energy[:,:, np.newaxis]
        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        return attens_energy
