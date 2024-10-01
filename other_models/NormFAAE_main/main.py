import argparse
import time

import torch
from torch import nn
from other_models.NormFAAE_main.data_loader import get_dataset
from other_models.NormFAAE_main.filter import TransformerEncoder
from other_models.NormFAAE_main import utils_
import thop


class AdapStand(nn.Module):
    # Deep hybrid normalization module
    def __init__(self, n_features, mea_, std_, min_, dis_, con_, device):
        super(AdapStand, self).__init__()
        self.EPS = 1E-8
        self.mea_ = torch.tensor(mea_).to(torch.float32).to(device).detach_()
        self.std_ = torch.tensor(std_).to(torch.float32).to(device).detach_()
        self.min_ = torch.tensor(min_).to(torch.float32).to(device).detach_()
        self.dis_ = torch.tensor(dis_).to(torch.float32).to(device).detach_()
        self.con_ = torch.tensor(con_).to(torch.float32).to(device).detach_()
        self.L_mea = nn.Linear(n_features, n_features, bias=False)
        self.L_std = nn.Linear(n_features, n_features, bias=False)
        self.L_min = nn.Linear(n_features, n_features, bias=False)
        self.L_dis = nn.Linear(n_features, n_features, bias=False)
        self.L_con = nn.Linear(n_features, n_features, bias=False)
        self.my_init(self.L_mea.weight.data)
        self.my_init(self.L_std.weight.data)
        self.my_init(self.L_min.weight.data)
        self.my_init(self.L_dis.weight.data)
        self.my_init(self.L_con.weight.data)

    def __call__(self, X):
        X1 = (X - self.L_mea(self.mea_)) / (torch.abs(self.L_std(self.std_)) + self.EPS)
        X2 = (X - self.L_min(self.min_)) / (torch.abs(self.L_dis(self.dis_)) + self.EPS)
        con = torch.round(torch.sigmoid(self.L_con(self.con_)))
        return X1 * con + X2 * (1 - con)

    def my_init(self, tensor):
        if tensor.ndimension() != 2:
            raise ValueError("Only tensors with 2 dimensions are supported")
        with torch.no_grad():
            torch.add(tensor, torch.eye(tensor.shape[0]), out=tensor)
        return tensor


class Encoder(nn.Module):
    # GRU Encoder
    def __init__(self, n_features, num_hiddens):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(n_features, num_hiddens, batch_first=True)

    def __call__(self, X):
        _, H = self.rnn(X)
        return H.squeeze(0)


class Decoder(nn.Module):
    # GRU Decoder
    def __init__(self, n_features, num_hiddens):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(num_hiddens, num_hiddens, batch_first=True)
        self.out = nn.Linear(num_hiddens, n_features, bias=False)

    def __call__(self, H, n_step):
        output, _ = self.rnn(H.repeat(n_step, 1, 1).permute(1, 0, 2), H.unsqueeze(0))
        return torch.flip(self.out(output), dims=[1])


# Training and testing process
def train_test(n_features, num_hiddens, num_epochs, lr1, lr2, weight_decay, patience, data_name, model_path,
               train_data, valid_data, test_data, mea_, std_, dis_, min_, con_, alpha, Lambda, device):
    # Define the model
    stander = AdapStand(n_features, mea_, std_, min_, dis_, con_, device).to(device)
    encoder = Encoder(n_features, num_hiddens).to(device)
    decoder = Decoder(n_features, num_hiddens).to(device)
    filter = TransformerEncoder(n_features, num_hiddens, num_hiddens).to(device)

    # Define the training parameters
    optim_ae = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                lr=lr1, weight_decay=weight_decay)
    optim_fil = torch.optim.Adam(filter.parameters(), lr=lr1, weight_decay=weight_decay)
    optim_std = torch.optim.Adam(stander.parameters(), lr=lr2, weight_decay=weight_decay)
    mse = torch.nn.MSELoss(reduction='mean')
    early_stopping = utils_.EarlyStopping(patience, data_name, model_path)

    # Training process
    for epoch in range(num_epochs):
        stander.train()
        encoder.train()
        decoder.train()
        filter.train()
        metric = utils_.Accumulator(5)

        for X in train_data:
            X = X.to(device)
            b_size, n_step, n_feature = X.shape[0], X.shape[1], X.shape[2]

            std_X = stander(X)
            X_h1 = decoder(encoder(std_X), n_step)
            X_h2 = decoder(encoder(X_h1), n_step)
            res1 = filter(std_X)
            rec1 = mse(X_h1, (std_X - res1 * alpha))
            rec2 = mse(X_h2, X_h1)
            loss1 = rec1 + rec2  # loss of encoder, decoder, and stander
            optim_ae.zero_grad()
            optim_std.zero_grad()
            loss1.backward()
            utils_.grad_clipping(encoder, 1)
            utils_.grad_clipping(decoder, 1)
            utils_.grad_clipping(stander, 1)
            optim_ae.step()
            optim_std.step()

            std_X = stander(X)
            X_h1 = decoder(encoder(std_X), n_step)
            res1 = filter(std_X)
            res2 = filter(X_h1)
            nom1 = mse(res1, (std_X - X_h1))
            nom2 = torch.norm(res1, p=1)
            nom3 = torch.norm(res2, p=1)
            loss2 = nom1 + nom2 + nom3  # loss of filter
            optim_fil.zero_grad()
            loss2.backward()
            utils_.grad_clipping(filter, 1)
            optim_fil.step()
            with torch.no_grad():
                metric.add(rec1, rec2, nom1, nom2, nom3)

        with torch.no_grad():
            stander.eval()
            encoder.eval()
            decoder.eval()
            valid_loss = 0.0
            for X in valid_data:
                X = X.to(device)
                b_size, n_step, n_feature = X.shape[0], X.shape[1], X.shape[2]

                std_X = stander(X)
                X_h = decoder(encoder(std_X), n_step)
                valid_loss += mse(X_h, std_X) / mse(std_X, torch.zeros((b_size, n_step, n_feature), device=device))

            print(f"epochs: {epoch + 1}")
            print(f"{metric[0]} | {metric[1]} | {metric[2]} | {metric[3]} | {metric[4]}")
            print(f"V: {valid_loss}")
            early_stopping(valid_loss, encoder, decoder, stander)
            if early_stopping.early_stop:
                print("EarlyStopping！")
                break

    # Testing process
    stander.load_state_dict(torch.load(model_path + data_name + '_stander.pth'))
    encoder.load_state_dict(torch.load(model_path + data_name + '_encoder.pth'))
    decoder.load_state_dict(torch.load(model_path + data_name + '_decoder.pth'))
    stander.eval()
    encoder.eval()
    decoder.eval()
    test_loss = torch.nn.MSELoss(reduction='none')
    label, score = [], []
    raw_x = []
    recons_x = []
    start_time = time.time()
    for data in test_data:
        X, y = data[0], data[1]
        X = X.to(device)
        b_size, n_step, n_feature = X.shape[0], X.shape[1], X.shape[2]
        std_X = stander(X)
        raw_x.append(std_X)
        X_hat = decoder(encoder(std_X), n_step)
        recons_x.append(X_hat)
        l = test_loss(X_hat, std_X)
        l = torch.mean(l, dim=-1, keepdim=False)
        y = (torch.reshape(y, (-1, 1))).squeeze(dim=-1)
        l = (torch.reshape(l, (-1, 1))).squeeze(dim=-1)
        label.append(y.cpu().detach())
        score.append(l.cpu().detach())
    sample_input = torch.randn(1, n_step, n_features, device=device)
    flops_1, params_1 = thop.profile(stander, inputs=(sample_input,))
    sample_input = stander(sample_input)
    flops_2, params_2 = thop.profile(encoder, inputs=(sample_input,))
    sample_input = encoder(sample_input)
    flops_3, params_3 = thop.profile(decoder, inputs=(sample_input, n_step))
    flops, params = flops_1 + flops_2 + flops_3, params_1 + params_2 + params_3
    flops, params = thop.clever_format([flops, params])

    label = torch.cat(label, dim=0).numpy()
    score = torch.cat(score, dim=0).numpy()
    raw_x = torch.cat(raw_x, dim=0).reshape(-1, n_features).detach().cpu().numpy()
    recons_x = torch.cat(recons_x, dim=0).reshape(-1, n_features).detach().cpu().numpy()
    duration = time.time() - start_time
    return label, score, raw_x, recons_x, flops, params, duration


if __name__ == '__main__':
    # python main.py --data 'MSL'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data name', type=str, default='MSL')
    parser.add_argument('--seed', help='random seed', type=int, default=3407)
    parser.add_argument('--device', help='device', type=int, default=0)
    parser.add_argument('--data_path', help='data path', type=str, default="data/")
    parser.add_argument('--model_path', help='model save path', type=str, default='model/')
    parser.add_argument('--score_path', help='score save path', type=str, default="score/")
    args = parser.parse_args()

    utils_.setup_seed(args.seed)
    device = utils_.try_gpu(args.device)
    data_name = args.data
    data_path, model_path, score_path = args.data_path, args.model_path, args.score_path
    utils_.creat_path(model_path)
    utils_.creat_path(score_path)

    num_epochs, patience, lr1, lr2, weight_decay = 1000, 5, 1e-4, 1e-4, 1e-4
    batch_size, step, win_size, num_hiddens = 128, 8, 128, 128
    alpha, Lambda = 1, 1

    train, valid, test, dim, mea, std, dis, min, con = get_dataset(data_name, data_path, batch_size, win_size, step)
    label, score = train_test(dim, num_hiddens, num_epochs, lr1, lr2, weight_decay, patience, data_name, model_path,
                              train, valid, test, mea, std, dis, min, con, alpha, Lambda, device)
    utils_.get_metrics(label, score, score_path, data_name)
