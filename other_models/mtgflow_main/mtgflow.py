import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from other_models.mtgflow_main.NF import MAF
import math
import tqdm
from torch.nn.utils import clip_grad_value_

configs = {
    "n_blocks": 2,
    "window_size": 60,
    "input_size": 1,
    "hidden_size": 32,
    "n_components": 1,
    "n_hidden": 1,
    "dropout": 0.0,
    "model": "MAF",
    "batch_norm": True,
    "weight_decay": 5e-4,
    "lr": 2e-3
}


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """

    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D
        # print(h.shape, A.shape)
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # print(h.shape, A.shape)
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0], shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))

        return score, k


class MTGFLOW(nn.Module):

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True, **kwargs):
        super(MTGFLOW, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model == "MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x, ):
        return self.test(x, ).mean()

    def test(self, x, ):
        # x: batch_size X window_len X num_channels
        x = x.unsqueeze(-1)  # x: batch_size X window_len X num_channels X 1
        x = x.permute(0, 2, 1, 3)  # x: batch_size X num_channels X window_len X 1
        full_shape = x.shape
        graph, _ = self.attention(x)
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0], -1])  #
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def get_graph(self):
        return self.graph

    def locate(self, x, ):
        # x: N X K X L X D
        # x: batch_size X window_len X num_channels
        x = x.unsqueeze(-1)  # x: batch_size X window_len X num_channels X 1
        x = x.permute(0, 2, 1, 3)  # x: batch_size X num_channels X window_len X 1
        full_shape = x.shape

        graph, _ = self.attention(x)
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob = a.reshape([full_shape[0], full_shape[1], -1])

        return log_prob.mean(dim=2)

    def fit(self, data_loader, optimizer, scheduler, epochs, device="cuda:0"):
        self.train()
        for epoch in tqdm.trange(epochs):
            loss_list = []
            for (data,) in data_loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = -self.forward(data)
                loss.backward()
                clip_grad_value_(self.parameters(), 1)
                optimizer.step()
                loss_list.append(loss.item())
            print(f"Epoch {epoch}: Loss {sum(loss_list) / len(loss_list)}")
            scheduler.step()
            print(f"current learning rate: {scheduler.get_last_lr()}")

    def predict(self, data_loader, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            pred_list = []
            for (data,) in data_loader:
                data = data.to(device)
                pred = -self.locate(data).cpu().numpy()
                pred_list.append(pred)
        pred_result = np.concatenate(pred_list, axis=0)

        return pred_result
