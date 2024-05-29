import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm


class Expert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out):
        super(Expert, self).__init__()
        self.conv = nn.Conv2d(1, n_kernel, (window, 1))
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(n_kernel * n_multiv, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(dim=1).contiguous()
        x = F.relu(self.conv(x))
        x = self.dropout(x)

        out = torch.flatten(x, start_dim=1).contiguous()

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_out):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class MMoE(nn.Module):
    def __init__(self, n_multiv, window_size):
        super(MMoE, self).__init__()
        self.n_multiv = n_multiv
        self.n_kernel = 16
        self.window = window_size
        self.num_experts = 5
        self.experts_out = 128
        self.experts_hidden = 256
        self.towers_hidden = 32
        self.sg_ratio = 0.7

        # task num = n_multiv
        self.tasks = n_multiv
        self.criterion = "l2"
        self.exp_dropout = 0.2
        self.tow_dropout = 0.1
        self.conv_dropout = 0.1
        self.lr = 1e-3

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList(
            [Expert(self.n_kernel, self.window, self.n_multiv, self.experts_hidden, self.experts_out, self.exp_dropout) \
             for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True) \
                                         for i in range(self.tasks)])
        self.share_gate = nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True)
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden, self.tow_dropout) \
                                     for i in range(self.tasks)])

    def forward(self, x):
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)

        gates_out = [self.softmax(
            (x[:, :, i] @ self.w_gates[i]) * (1 - self.sg_ratio) + (x[:, :, i] @ self.share_gate) * self.sg_ratio)
            for i in range(self.tasks)]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_out_tensor for g in gates_out]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        tower_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        tower_output = torch.stack(tower_output, dim=0).permute(1, 2, 0)

        final_output = tower_output
        return final_output

    def loss(self, labels, predictions):
        if self.criterion == "l1":
            loss = F.l1_loss(predictions, labels)
        elif self.criterion == "l2":
            loss = F.mse_loss(predictions, labels)
        return loss

    def fit(self, train_loader, epochs, device="cuda:0"):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        for epoch in tqdm.trange(epochs):
            loss_list = []
            for (data,) in train_loader:
                data = data.to(device)
                x = data[:, :-3, :]
                y = data[:, -1, :]
                y_hat_ = self.forward(x)
                y_hat_ = y_hat_.squeeze()
                loss = self.loss(y.squeeze(-1), y_hat_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            scheduler.step()
            print("Epoch: %d, Loss: %.6f" % (epoch, np.mean(loss_list)))

    def predict(self, test_loader, device="cuda:0"):
        self.eval()
        predictions = []
        with torch.no_grad():
            for (data,) in test_loader:
                data = data.to(device)
                x = data[:, :-3, :]
                y = data[:, -1, :]
                y_hat_ = self.forward(x)
                predictions.append(y_hat_.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        predictions = np.squeeze(predictions)
        return predictions
