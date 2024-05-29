import torch
import torch.nn as nn
import tqdm
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def forward(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def fit(self, epochs, train_loader, opt_func=torch.optim.Adam, device="cuda:0"):
        self.train()
        optimizer1 = opt_func(list(self.encoder.parameters()) + list(self.decoder1.parameters()))
        optimizer2 = opt_func(list(self.encoder.parameters()) + list(self.decoder2.parameters()))
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, 5, 0.9)
        # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 5, 0.9)
        for epoch in tqdm.trange(epochs):
            loss1_list = []
            loss2_list = []

            for (data,) in train_loader:
                data = data.to(device)
                data = data.view(data.shape[0], -1)
                # Train AE1
                loss1, loss2 = self(data, epoch + 1)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                # Train AE2
                loss1, loss2 = self(data, epoch + 1)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

            # scheduler1.step()
            # scheduler2.step()

            tqdm.tqdm.write(f"Epoch {epoch}: Loss1: {np.mean(loss1_list)}, Loss2: {np.mean(loss2_list)}")

    def predict(self, test_loader, alpha=1.0, beta=0.0, device="cuda:0"):
        results = []
        self.eval()
        with (torch.no_grad()):
            for (data,) in test_loader:
                data = data.to(device)
                batch_size, seq_len, num_channels = data.shape
                data = data.view(batch_size, -1)
                w1 = self.decoder1(self.encoder(data))
                w2 = self.decoder2(self.encoder(w1))
                w1 = w1.view(batch_size, seq_len, num_channels)
                w2 = w2.view(batch_size, seq_len, num_channels)
                recon = alpha * w1 + beta * w2
                results.append(recon)
        results = torch.cat(results, dim=0).cpu().numpy()
        return results
