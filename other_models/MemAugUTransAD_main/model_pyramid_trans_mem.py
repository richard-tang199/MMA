import numpy as np
import torch
import torch.nn as nn
import tqdm
from other_models.MemAugUTransAD_main.modules import ConvLayer, MaxPoolLayer
from other_models.MemAugUTransAD_main.transformer import EncoderLayer, DecoderLayer
from other_models.MemAugUTransAD_main.memory import MemoryLocal


class PYRAMID_TRANS_MEM(nn.Module):
    """ EMB_2TRANS model class.
    """

    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            kernel_size=7,
            dropout=0.2
    ):
        super(PYRAMID_TRANS_MEM, self).__init__()
        self.window_size = window_size
        self.out_dim = out_dim

        if n_features <= 32:
            self.f_dim = 32
        elif n_features <= 64:
            self.f_dim = 64
        elif n_features <= 128:
            self.f_dim = 128

        self.conv = ConvLayer(n_features, kernel_size)
        self.conv_dim = nn.Linear(n_features, self.f_dim)

        self.fuse_type = 0  # 0:同时使用pyramid and multi-resolution  1:仅使用pyramid  2:仅使用multi-resolution
        self.layer_num = 3  # 1:一层  2:两层 3:三层
        self.use_mem = 1  # 0:不使用   1:使用local memory module
        self.use_pyramid = 1  # 0:不进行下采用和上采样   1:使用下采样和上采样
        self.num_slots = 64  # 32 64 128 256

        heads = 4
        self.win2 = int(self.window_size / 2)
        self.win3 = int(self.win2 / 2)
        self.win4 = int(self.win3 / 2 + 0.5)

        if self.layer_num >= 1:
            self.enc_layer1 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem1 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer1 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)

        if self.layer_num >= 2:
            self.enc_layer2 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem2 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer2 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_pyramid == 1:
                self.d_sampling2 = MaxPoolLayer()
                self.u_sampling2 = nn.Conv1d(in_channels=self.win2, out_channels=self.window_size, kernel_size=3,
                                             padding=1)

        if self.layer_num >= 3:
            self.enc_layer3 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem3 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer3 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_pyramid == 1:
                self.d_sampling3 = MaxPoolLayer()
                self.u_sampling3 = nn.Conv1d(in_channels=self.win3, out_channels=self.win2, kernel_size=3, padding=1)

        if self.layer_num == 4:
            self.enc_layer4 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem4 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer4 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_pyramid == 1:
                self.d_sampling4 = MaxPoolLayer()
                self.u_sampling4 = nn.Conv1d(in_channels=self.win4, out_channels=self.win3, kernel_size=3, padding=1)

        # mlp layer to resconstruct output
        self.mlp = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.out_dim)
        )

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # print('x:', x.shape)
        raw = x.clone()
        x = self.conv(x)
        x = self.conv_dim(x)

        if self.layer_num == 1:
            enc1, _ = self.enc_layer1(x)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                dec1, _, _ = self.dec_layer1(x, mem1)
                # memory loss
                loss_mem = weight1
            else:
                dec1, _, _ = self.dec_layer1(x, enc1)

        elif self.layer_num == 2:
            enc1, _ = self.enc_layer1(x)
            if self.use_pyramid == 1:
                down2 = self.d_sampling2(enc1)
                enc2, _ = self.enc_layer2(down2)
            else:
                enc2, _ = self.enc_layer2(enc1)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                mem2, weight2 = self.mem2(enc2)
                if self.use_pyramid == 1:
                    dec2, _, _ = self.dec_layer2(down2, mem2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, mem1)
                else:
                    dec2, _, _ = self.dec_layer2(enc1, mem2)
                    dec1, _, _ = self.dec_layer1(dec2, mem1)
                # memory loss
                loss_mem = weight1 + weight2
            else:
                if self.use_pyramid == 1:
                    dec2, _, _ = self.dec_layer2(down2, enc2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, enc1)
                else:
                    dec2, _, _ = self.dec_layer2(enc1, enc2)
                    dec1, _, _ = self.dec_layer1(dec2, enc1)

        elif self.layer_num == 3:
            enc1, _ = self.enc_layer1(x)
            if self.use_pyramid == 1:
                down2 = self.d_sampling2(enc1)
                enc2, _ = self.enc_layer2(down2)
                down3 = self.d_sampling3(enc2)
                enc3, _ = self.enc_layer3(down3)
            else:
                enc2, _ = self.enc_layer2(enc1)
                enc3, _ = self.enc_layer3(enc2)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                mem2, weight2 = self.mem2(enc2)
                mem3, weight3 = self.mem3(enc3)
                if self.use_pyramid == 1:
                    dec3, _, _ = self.dec_layer3(down3, mem3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, mem2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, mem1)
                else:
                    dec3, _, _ = self.dec_layer3(enc2, mem3)
                    dec2, _, _ = self.dec_layer2(dec3, mem2)
                    dec1, _, _ = self.dec_layer1(dec2, mem1)
                # memory loss
                loss_mem = weight1 + weight2 + weight3
            else:
                if self.use_pyramid == 1:
                    dec3, _, _ = self.dec_layer3(down3, enc3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, enc2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, enc1)
                else:
                    dec3, _, _ = self.dec_layer3(enc2, enc3)
                    dec2, _, _ = self.dec_layer2(dec3, enc2)
                    dec1, _, _ = self.dec_layer1(dec2, enc1)

        elif self.layer_num == 4:
            enc1, _ = self.enc_layer1(x)
            if self.use_pyramid == 1:
                down2 = self.d_sampling2(enc1)
                enc2, _ = self.enc_layer2(down2)
                down3 = self.d_sampling3(enc2)
                enc3, _ = self.enc_layer3(down3)
                down4 = self.d_sampling3(enc3)
                enc4, _ = self.enc_layer3(down4)
            else:
                enc2, _ = self.enc_layer2(enc1)
                enc3, _ = self.enc_layer3(enc2)
                enc4, _ = self.enc_layer3(enc3)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                mem2, weight2 = self.mem2(enc2)
                mem3, weight3 = self.mem3(enc3)
                mem4, weight4 = self.mem3(enc4)
                if self.use_pyramid == 1:
                    dec4, _, _ = self.dec_layer3(down4, mem4)
                    up4 = self.u_sampling3(dec4)
                    dec3, _, _ = self.dec_layer3(up4, mem3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, mem2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, mem1)
                else:
                    dec4, _, _ = self.dec_layer3(enc3, mem4)
                    dec3, _, _ = self.dec_layer3(dec4, mem3)
                    dec2, _, _ = self.dec_layer2(dec3, mem2)
                    dec1, _, _ = self.dec_layer1(dec2, mem1)
                # memory loss
                loss_mem = weight1 + weight2 + weight3
            else:
                if self.use_pyramid == 1:
                    dec4, _, _ = self.dec_layer3(down4, enc4)
                    up4 = self.u_sampling3(dec4)
                    dec3, _, _ = self.dec_layer3(up4, enc3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, enc2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, enc1)
                else:
                    dec4, _, _ = self.dec_layer3(enc3, enc4)
                    dec3, _, _ = self.dec_layer3(dec4, enc3)
                    dec2, _, _ = self.dec_layer2(dec3, enc2)
                    dec1, _, _ = self.dec_layer1(dec2, enc1)

        recon = self.mlp(dec1)
        loss_fn = torch.nn.MSELoss()

        if self.use_mem == 0:
            loss_mem = torch.zeros([1]).cuda()

        loss_mem = 0.001 * loss_mem
        recon_loss = loss_fn(recon, raw)
        total_loss = recon_loss + loss_mem

        return recon, recon_loss, loss_mem, total_loss

    def fit(self, train_loader, epochs, optimizer, device="cuda:0"):
        self.train()
        for epoch in tqdm.trange(epochs):
            recon_loss_list = []
            total_loss_list = []
            mem_loss_list = []
            for (data,) in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon, recon_loss, mem_loss, total_loss = self(data)
                total_loss.backward()
                optimizer.step()
                recon_loss_list.append(recon_loss.item())
                total_loss_list.append(total_loss.item())
                mem_loss_list.append(mem_loss.item())
            tqdm.tqdm.write("Epoch: {} / {}, Total Loss: {:.6f}, Recon Loss: {:.6f},  Mem Loss: {:.6f}".format(
                epoch,
                epochs,
                np.mean(total_loss_list),
                np.mean(recon_loss_list),
                np.mean(mem_loss_list)
            ))

    def predict(self, test_loader, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            recons_list = []
            for (data,) in test_loader:
                data = data.to(device)
                recon, _, _, _ = self(data)
                recons_list.append(recon.cpu().numpy())

        recons = np.concatenate(recons_list, axis=0)
        return recons
