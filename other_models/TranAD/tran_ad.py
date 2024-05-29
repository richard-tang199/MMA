from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
import torch.nn as nn
import torch
import math
import tqdm
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        # cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranAD(nn.Module):
    def __init__(self, feats, window_length):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.batch = 128
        self.n_feats = feats
        self.n_window = window_length
        self.n = self.n_feats * self.n_window
        self.lr = 0.002
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * feats, feats))

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # src: (window_length, batch, n_feats)
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

    def fit(self, train_loader, epochs, optimizer, scheduler, device="cuda:0"):
        loss_func = nn.MSELoss(reduction='mean')
        self.train()
        for epoch in tqdm.trange(epochs):
            loss_list = []
            for (data, ) in train_loader:
                data = data.to(device)
                n = epoch + 1
                # data: (batch, window_length, n_feats)
                local_batch, window_length, n_feats = data.shape
                window = data.permute(1, 0, 2) # (window_length, batch, n_feats)
                elem = window[-1, :, :].view(1, local_batch, n_feats) # (1, batch, n_feats)
                out = self.forward(window, elem)
                loss = (1 / n) * loss_func(out[0], elem) + (1 - 1 / n) * loss_func(out[1], elem)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_list.append(loss.item())
            scheduler.step()
            tqdm.tqdm.write(f'Epoch {epoch}/{epochs}, Loss: {np.mean(loss_list):.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')

    def predict(self, data_loader, device="cuda:0"):
        recon_out = None
        with torch.no_grad():
            self.eval()
            for (data, ) in data_loader:
                data = data.to(device)
                local_batch, window_length, n_feats = data.shape
                window = data.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_batch, n_feats)
                out = self.forward(window, elem)
                out = out[1] # length x batch x n_feats
                out = out.permute(1, 0, 2) # batch x length x n_feats
                if recon_out is None:
                    recon_out = out
                else:
                    recon_out = torch.cat((recon_out, out), dim=0)

            recon_out = recon_out.reshape(-1, n_feats)
            recon_out = recon_out.cpu().numpy()
            return recon_out






