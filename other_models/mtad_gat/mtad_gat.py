import numpy as np
import torch
import torch.nn as nn
import tqdm

from other_models.mtad_gat.modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
)



class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
        **kwargs
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons

    def fit(self, train_loader, epochs, optimizer, device="cuda:0"):
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in tqdm.trange(epochs):
            predict_loss_list = []
            recon_loss_list = []
            for (data, ) in train_loader:
                # batch_size, sequence_length, n_features
                data = data.to(device)
                x = data[:, :-1, :]
                y = data[:, -1, :]
                optimizer.zero_grad()
                predictions, recons = self(x)
                predict_loss = loss_fn(predictions, y)
                recon_loss = loss_fn(recons, x)
                loss = predict_loss + recon_loss
                loss.backward()
                optimizer.step()
                predict_loss_list.append(predict_loss.item())
                recon_loss_list.append(recon_loss.item())

            tqdm.tqdm.write(f"Epoch {epoch}/{epochs}, "
                            f"Predict Loss: {sum(predict_loss_list)/len(predict_loss_list)},"
                            f"Recon Loss: {sum(recon_loss_list)/len(recon_loss_list)}")

    def predict(self, data_loader, device="cuda:0"):
        self.eval()
        predicts = []
        recons = []

        with torch.no_grad():
            for (data, ) in tqdm.tqdm(data_loader):
                data = data.to(device)
                x = data[:, :-1, :]
                y = data[:, -1, :]
                pre, _ = self(x)
                recon_x = torch.cat([x[:, 1:, :], y.unsqueeze(-2)], dim=1)
                _, recon = self(recon_x)

                predicts.append(pre.detach().cpu().numpy())
                recons.append(recon[:, -1, :].detach().cpu().numpy())

        predicts = np.concatenate(predicts, axis=0)
        recons = np.concatenate(recons, axis=0)

        return predicts, recons




