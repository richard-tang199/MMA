import os.path
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model.PatchTSMixerLayer import *
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Transformation(nn.Module):
    def __init__(self, modes, num=8):
        """
        @param mode: transformation mode
        """
        super(Transformation, self).__init__()
        self.num = num
        for mode in modes:
            assert mode in ('flip', 'uniform', 'amplitude', 'trend'), f'Invalid transformation mode: {mode}'
        self.modes = modes
        self.transform = {
            "flip": self._flip,
            "uniform": self._uniform,
            "amplitude": self._amplitude,
            "trend": self._trend
        }

    # TODO: deal with 0
    def forward(self, input: Tensor) -> Tensor:
        """
        @param input: (batch_size, num_channels, num_patches, patch_length)
        @return: transformed input: (batch_size, num_channels, num_patches, augment_num, patch_length)
        """
        transformed_inputs = None
        for _ in range(self.num):
            for mode in self.modes:
                transformed_input = self.transform[mode](input)
                transformed_input = transformed_input.unsqueeze(-2)
                if transformed_inputs is None:
                    transformed_inputs = transformed_input
                else:
                    transformed_inputs = torch.concatenate([transformed_inputs, transformed_input], dim=-2)

        return transformed_inputs

    def _flip(self, input: Tensor) -> Tensor:
        patch_mean = input.mean(dim=-1, keepdim=True)
        input_mirror = 2 * patch_mean - input
        input_mirror = self._amplitude(input_mirror)
        return input_mirror

    # TODO: only shape
    def _uniform(self, input: Tensor, ratio: int = 2) -> Tensor:
        batch_size, num_channels, num_patches, patch_length = input.shape
        patch_mean = input.mean(dim=-1, keepdim=True).repeat(1, 1, 1, patch_length)
        patch_mean = self._amplitude(patch_mean)
        # ratio = ratio * torch.rand(size=[batch_size, num_channels, num_patches, 1],
        #                            device=input.device)
        # # avoid too close to origin value
        # ratio[(ratio > 0.9) & (ratio < 1.0)] = 0.9 - 0.9 * torch.rand(1)
        # ratio[(ratio > 1.0) & (ratio < 1.1)] = 1.1 + torch.rand(1)
        # # avoid patch mean is too close zero
        # patch_mean = patch_mean * ratio
        # patch_mean[(patch_mean <= 0.2) & (patch_mean >= 0)] = 0.2 + 0.2 * torch.rand(1)
        # patch_mean[(patch_mean >= -0.2) & (patch_mean < 0)] = -0.2 - 0.2 * torch.rand(1)
        return patch_mean

    @staticmethod
    def _amplitude(input: Tensor) -> Tensor:
        batch_size, num_channels, num_patches, patch_length = input.shape
        noise = 2 * torch.rand(size=[batch_size, num_channels, num_patches, 1], device=input.device) - 1
        # avoid noise too close to zero
        noise[(noise < 0.2) & (noise >= 0)] = 0.2
        noise[(noise > -0.2) & (noise < 0)] = -0.2
        output = input + noise
        output[output < 0] = -output[output < 0]
        output[output > 1] = 2 - output[output > 1]
        return output

    def _trend(self, input: Tensor, ratio: tuple[float, float] = (0.1, 0.5)) -> Tensor:
        batch_size, num_channels, num_patches, patch_length = input.shape
        choose_ratio = np.random.uniform(low=ratio[0], high=ratio[1])
        new_length = int(patch_length * choose_ratio)
        start = np.random.randint(0, patch_length - new_length)
        _slice = slice(start, start + new_length)
        input = input.reshape(batch_size * num_channels, num_patches, -1)
        interplot_input = F.interpolate(input[..., _slice], size=patch_length, mode='linear')
        interplot_input = interplot_input.reshape(batch_size, num_channels, num_patches, -1)
        interplot_input = self._amplitude(interplot_input)
        return interplot_input

def contrastive_loss(mask_embedding: Tensor,
                     origin_embedding: Tensor,
                     batch_size: int,
                     num_channels: int,
                     temperature: float = 0.1,
                     mask_pos: Tensor = None,
                     augmented_embedding: Tensor = None
                     ) -> Tensor:
    """
    Compute the contrastive
    @param temperature: contrastive temperature
    @param mask_embedding: [bs x n_vars x num_patch x d_model]
    @param origin_embedding: [bs x n_vars x num_patch x d_model]
    @param augmented_embedding: [bs x n_vars x num_patch x augment_num x d_model]
    @param mask_pos: [mask_num x 3]
    @return: loss
    """
    if mask_pos is not None:
        mask_embedding = mask_embedding[mask_pos[:, 0], mask_pos[:, 1], mask_pos[:, 2], :]
        origin_embedding = origin_embedding[mask_pos[:, 0], mask_pos[:, 1], mask_pos[:, 2], :]
        _, d_model = mask_embedding.shape
        # reshape to [bs x n_vars x num_patches x d_model]
        mask_embedding = mask_embedding.reshape(batch_size, num_channels, -1, d_model)
        origin_embedding = origin_embedding.reshape(batch_size, num_channels, -1, d_model)

    _, _, num_patches, _ = mask_embedding.shape

    loss_fun = nn.CrossEntropyLoss()
    # mask_origin_similarity [bs x n_vars x num_patches x num_patches]
    mask_origin_similarity = torch.matmul(mask_embedding, origin_embedding.transpose(-1, -2)) / temperature
    mask_origin_similarity = mask_origin_similarity.reshape(batch_size * num_channels, num_patches, num_patches)
    label = torch.eye(num_patches, device=mask_origin_similarity.device)
    label = label.unsqueeze(0).repeat(batch_size * num_channels, 1, 1)
    loss = loss_fun(mask_origin_similarity, label)

    if augmented_embedding is not None:
        # augmented_embedding: [mask_num x aug_num x d_model]
        augmented_embedding = augmented_embedding[mask_pos[:, 0], mask_pos[:, 1], mask_pos[:, 2], :, :]
        _, aug_num, _ = augmented_embedding.shape
        augmented_embedding = augmented_embedding.reshape(batch_size, num_channels, -1, aug_num, d_model)
        # only take Diagonal element [mask_num x 1]
        mask_origin_similarity = torch.diagonal(mask_origin_similarity, dim1=-2, dim2=-1).unsqueeze(-1)
        # mask_augmented_similarity [mask_num x 4]
        mask_augmented_similarity = torch.matmul(mask_embedding.unsqueeze(-2),
                                                 augmented_embedding.transpose(-1, -2)).squeeze(-2)
        # obtain similarity matrix [mask_num x (1 + 4)]
        mask_augmented_similarity = mask_augmented_similarity.reshape(batch_size * num_channels, num_patches, -1)
        all_similarity = torch.cat([mask_origin_similarity, mask_augmented_similarity], dim=-1) / temperature
        all_similarity = all_similarity.reshape(-1, aug_num + 1)
        label = torch.ones(size=(all_similarity.shape[0], aug_num + 1), device=all_similarity.device)
        label[:, 1:] = 0
        weight = torch.ones(aug_num + 1, device=all_similarity.device) / aug_num
        weight[0] = 1

        loss_fun = nn.CrossEntropyLoss(reduction="mean", weight=weight)
        loss = loss_fun(all_similarity, label)

    return loss


class PatchTSMixerContrastBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchDetectorConfig):
        super().__init__()

        num_layers = config.num_layers

        self.mixers = nn.ModuleList([PatchTSMixerLayer(config=config) for _ in range(num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state  # [bs x n_vars x num_patch x d_model]

        for mod in self.mixers:
            embedding = mod(embedding)
            embedding = embedding / torch.sqrt(torch.sum(embedding ** 2, dim=-1, keepdim=True) + 1e-4)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchEmbedding(nn.Module):
    def __init__(self, config: PatchDetectorConfig):
        super().__init__()
        self.proj = nn.Linear(config.patch_length, config.d_model)
        if config.use_position_encoder:
            self.positional_encoder = PositionalEncoding(config, config.num_patches)
        else:
            self.positional_encoder = None

    def forward(self, patches: torch.Tensor):
        patches = self.proj(patches)
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        patches = patches / torch.sqrt(torch.sum(patches ** 2, dim=-1, keepdim=True) + 1e-4)
        return patches


class PatchMixerContrastEncoder(nn.Module):
    def __init__(self, config: PatchDetectorConfig):
        super().__init__()
        self.patch_embedding = PatchEmbedding(config)
        if config.use_position_encoder:
            self.positional_encoder = PositionalEncoding(config, config.num_patches)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = PatchTSMixerContrastBlock(config=config)

    def forward(self,
                patch_inputs: Tensor,
                output_hidden_states: Optional[bool] = True):
        """
        Parameters:
           patch_inputs (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
        """

        patches = self.patch_embedding(patch_inputs)  # patches [bs x n_vars x num_patch x d_model]

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(
            hidden_state=patches,
            output_hidden_states=output_hidden_states)

        return PatchTSMixerEncoderOutput(last_hidden_state=last_hidden_state,
                                         hidden_states=hidden_states)


if __name__ == '__main__':
    # batch_size = 2
    # num_channels = 3
    # num_patches = 8
    # patch_length = 16
    # input = torch.randn(batch_size, num_channels, num_patches, patch_length, device="cuda")
    # augmenter = Transformation(modes=("flip", "uniform", "amplitude", "trend"))
    # augmented_input = augmenter(input)
    # print(augmented_input.shape)
    seq_len = 512
    time = np.arange(0, seq_len / 10, 0.1)
    trend = np.sin(np.arange(0, 1, 1 / seq_len) * np.pi) * 2
    a1 = np.sin(time) + trend
    a2 = np.cos(time) + trend
    a3 = np.sin(time) + np.cos(time) + trend
    a4 = np.zeros_like(time)
    signal = np.stack([a1, a2, a3, a4], axis=0)
    scaler = MinMaxScaler()
    signal = scaler.fit_transform(signal.transpose()).transpose()
    a5 = np.ones_like(a4)
    a5 = a5[np.newaxis, :]
    signal = np.concatenate([signal, a5], axis=0)
    num_channels, seq_len = signal.shape

    # raw plot
    fig, axes = plt.subplots(
        nrows=num_channels,
        ncols=1,
        figsize=(10, 5),
        tight_layout=True
    )

    for i in range(num_channels):
        axes[i].plot(signal[i, :])

    save_dir = "../augment_analysis"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "raw.png"), dpi=100)

    # test
    patch_length = 16
    signal_input = torch.tensor(signal).unsqueeze(0)

    # signal_input: (batch_size, num_channels, num_patches, patch_length)
    signal_input = signal_input.unfold(dimension=-1,
                                       size=patch_length,
                                       step=patch_length)
    # choose several windows
    choose_index = random.choices(range(signal_input.shape[-2]), k=5)
    other_index = [i for i in range(signal_input.shape[-2]) if i not in choose_index]
    choose_windows = signal_input[:, :, choose_index, :]

    augmenter = Transformation(modes=("flip", "uniform", "amplitude", "trend"))
    augment_output = augmenter(choose_windows)

    output_list = []

    for i in range(0, 4):
        output = torch.zeros_like(signal_input)
        output[..., choose_index, :] = augment_output[..., i, :]
        output[..., other_index, :] = signal_input[..., other_index, :]
        flattened = output.flatten(-2).squeeze(0)
        output_list.append(flattened)

    for index in range(len(output_list)):
        use_output = output_list[index]
        fig, axes = plt.subplots(
            nrows=num_channels,
            ncols=1,
            figsize=(10, 5),
            tight_layout=True
        )
        for i in range(num_channels):
            axes[i].scatter(np.arange(use_output.shape[-1]), use_output[i, :], s=1)
            axes[i].scatter(np.arange(use_output.shape[-1]), signal[i, :], s=1, c='y', alpha=0.5, marker='x')

        plt.savefig(os.path.join(save_dir, f"aug_{index}.png"), dpi=100)
        plt.close()
