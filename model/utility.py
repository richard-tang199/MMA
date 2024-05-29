import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from config.patchDetectorConfig import PatchDetectorConfig


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.flatten = nn.Flatten(-2)
        if self.affine:
            self._init_params()

    def forward(self, x: Tensor, mode: str):
        r"""
        x (shape '(batch_size, sequence_length, num_channels)'
        """

        # x (batch_size, num_channels, sequence_length)

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = torch.ones(self.num_features)
        self.affine_bias = torch.zeros(self.num_features)
        self.affine_weight = self.affine_weight.to(
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.affine_bias = self.affine_bias.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=-2, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=-2, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        r"""
        x (shape '(batch_size, sequence_length, num_channels)'
        """
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        @param x: (batch_size, num_channels, num_patches, patch_length)
        """
        patch_length = x.shape[-1]
        # x: batch_size X sequence_length X num_channels
        x = self.flatten(x).permute(0, 2, 1)
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        # x: (batch_size, num_patches, num_channels, patch_length)
        x = x.unfold(dimension=-2, size=patch_length, step=patch_length)
        # x: (batch_size, num_channels, num_patches, patch_length)
        x = x.transpose(-2, -3).contiguous()
        return x


class Patchify(nn.Module):
    def __init__(self, sequence_length, patch_length, patch_stride):
        super().__init__()

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, time_values: torch.Tensor):
        """
        Parameters:
            time_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for Patchify

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = time_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = time_values[:, self.sequence_start:, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


def random_masking(
        inputs: torch.Tensor,
        mask_ratio: float,
        unmasked_channel_indices: list = None,
        channel_consistent_masking: bool = False,
        mask_value: int = 0,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Masking ratio applied to mask the input data during random pretraining. It is the number between 0 and 1.
        unmasked_channel_indices (list, *optional*):
            Indices of channels that will not be masked.
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        mask_value (int, *optional*, defaults to 0):
            Define the value of masked patches for pretraining.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        # noise in [0, 1], bs x num_channels x L
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # mask: [bs x num_channels x num_patch]
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


@dataclass
class MaskOutput:
    inputs_mask_odd: Optional[torch.FloatTensor] = None
    inputs_mask_even: Optional[torch.FloatTensor] = None
    inputs_mask_odd_even: Optional[torch.FloatTensor] = None
    inputs_mask_even_odd: Optional[torch.FloatTensor] = None
    mask_odd: Optional[torch.FloatTensor] = None
    mask_even: Optional[torch.FloatTensor] = None
    mask_odd_even: Optional[torch.FloatTensor] = None
    mask_even_odd: Optional[torch.FloatTensor] = None


class MaskingStrategy(nn.Module):
    even_odd_masking: Tensor
    odd_even_masking: Tensor
    odd_odd_masking: Tensor
    even_even_masking: Tensor

    def __init__(self,
                 num_channels: int,
                 num_patches: int,
                 patch_length: int,
                 mask_ratio: float = 0.5,
                 mask: Tensor = None,
                 device: str = "cuda",
                 channel_consistent: bool = False,
                 switching: bool = True,
                 mask_value: int = 0,
                 mode: str = "random",
                 ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.channel_consistent = channel_consistent
        self.switching = switching
        self.mask_value = mask_value
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_length = patch_length
        self.device = device

        if mode not in ["random", "gating"]:
            raise ValueError("mode must be either 'random' or 'switching'")
        self.mode = mode

        if self.mode == "gating":
            self._mask_generate()

        self.mask = mask

    def _mask_generate(self):

        odd_masking = torch.tensor([i % 2 for i in range(self.num_patches)],
                                   device=self.device).unsqueeze(0).repeat(1, self.patch_length, 1)
        # odd_masking: [1 x num_patches x patch_length]
        odd_masking = torch.transpose(odd_masking, -1, -2).contiguous()

        even_masking = torch.tensor([(i + 1) % 2 for i in range(self.num_patches)],
                                    device=self.device).unsqueeze(0).repeat(1, self.patch_length, 1)
        # even_masking: [1 x num_patches x patch_length]
        even_masking = torch.transpose(even_masking, -1, -2).contiguous()

        # repeat the masking for each channel
        # odd_masking: [num_channels x num_patches x patch_length]
        # even_masking: [num_channels x num_patches x patch_length]
        self.odd_odd_masking = odd_masking.repeat(self.num_channels, 1, 1)
        self.even_even_masking = even_masking.repeat(self.num_channels, 1, 1)

        # switching masking
        odd_even_masking = torch.concat([odd_masking, even_masking], dim=0)
        even_odd_masking = torch.concat([even_masking, odd_masking], dim=0)

        self.odd_even_masking = odd_even_masking.repeat(self.num_channels // 2, 1, 1)
        self.even_odd_masking = even_odd_masking.repeat(self.num_channels // 2, 1, 1)

        if self.num_channels % 2 == 1:
            # odd_even_masking: [num_channels x num_patches x patch_length]
            self.odd_even_masking = torch.concat([self.odd_even_masking, odd_masking], dim=0)
            # even_odd_masking: [num_channels x num_patches x patch_length]
            self.even_odd_masking = torch.concat([self.even_odd_masking, even_masking], dim=0)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            time_values `(batch_size, num_channels, num_patches, patch_length)`:
                Input for Masking

        Returns:
            x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points
        """
        batch_size = inputs.shape[0]

        inputs_mask_odd = None
        inputs_mask_even = None
        inputs_mask_odd_even = None
        inputs_mask_even_odd = None
        odd_odd_masking = None
        even_even_masking = None
        odd_even_masking = None
        even_odd_masking = None

        if self.mask is None:
            if self.mode == "gating":
                odd_odd_masking = self.odd_odd_masking.repeat(batch_size, 1, 1, 1)
                even_even_masking = self.even_even_masking.repeat(batch_size, 1, 1, 1)
                odd_even_masking = self.odd_even_masking.repeat(batch_size, 1, 1, 1)
                even_odd_masking = self.even_odd_masking.repeat(batch_size, 1, 1, 1)

                if self.switching:
                    if self.channel_consistent:
                        inputs_mask_odd = inputs.masked_fill(odd_odd_masking.bool(), self.mask_value)
                        inputs_mask_even = inputs.masked_fill(even_even_masking.bool(), self.mask_value)
                    else:
                        inputs_mask_odd_even = inputs.masked_fill(odd_even_masking.bool(), self.mask_value)
                        inputs_mask_even_odd = inputs.masked_fill(even_odd_masking.bool(), self.mask_value)

                else:
                    if self.channel_consistent:
                        inputs_mask_odd = inputs.masked_fill(odd_odd_masking.bool(), self.mask_value)
                    else:
                        inputs_mask_odd_even = inputs.masked_fill(odd_even_masking.bool(), self.mask_value)

            elif self.mode == "random":
                # `(batch_size, num_channels, num_patches, patch_length)`
                batch_size, num_channels, sequence_length, num_features = inputs.shape
                mask_ratio = self.mask_ratio
                len_keep = int(sequence_length * (1 - mask_ratio))

                mask = torch.ones(batch_size, num_channels, sequence_length, device=self.device)
                mask[:, :, :len_keep] = 0

                if self.channel_consistent:
                    odd_odd_noise = torch.rand(batch_size, 1, sequence_length,
                                               device=self.device)  # noise in [0, 1], bs x 1 x  L
                    odd_odd_noise = odd_odd_noise.repeat(1, num_channels, 1)  # bs x num_channels x time
                    # sort noise for each sample
                    ids_shuffle = torch.argsort(odd_odd_noise, dim=-1)  # ascend: small is keep, large is remove
                    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

                    odd_odd_masking = torch.gather(mask, dim=-1, index=ids_restore)
                    odd_odd_masking = odd_odd_masking.unsqueeze(-1).repeat(1, 1, 1,
                                                                           num_features)  # mask: [bs x num_channels x num_patches x patch_length]
                    even_even_masking = (odd_odd_masking == 0).long()

                    inputs_mask_odd = inputs.masked_fill(odd_odd_masking.bool(), self.mask_value)
                    inputs_mask_even = inputs.masked_fill(even_even_masking.bool(), self.mask_value)

                else:
                    # noise in [0, 1], bs x num_channels x L
                    odd_even_noise = torch.rand(batch_size, num_channels, sequence_length, device=self.device)
                    ids_shuffle = torch.argsort(odd_even_noise, dim=-1)  # ascend: small is keep, large is remove
                    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]
                    odd_even_masking = torch.gather(mask, dim=-1, index=ids_restore)
                    odd_even_masking = odd_even_masking.unsqueeze(-1).repeat(1, 1, 1,
                                                                             num_features)  # mask: [bs x num_channels x num_patches x patch_length]
                    even_odd_masking = (odd_even_masking == 0).long()
                    inputs_mask_odd_even = inputs.masked_fill(odd_even_masking.bool(), self.mask_value)
                    inputs_mask_even_odd = inputs.masked_fill(even_odd_masking.bool(), self.mask_value)
            else:
                return NotImplementedError
        else:
            odd_even_masking = self.mask
            inputs_mask_odd_even = inputs.masked_fill(odd_even_masking, self.mask_value)

        return MaskOutput(inputs_mask_odd=inputs_mask_odd,
                          inputs_mask_even=inputs_mask_even,
                          inputs_mask_odd_even=inputs_mask_odd_even,
                          inputs_mask_even_odd=inputs_mask_even_odd,
                          mask_odd=odd_odd_masking,
                          mask_even=even_even_masking,
                          mask_even_odd=even_odd_masking,
                          mask_odd_even=odd_even_masking)


class ReconHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: PatchDetectorConfig):
        """
        @param in_features: d_model
        @param out_features: patch_size
        """
        super().__init__()

        num_hidden = in_features
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_features, num_hidden),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(num_hidden, out_features),
        )

    def forward(self, inputs: Tensor):
        inputs = self.head(inputs)

        return inputs


class PositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self,
                 config: PatchDetectorConfig,
                 num_patches: int):
        super().__init__()

        self.num_patches = num_patches
        # positional encoding: [num_patches x d_model]
        if config.use_position_encoder:
            self.position_enc = self._init_pe(config=config)
        else:
            self.position_enc = nn.Parameter(torch.zeros(num_patches, config.d_model))

    def _init_pe(self, config: PatchDetectorConfig) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(self.num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(self.num_patches, config.d_model)
            position = torch.arange(0, self.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' "
                f"and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state
