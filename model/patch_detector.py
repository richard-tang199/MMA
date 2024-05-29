from typing import Any, Tuple

import torch.nn
from torch import Tensor

from model.utility import *
from model.PatchTSMixerLayer import *
from model.NF import MAF
from model.PatchTST_Layer import *
from model.PatchContrastLayer import *
from config.patchDetectorConfig import *


@dataclass
class PatchDetectorOutput:
    """
    @param over_output: (batch_size,sequence_length, num_channels)
    """
    loss: torch.FloatTensor
    even_output: Optional[torch.FloatTensor] = None
    odd_output: Optional[torch.FloatTensor] = None
    over_output: Optional[torch.FloatTensor] = None


@dataclass
class PatchDetectorAttentionOutput:
    """
    @param over_output: (batch_size,sequence_length, num_channels)
    """
    loss: torch.FloatTensor
    contrast_loss: Optional[torch.FloatTensor] = None
    recon_loss: Optional[torch.FloatTensor] = None
    even_output: Optional[torch.FloatTensor] = None
    odd_output: Optional[torch.FloatTensor] = None
    over_output: Optional[torch.FloatTensor] = None
    Attention_output: Optional[torch.FloatTensor] = None
    sim_score: Optional[torch.FloatTensor] = None


@dataclass
class PatchDetectorContrastOutput:
    loss: torch.FloatTensor
    recon_loss: Optional[torch.FloatTensor] = None
    contrast_loss: Optional[torch.FloatTensor] = None
    even_output: Optional[torch.FloatTensor] = None
    odd_output: Optional[torch.FloatTensor] = None
    over_output: Optional[torch.FloatTensor] = None
    sim_score: Optional[torch.FloatTensor] = None


class PatchDetector(nn.Module):
    def __init__(self, config: PatchDetectorConfig):
        """
        @type config: PatchDetectorConfig
        """
        super().__init__()
        self.encoder = PatchMixerEncoder(config)
        if config.instance_normalization:
            self.norm = RevIN(num_features=config.num_channels, affine=False)
        else:
            self.norm = None

        self.switching = config.mask_switching
        self.channel_consistent = config.channel_consistent
        self.patcher = Patchify(sequence_length=config.window_length,
                                patch_length=config.patch_length,
                                patch_stride=config.stride)

        self.masker = MaskingStrategy(mask_ratio=config.mask_ratio,
                                      num_channels=config.num_channels,
                                      num_patches=config.num_patches,
                                      patch_length=config.patch_length,
                                      device=config.device,
                                      mode=config.masking_mode,
                                      channel_consistent=config.channel_consistent,
                                      switching=config.mask_switching,
                                      mask_value=config.mask_value)
        self.encoder = PatchMixerEncoder(config)
        # self.head = ReconHead(in_features=config.d_model, out_features=config.patch_length, config=config)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.patch_length, bias=True),
        )
        self.loss = torch.nn.MSELoss(reduction="none")
        self.flatten = nn.Flatten(-2)
        self.remove_anomaly = config.remove_anomaly

    def forward(self,
                inputs: Tensor,
                epoch: int = 0,
                mode: str = "train",
                window_threshold: float = 1.02,
                patch_threshold: float = 1.01,
                ) -> PatchDetectorContrastOutput:
        """
        @param inputs: (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
        Input for Patchify
        @param epoch:
        @param mode:
        @param window_threshold:
        @param patch_threshold:
        """
        if self.norm is not None:
            inputs = self.norm(inputs, mode="norm")

        # patches: (batch_size, num_channels, num_patches, patch_length)
        patches = self.patcher(inputs)

        # masking
        mask_output: MaskOutput = self.masker(patches)
        odd_mask_patches = mask_output.inputs_mask_odd_even
        even_mask_patches = mask_output.inputs_mask_even_odd
        odd_mask = mask_output.mask_odd_even
        even_mask = mask_output.mask_even_odd

        encoder_output_odd: PatchTSMixerEncoderOutput = self.encoder(odd_mask_patches)
        encoder_output_even: PatchTSMixerEncoderOutput = self.encoder(even_mask_patches)

        odd_encoder_hidden = encoder_output_odd.last_hidden_state
        even_encoder_hidden = encoder_output_even.last_hidden_state

        # obtain unmask origin value
        odd_head = self.head(odd_encoder_hidden)
        even_head = self.head(even_encoder_hidden)

        if self.norm is not None:
            odd_head = self.norm(odd_head, "denorm")
            even_head = self.norm(even_head, "denorm")

        over_head = odd_head * odd_mask + even_head * even_mask
        odd_loss = self.loss(patches, odd_head)
        odd_loss = (odd_loss.mean(dim=-1) * odd_mask[:, :, :, 0]).sum() / (
                odd_mask[:, :, :, 0].sum() + 1e-10)
        even_loss = self.loss(patches, even_head)
        even_loss = (even_loss.mean(dim=-1) * even_mask[:, :, :, 0]).sum() / (
                even_mask[:, :, :, 0].sum() + 1e-10)
        total_loss = (odd_loss + even_loss) / 2

        # odd_output, even_output, over_output: (batch_size,sequence_length,num_channels)
        # TODO: substitute
        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)

        if mode == "test":
            if self.remove_anomaly:
                over_output = self.rejudge(
                    raw_inputs=inputs,
                    recon_inputs=over_output,
                    window_threshold=window_threshold,
                    patch_threshold=patch_threshold
                )

        return PatchDetectorContrastOutput(loss=total_loss,
                                           odd_output=odd_output,
                                           even_output=even_output,
                                           over_output=over_output)

    def rejudge(self,
                raw_inputs: Tensor,
                recon_inputs: Tensor,
                window_threshold: float,
                patch_threshold: float) -> Tensor:
        """
        @param raw_inputs: (batch_size, window_length, num_channels)
        @param recon_inputs: (batch_size, window_length, num_channels)
        @param window_threshold:
        @param patch_threshold:
        @return:
        """
        # calculate window reconstruction diff
        window_diff = (raw_inputs - recon_inputs).abs().mean(dim=-1).mean(dim=-1)  # (batch_size, )

        if (window_diff > window_threshold).sum() == 0:
            return recon_inputs

        # window triggered (window_triggered_num, window_length, num_channels)
        window_triggred_index = torch.nonzero(window_diff > window_threshold, as_tuple=False).squeeze(-1)
        window_triggered = raw_inputs[window_triggred_index, ...]
        recon_triggered = recon_inputs[window_triggred_index, ...]

        # calculate patch reconstruction diff
        # patches: (window_triggered_num, num_channels, num_patches, patch_length)
        raw_patches = self.patcher(window_triggered)
        recon_patches = self.patcher(recon_triggered)
        num_window_triggered, num_channels, num_patches, patch_length = raw_patches.shape
        patch_diff = (raw_patches - recon_patches).abs().mean(dim=1).mean(dim=-1)  # (window_triggered_num, num_patches)

        # patch triggered (window_triggered_num, num_triggered_patches)
        patch_triggered_index = patch_diff > patch_threshold
        patch_triggered_num = torch.sum(patch_triggered_index, dim=-1)
        # (window_triggered_num, num_triggered_patches)
        patch_triggered_index = [torch.nonzero(i, as_tuple=False).squeeze() for i in patch_triggered_index]
        # (window_triggered_num)
        patch_triggered_median = [torch.median(i) if i.numel() > 0 else 0 for i in patch_triggered_index]
        patch_triggered_median = torch.tensor(patch_triggered_median, device=raw_inputs.device)

        patch_triggered_highest = patch_diff.argmax(dim=-1)  # (window_triggered_num)
        patch_triggered_median = torch.where(patch_triggered_median == 0, patch_triggered_highest,
                                             patch_triggered_median)

        patch_triggered_center = patch_triggered_median  # (window_triggered_num)
        patch_triggered_center = torch.round(patch_triggered_center).long()  # (window_triggered_num)

        # num_triggered windows
        patch_triggered_num[patch_triggered_num > 0.4 * num_patches] = int(0.4 * num_patches)
        patch_triggered_start = patch_triggered_center - patch_triggered_num // 2
        patch_triggered_end = patch_triggered_start + patch_triggered_num
        patch_triggered_start = torch.clamp(patch_triggered_start, 0, num_patches - 1)
        patch_triggered_end = torch.clamp(patch_triggered_end, 0, num_patches - 1)

        mask = torch.zeros([num_window_triggered, num_patches], device=raw_inputs.device)
        for i in range(num_window_triggered):
            mask[i, patch_triggered_start[i]:patch_triggered_end[i]] = 1

        mask = mask.unsqueeze(1).repeat(1, num_channels, 1)  # (batch_size, num_channels, num_patches)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1,
                                         patch_length).bool()  # (batch_size, num_channels, num_patches, patch_length)

        masker = MaskingStrategy(
            num_channels=num_channels,
            num_patches=num_patches,
            patch_length=patch_length,
            mask=mask
        )

        # recalculate
        recalculate_mask_output: MaskOutput = masker(raw_patches)
        recalculate_mask_patches = recalculate_mask_output.inputs_mask_odd_even
        recalculate_mask = recalculate_mask_output.mask_odd_even
        encoder_mask_hidden = self.encoder(recalculate_mask_patches).last_hidden_state
        recalculate_head = self.head(encoder_mask_hidden)
        recalculate_output = recalculate_head * recalculate_mask
        recalculate_output = self.flatten(recalculate_output).transpose(-1, -2)

        recalculate_mask = self.flatten(recalculate_mask).transpose(-1, -2)
        recalculate_output = window_triggered * (1 - recalculate_mask.long()) + recalculate_output

        recalculate_output = self.forward(inputs=recalculate_output, mode="valid").over_output
        recon_inputs[window_triggred_index, ...] = recalculate_output

        return recon_inputs


class PatchDetectorAttention(nn.Module):
    def __init__(self, config: PatchDetectorAttentionConfig):
        """
        @type config: PatchDetectorConfig
        """
        super().__init__()
        self.encoder = PatchMixerEncoder(config)
        if config.instance_normalization:
            self.norm = RevIN(num_features=config.num_channels, affine=False)
        else:
            self.norm = None

        self.switching = config.mask_switching
        self.channel_consistent = config.channel_consistent
        self.patcher = Patchify(sequence_length=config.window_length,
                                patch_length=config.patch_length,
                                patch_stride=config.stride)

        self.masker = MaskingStrategy(mask_ratio=config.mask_ratio,
                                      num_channels=config.num_channels,
                                      num_patches=config.num_patches,
                                      patch_length=config.patch_length,
                                      device=config.device,
                                      mode=config.masking_mode,
                                      channel_consistent=config.channel_consistent,
                                      switching=config.mask_switching,
                                      mask_value=config.mask_value)

        self.encoder = PatchTSTEncoder(config)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.patch_length, bias=True),
        )
        self.loss = torch.nn.MSELoss(reduction="none")
        self.flatten = nn.Flatten(-2)

    def forward(self, inputs: Tensor,
                epoch: int = 0,
                mode: str = "train",
                window_threshold: float = 1.02,
                patch_threshold: float = 1.01,
                ) -> PatchDetectorAttentionOutput:
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for Patchify

        Returns: PatchDetectorOutput
        """

        if self.norm is not None:
            inputs = self.norm(inputs, mode="norm")

        # patches: (batch_size, num_channels, num_patches, patch_length)
        patches = self.patcher(inputs)

        # masking: (batch_size, num_channels, num_patches, patch_length)
        mask_output: MaskOutput = self.masker(patches)
        odd_mask_patches = mask_output.inputs_mask_odd_even
        even_mask_patches = mask_output.inputs_mask_even_odd
        odd_mask = mask_output.mask_odd_even
        even_mask = mask_output.mask_even_odd

        encoder_output_odd: PatchAttentionOutput = self.encoder(odd_mask_patches)
        encoder_output_even: PatchAttentionOutput = self.encoder(even_mask_patches)

        odd_encoder_hidden = encoder_output_odd.last_hidden_state
        even_encoder_hidden = encoder_output_even.last_hidden_state

        odd_head = self.head(odd_encoder_hidden)
        even_head = self.head(even_encoder_hidden)

        if self.norm is not None:
            odd_head = self.norm(odd_head, "denorm")
            even_head = self.norm(even_head, "denorm")

        over_head = odd_head * odd_mask + even_head * even_mask
        odd_loss = self.loss(patches, odd_head)
        odd_loss = (odd_loss.mean(dim=-1) * odd_mask[:, :, :, 0]).sum() / (
                odd_mask[:, :, :, 0].sum() + 1e-10)
        even_loss = self.loss(patches, even_head)
        even_loss = (even_loss.mean(dim=-1) * even_mask[:, :, :, 0]).sum() / (
                even_mask[:, :, :, 0].sum() + 1e-10)
        total_loss = odd_loss + even_loss

        # odd_output, even_output: (batch_size, num_channels, sequence_length)
        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)

        return PatchDetectorAttentionOutput(loss=total_loss,
                                            odd_output=odd_output,
                                            even_output=even_output,
                                            over_output=over_output,
                                            Attention_output=encoder_output_odd.attention_weights)

    def rejudge(self,
                raw_inputs: Tensor,
                recon_inputs: Tensor,
                window_threshold: float,
                patch_threshold: float) -> Tensor:
        """
        @param raw_inputs: (batch_size, window_length, num_channels)
        @param recon_inputs: (batch_size, window_length, num_channels)
        @param window_threshold:
        @param patch_threshold:
        @return:
        """
        # calculate window reconstruction diff
        window_diff = (raw_inputs - recon_inputs).abs().mean(dim=-1).mean(dim=-1)  # (batch_size, )

        if (window_diff > window_threshold).sum() == 0:
            return recon_inputs

        # window triggered (window_triggered_num, window_length, num_channels)
        window_triggred_index = torch.nonzero(window_diff > window_threshold, as_tuple=False).squeeze(-1)
        window_triggered = raw_inputs[window_triggred_index, ...]
        recon_triggered = recon_inputs[window_triggred_index, ...]

        # calculate patch reconstruction diff
        # patches: (window_triggered_num, num_channels, num_patches, patch_length)
        raw_patches = self.patcher(window_triggered)
        recon_patches = self.patcher(recon_triggered)
        num_window_triggered, num_channels, num_patches, patch_length = raw_patches.shape
        patch_diff = (raw_patches - recon_patches).abs().mean(dim=1).mean(dim=-1)  # (window_triggered_num, num_patches)

        # patch triggered (window_triggered_num, num_triggered_patches)
        patch_triggered_index = patch_diff > patch_threshold
        patch_triggered_num = torch.sum(patch_triggered_index, dim=-1)
        # (window_triggered_num, num_triggered_patches)
        patch_triggered_index = [torch.nonzero(i, as_tuple=False).squeeze() for i in patch_triggered_index]
        # (window_triggered_num)
        patch_triggered_median = [torch.median(i) if i.numel() > 0 else 0 for i in patch_triggered_index]
        patch_triggered_median = torch.tensor(patch_triggered_median, device=raw_inputs.device)

        patch_triggered_highest = patch_diff.argmax(dim=-1)  # (window_triggered_num)
        patch_triggered_median = torch.where(patch_triggered_median == 0, patch_triggered_highest,
                                             patch_triggered_median)

        patch_triggered_center = patch_triggered_median  # (window_triggered_num)
        patch_triggered_center = torch.round(patch_triggered_center).long()  # (window_triggered_num)

        # num_triggered windows
        patch_triggered_num[patch_triggered_num > 0.4 * num_patches] = int(0.4 * num_patches)
        patch_triggered_start = patch_triggered_center - patch_triggered_num // 2
        patch_triggered_end = patch_triggered_start + patch_triggered_num
        patch_triggered_start = torch.clamp(patch_triggered_start, 0, num_patches - 1)
        patch_triggered_end = torch.clamp(patch_triggered_end, 0, num_patches - 1)

        mask = torch.zeros([num_window_triggered, num_patches], device=raw_inputs.device)
        for i in range(num_window_triggered):
            mask[i, patch_triggered_start[i]:patch_triggered_end[i]] = 1

        mask = mask.unsqueeze(1).repeat(1, num_channels, 1)  # (batch_size, num_channels, num_patches)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1,
                                         patch_length).bool()  # (batch_size, num_channels, num_patches, patch_length)

        masker = MaskingStrategy(
            num_channels=num_channels,
            num_patches=num_patches,
            patch_length=patch_length,
            mask=mask
        )

        # recalculate
        recalculate_mask_output: MaskOutput = masker(raw_patches)
        recalculate_mask_patches = recalculate_mask_output.inputs_mask_odd_even
        recalculate_mask = recalculate_mask_output.mask_odd_even
        encoder_mask_hidden = self.encoder(recalculate_mask_patches).last_hidden_state
        recalculate_head = self.head(encoder_mask_hidden)
        recalculate_output = recalculate_head * recalculate_mask
        recalculate_output = self.flatten(recalculate_output).transpose(-1, -2)

        recalculate_mask = self.flatten(recalculate_mask).transpose(-1, -2)
        recalculate_output = window_triggered * (1 - recalculate_mask.long()) + recalculate_output

        recalculate_output = self.forward(inputs=recalculate_output, mode="valid").over_output
        recon_inputs[window_triggred_index, ...] = recalculate_output

        return recon_inputs


class PatchContrastDetector(nn.Module):
    def __init__(self, config: PatchDetectorConfig):
        """
        @type config: PatchDetectorConfig
        """
        super().__init__()
        self.encoder = PatchMixerEncoder(config)
        if config.instance_normalization:
            self.norm = RevIN(num_features=config.num_channels, affine=False)
        else:
            self.norm = None

        self.patcher = Patchify(sequence_length=config.window_length,
                                patch_length=config.patch_length,
                                patch_stride=config.stride)

        self.masker = MaskingStrategy(mask_ratio=config.mask_ratio,
                                      num_channels=config.num_channels,
                                      num_patches=config.num_patches,
                                      patch_length=config.patch_length,
                                      device=config.device,
                                      mode=config.masking_mode,
                                      channel_consistent=config.channel_consistent,
                                      switching=config.mask_switching,
                                      mask_value=config.mask_value)

        self.encoder = PatchMixerContrastEncoder(config)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.patch_length, bias=True),
        )
        self.loss = torch.nn.MSELoss(reduction="none")
        self.flatten = nn.Flatten(-2)
        self.remove_anomaly = config.remove_anomaly

    def forward(self,
                inputs: Tensor,
                epoch: int = 0,
                mode: str = "train",
                window_threshold: float = 1.02,
                patch_threshold: float = 1.01,
                ratio: float = 0.005
                ) -> PatchDetectorOutput:

        # patches: (batch_size, num_channels, num_patches, patch_length)
        patches = self.patcher(inputs)
        batch_size, num_channels, num_patches, patch_length = patches.shape

        # initial embedding [bs x n_vars x num_patch x d_model]
        initial_patch_embeddings = self.encoder.patch_embedding(patches)
        # masking
        mask_output: MaskOutput = self.masker(patches)
        odd_mask_patches = mask_output.inputs_mask_odd_even
        even_mask_patches = mask_output.inputs_mask_even_odd
        odd_mask = mask_output.mask_odd_even
        even_mask = mask_output.mask_even_odd
        odd_mask_indices = torch.nonzero(odd_mask[..., 0])
        even_mask_indices = torch.nonzero(even_mask[..., 0])

        # obtain embedding
        encoder_output_odd: PatchTSMixerEncoderOutput = self.encoder(odd_mask_patches)
        encoder_output_even: PatchTSMixerEncoderOutput = self.encoder(even_mask_patches)
        odd_encoder_hidden = encoder_output_odd.last_hidden_state
        even_encoder_hidden = encoder_output_even.last_hidden_state

        # obtain reconstruct loss
        odd_head = self.head(odd_encoder_hidden)
        even_head = self.head(even_encoder_hidden)

        over_head = odd_head * odd_mask + even_head * even_mask
        recon_loss = self.loss(patches, over_head).mean()

        # contrastive loss
        over_head_embedding = self.encoder.patch_embedding(over_head)
        # over_similarity: batch_size x num_channels x num_patch x num_patch
        over_similarity = torch.matmul(over_head_embedding, initial_patch_embeddings.transpose(-1, -2))
        # over_similarity: batch_size x num_channels x num_patch
        over_similarity = torch.diagonal(over_similarity, dim1=-2, dim2=-1)
        # over_similarity: batch_size x num_channels x num_patch x patch_length
        over_similarity = over_similarity.unsqueeze(-1).repeat(1, 1, 1, patch_length)

        # obtain contrastive loss
        contrast_loss = contrastive_loss(mask_embedding=over_head_embedding,
                                         origin_embedding=initial_patch_embeddings,
                                         augmented_embedding=None,
                                         batch_size=batch_size,
                                         num_channels=num_channels)

        # odd_output, even_output, over_output: (batch_size,sequence_length,num_channels)
        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)
        over_similarity = self.flatten(over_similarity).transpose(-1, -2)
        total_loss = recon_loss + ratio * contrast_loss

        if mode == "test":
            if self.remove_anomaly:
                over_output, over_similarity = self.rejudge(raw_inputs=inputs,
                                                            recon_inputs=over_output,
                                                            raw_similatiy=over_similarity,
                                                            window_threshold=window_threshold,
                                                            patch_threshold=patch_threshold)

        # final output
        final_output = PatchDetectorContrastOutput(
            loss=total_loss,
            recon_loss=recon_loss,
            contrast_loss=contrast_loss,
            over_output=over_output,
            sim_score=1 - over_similarity
        )

        return final_output

    def rejudge(self,
                raw_inputs: Tensor,
                recon_inputs: Tensor,
                raw_similatiy: Tensor,
                window_threshold: float,
                patch_threshold: float) -> tuple[Tensor, Tensor]:
        """
        @param raw_inputs: (batch_size, window_length, num_channels)
        @param recon_inputs: (batch_size, window_length, num_channels)
        @param window_threshold:
        @param patch_threshold:
        @return:
        """
        # calculate window reconstruction diff
        window_diff = (raw_inputs - recon_inputs).abs().mean(dim=-1).mean(dim=-1)  # (batch_size, )

        if (window_diff > window_threshold).sum() == 0:
            return recon_inputs, raw_similatiy

        # window triggered (window_triggered_num, window_length, num_channels)
        window_triggred_index = torch.nonzero(window_diff > window_threshold, as_tuple=False).squeeze(-1)
        window_triggered = raw_inputs[window_triggred_index, ...]
        recon_triggered = recon_inputs[window_triggred_index, ...]

        # calculate patch reconstruction diff
        # patches: (window_triggered_num, num_channels, num_patches, patch_length)
        raw_patches = self.patcher(window_triggered)
        recon_patches = self.patcher(recon_triggered)
        num_window_triggered, num_channels, num_patches, patch_length = raw_patches.shape
        patch_diff = (raw_patches - recon_patches).abs().mean(dim=1).mean(dim=-1)  # (window_triggered_num, num_patches)

        # patch triggered (window_triggered_num, num_triggered_patches)
        patch_triggered_index = patch_diff > patch_threshold
        patch_triggered_num = torch.sum(patch_triggered_index, dim=-1)
        # (window_triggered_num, num_triggered_patches)
        patch_triggered_index = [torch.nonzero(i, as_tuple=False).squeeze() for i in patch_triggered_index]
        # (window_triggered_num)
        patch_triggered_median = [torch.median(i) if i.numel() > 0 else 0 for i in patch_triggered_index]
        patch_triggered_median = torch.tensor(patch_triggered_median, device=raw_inputs.device)

        patch_triggered_highest = patch_diff.argmax(dim=-1)  # (window_triggered_num)
        patch_triggered_median = torch.where(patch_triggered_median == 0, patch_triggered_highest,
                                             patch_triggered_median)

        patch_triggered_center = patch_triggered_median  # (window_triggered_num)
        patch_triggered_center = torch.round(patch_triggered_center).long()  # (window_triggered_num)

        # num_triggered windows
        patch_triggered_num[patch_triggered_num > 0.4 * num_patches] = int(0.4 * num_patches)
        patch_triggered_start = patch_triggered_center - patch_triggered_num // 2
        patch_triggered_end = patch_triggered_start + patch_triggered_num
        patch_triggered_start = torch.clamp(patch_triggered_start, 0, num_patches - 1)
        patch_triggered_end = torch.clamp(patch_triggered_end, 0, num_patches - 1)

        mask = torch.zeros([num_window_triggered, num_patches], device=raw_inputs.device)
        for i in range(num_window_triggered):
            mask[i, patch_triggered_start[i]:patch_triggered_end[i]] = 1

        mask = mask.unsqueeze(1).repeat(1, num_channels, 1)  # (batch_size, num_channels, num_patches)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1,
                                         patch_length).bool()  # (batch_size, num_channels, num_patches, patch_length)

        masker = MaskingStrategy(
            num_channels=num_channels,
            num_patches=num_patches,
            patch_length=patch_length,
            mask=mask
        )

        # recalculate
        recalculate_mask_output: MaskOutput = masker(raw_patches)
        recalculate_mask_patches = recalculate_mask_output.inputs_mask_odd_even
        recalculate_mask = recalculate_mask_output.mask_odd_even
        encoder_mask_hidden = self.encoder(recalculate_mask_patches).last_hidden_state
        recalculate_head = self.head(encoder_mask_hidden)
        recalculate_output = recalculate_head * recalculate_mask
        recalculate_output = self.flatten(recalculate_output).transpose(-1, -2)

        recalculate_mask = self.flatten(recalculate_mask).transpose(-1, -2)
        recalculate_output = window_triggered * (1 - recalculate_mask.long()) + recalculate_output

        recalculate_output = self.forward(inputs=recalculate_output, mode="valid")
        recalculate_recon = recalculate_output.over_output

        recon_inputs[window_triggred_index, ...] = recalculate_recon

        # recalculate similarity
        new_patches = self.patcher(recalculate_recon)
        # obtain embedding
        raw_patch_embedding = self.encoder.patch_embedding(raw_patches)
        new_patch_embedding = self.encoder.patch_embedding(new_patches)
        # over_similarity: batch_size x num_channels x num_patch x num_patch
        similarity = torch.matmul(new_patch_embedding, raw_patch_embedding.transpose(-1, -2))
        # over_similarity: batch_size x num_channels x num_patch
        similarity = torch.diagonal(similarity, dim1=-2, dim2=-1)
        # over_similarity: batch_size x num_channels x num_patch x patch_length
        similarity = similarity.unsqueeze(-1).repeat(1, 1, 1, patch_length)
        similarity = self.flatten(similarity).transpose(-1, -2)
        raw_similatiy[window_triggred_index, ...] = similarity
        return recon_inputs, raw_similatiy


class PatchDetectorGru(nn.Module):
    def __init__(self, config: PatchDetectorConfig):
        """
        @type config: PatchDetectorConfig
        """
        super().__init__()
        self.encoder = nn.GRU(input_size=config.d_model,
                              hidden_size=config.expansion_factor * config.d_model,
                              num_layers=config.num_layers,
                              batch_first=True,
                              dropout=config.dropout,
                              bidirectional=True)
        self.patcher = Patchify(sequence_length=config.window_length,
                                patch_length=config.patch_length,
                                patch_stride=config.stride)
        self.masker = MaskingStrategy(mask_ratio=config.mask_ratio,
                                      num_channels=config.num_channels,
                                      num_patches=config.num_patches,
                                      patch_length=config.patch_length,
                                      device=config.device,
                                      mode=config.masking_mode,
                                      channel_consistent=config.channel_consistent,
                                      switching=config.mask_switching,
                                      mask_value=config.mask_value)
        self.embedding = PatchEmbedding(config)
        self.projection = nn.Linear(config.d_model * config.expansion_factor * 2, config.d_model)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.patch_length, bias=True),
        )
        self.loss = torch.nn.MSELoss(reduction="none")
        self.flatten = nn.Flatten(-2)
        self.remove_anomaly = config.remove_anomaly

    def forward(self,
                inputs: Tensor,
                mode: str = "train",
                epoch: int = 0,
                window_threshold: float = 1.02,
                patch_threshold: float = 1.01,
                ratio: float = 0.005
                ) -> PatchDetectorContrastOutput:
        # patches: (batch_size, num_channels, num_patches, patch_length)
        patches = self.patcher(inputs)
        batch_size, num_channels, num_patches, patch_length = patches.shape
        # masking
        mask_output: MaskOutput = self.masker(patches)
        odd_mask_patches = mask_output.inputs_mask_odd_even
        odd_mask_patches = odd_mask_patches.reshape(batch_size * num_channels, num_patches, patch_length)
        even_mask_patches = mask_output.inputs_mask_even_odd
        even_mask_patches = even_mask_patches.reshape(batch_size * num_channels, num_patches, patch_length)
        odd_mask = mask_output.mask_odd_even
        even_mask = mask_output.mask_even_odd

        # obtain embedding
        encoder_output_odd, _ = self.encoder(self.embedding(odd_mask_patches))
        encoder_output_even, _ = self.encoder(self.embedding(even_mask_patches))
        odd_encoder_hidden = self.projection(encoder_output_odd)
        even_encoder_hidden = self.projection(encoder_output_even)

        odd_encoder_hidden = odd_encoder_hidden / torch.sqrt(
            torch.sum(odd_encoder_hidden ** 2, dim=-1, keepdim=True) + 1e-6)
        even_encoder_hidden = even_encoder_hidden / torch.sqrt(
            torch.sum(even_encoder_hidden ** 2, dim=-1, keepdim=True) + 1e-6)

        # obtain reconstruct loss
        odd_head = self.head(odd_encoder_hidden)
        odd_head = odd_head.reshape(batch_size, num_channels, num_patches, -1)
        even_head = self.head(even_encoder_hidden)
        even_head = even_head.reshape(batch_size, num_channels, num_patches, -1)
        over_head = odd_head * odd_mask + even_head * even_mask
        recon_loss = self.loss(patches, over_head).mean()

        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)

        if mode == "test":
            if self.remove_anomaly:
                over_output = self.rejudge(raw_inputs=inputs,
                                           recon_inputs=over_output,
                                           window_threshold=window_threshold,
                                           patch_threshold=patch_threshold)

        return PatchDetectorContrastOutput(
            loss=recon_loss,
            odd_output=odd_output,
            even_output=even_output,
            over_output=over_output,
        )

    def rejudge(self,
                raw_inputs: Tensor,
                recon_inputs: Tensor,
                window_threshold: float,
                patch_threshold: float) -> Tensor:

        # calculate window reconstruction diff
        window_diff = (raw_inputs - recon_inputs).abs().mean(dim=-1).mean(dim=-1)  # (batch_size, )

        if (window_diff > window_threshold).sum() == 0:
            return recon_inputs

        # window triggered (window_triggered_num, window_length, num_channels)
        window_triggred_index = torch.nonzero(window_diff > window_threshold, as_tuple=False).squeeze(-1)
        window_triggered = raw_inputs[window_triggred_index, ...]
        recon_triggered = recon_inputs[window_triggred_index, ...]

        # calculate patch reconstruction diff
        # patches: (window_triggered_num, num_channels, num_patches, patch_length)
        raw_patches = self.patcher(window_triggered)
        recon_patches = self.patcher(recon_triggered)
        num_window_triggered, num_channels, num_patches, patch_length = raw_patches.shape
        patch_diff = (raw_patches - recon_patches).abs().mean(dim=1).mean(dim=-1)  # (window_triggered_num, num_patches)

        # patch triggered (window_triggered_num, num_triggered_patches)
        patch_triggered_index = patch_diff > patch_threshold
        patch_triggered_num = torch.sum(patch_triggered_index, dim=-1)
        # (window_triggered_num, num_triggered_patches)
        patch_triggered_index = [torch.nonzero(i, as_tuple=False).squeeze() for i in patch_triggered_index]
        # (window_triggered_num)
        patch_triggered_median = [torch.median(i) if i.numel() > 0 else 0 for i in patch_triggered_index]
        patch_triggered_median = torch.tensor(patch_triggered_median, device=raw_inputs.device)

        patch_triggered_highest = patch_diff.argmax(dim=-1)  # (window_triggered_num)
        patch_triggered_median = torch.where(patch_triggered_median == 0, patch_triggered_highest,
                                             patch_triggered_median)

        patch_triggered_center = patch_triggered_median  # (window_triggered_num)
        patch_triggered_center = torch.round(patch_triggered_center).long()  # (window_triggered_num)

        # num_triggered windows
        patch_triggered_num[patch_triggered_num > 0.4 * num_patches] = int(0.4 * num_patches)
        patch_triggered_start = patch_triggered_center - patch_triggered_num // 2
        patch_triggered_end = patch_triggered_start + patch_triggered_num
        patch_triggered_start = torch.clamp(patch_triggered_start, 0, num_patches - 1)
        patch_triggered_end = torch.clamp(patch_triggered_end, 0, num_patches - 1)

        mask = torch.zeros([num_window_triggered, num_patches], device=raw_inputs.device)
        for i in range(num_window_triggered):
            mask[i, patch_triggered_start[i]:patch_triggered_end[i]] = 1

        mask = mask.unsqueeze(1).repeat(1, num_channels, 1)  # (batch_size, num_channels, num_patches)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1,
                                         patch_length).bool()  # (batch_size, num_channels, num_patches, patch_length)

        masker = MaskingStrategy(
            num_channels=num_channels,
            num_patches=num_patches,
            patch_length=patch_length,
            mask=mask
        )

        # recalculate
        recalculate_mask_output: MaskOutput = masker(raw_patches)
        recalculate_mask_patches = recalculate_mask_output.inputs_mask_odd_even
        recalculate_mask = recalculate_mask_output.mask_odd_even

        recalculate_mask_patches = recalculate_mask_patches.reshape(num_window_triggered * num_channels, num_patches,
                                                                    patch_length)
        encoder_mask_hidden, _ = self.encoder(self.embedding(recalculate_mask_patches))
        encoder_mask_hidden = self.projection(encoder_mask_hidden)
        encoder_mask_hidden = encoder_mask_hidden / torch.sqrt(
            torch.sum(encoder_mask_hidden ** 2, dim=-1, keepdim=True) + 1e-6)
        recalculate_head = self.head(encoder_mask_hidden)
        recalculate_head = recalculate_head.reshape(num_window_triggered, num_channels, num_patches, -1)
        recalculate_output = recalculate_head * recalculate_mask
        recalculate_output = self.flatten(recalculate_output).transpose(-1, -2)

        recalculate_mask = self.flatten(recalculate_mask).transpose(-1, -2)
        recalculate_output = window_triggered * (1 - recalculate_mask.long()) + recalculate_output

        recalculate_output = self.forward(inputs=recalculate_output, mode="valid")
        recalculate_recon = recalculate_output.over_output

        recon_inputs[window_triggred_index, ...] = recalculate_recon

        return recon_inputs


if __name__ == "__main__":
    config = PatchDetectorGruConfig()
    model = PatchDetectorGru(config).to("cuda")
    # sample (batch_size, sequence_length, num_channels)
    for i in range(1000):
        sample = 2 * torch.rand(64, config.window_length, config.num_channels, device='cuda')
        model.eval()
        output = model(sample, mode="test")
        print('finish')
