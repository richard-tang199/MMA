class PatchDenoiseContrast(nn.Module):
    def __init__(self, config: PatchDetectorConfig):
        """
        @type config: PatchDetectorConfig
        """
        super().__init__()
        self.encoder = PatchMixerEncoder(config)
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
        # self.head = ReconHead(in_features=config.d_model, out_features=config.patch_length, config=config)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.patch_length, bias=True),
        )
        self.loss = torch.nn.MSELoss()
        self.flatten = nn.Flatten(-2)

    def forward(self, inputs: Tensor, epoch: int = 0) -> PatchDetectorContrastOutput:
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for Patchify

        Returns: PatchDetectorOutput
        """

        # patches: (batch_size, num_channels, num_patches, patch_length)
        patches = self.patcher(inputs)
        batch_size, num_channels, num_patches, patch_length = patches.shape

        # masking
        mask_output: MaskOutput = self.masker(patches)
        odd_mask_patches = mask_output.inputs_mask_odd_even
        even_mask_patches = mask_output.inputs_mask_even_odd
        odd_mask = mask_output.mask_odd_even
        even_mask = mask_output.mask_even_odd
        odd_mask_indices = torch.nonzero(odd_mask[..., 0])
        even_mask_indices = torch.nonzero(even_mask[..., 0])

        # obtain encoding
        encoder_output_odd: PatchTSMixerEncoderOutput = self.encoder(odd_mask_patches)
        encoder_output_even: PatchTSMixerEncoderOutput = self.encoder(even_mask_patches)
        odd_encoder_hidden = encoder_output_odd.last_hidden_state
        even_encoder_hidden = encoder_output_even.last_hidden_state

        # obtain reconstruct loss
        odd_head = self.head(odd_encoder_hidden)
        even_head = self.head(even_encoder_hidden)

        over_head = odd_head * odd_mask + even_head * even_mask
        odd_loss = self.loss(patches, odd_head)
        even_loss = self.loss(patches, even_head)
        recon_loss = (odd_loss + even_loss) / 2

        # odd_output, even_output, over_output: (batch_size,sequence_length,num_channels)
        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)

        # obtain contrastive loss
        odd_contrast_loss = contrastive_loss(mask_embedding=odd_encoder_hidden,
                                             origin_embedding=even_encoder_hidden,
                                             batch_size=batch_size,
                                             num_channels=num_channels,
                                             mask_pos=odd_mask_indices)
        even_contrast_loss = contrastive_loss(mask_embedding=even_encoder_hidden,
                                              origin_embedding=odd_encoder_hidden,
                                              batch_size=batch_size,
                                              num_channels=num_channels,
                                              mask_pos=even_mask_indices)
        contrast_loss = (odd_contrast_loss + even_contrast_loss) / 2

        # obtain mask embedding
        odd_embedding_mask = odd_mask[..., 0].unsqueeze(-1).repeat(1, 1, 1, odd_encoder_hidden.shape[-1])
        even_embedding_mask = even_mask[..., 0].unsqueeze(-1).repeat(1, 1, 1, even_encoder_hidden.shape[-1])
        over_mask_embedding = odd_encoder_hidden * odd_embedding_mask + even_encoder_hidden * even_embedding_mask
        over_unmask_embedding = odd_encoder_hidden * even_embedding_mask + even_encoder_hidden * odd_embedding_mask
        over_similarity = torch.matmul(over_mask_embedding, over_unmask_embedding.transpose(-1, -2))
        # over_similarity: batch_size x num_channels x num_patch
        over_similarity = torch.diagonal(over_similarity, dim1=-2, dim2=-1)
        # over_similarity: batch_size x num_channels x num_patch x patch_length
        over_similarity = over_similarity.unsqueeze(-1).repeat(1, 1, 1, patch_length)
        over_similarity = self.flatten(over_similarity).transpose(-1, -2)
        ratio = 0.1

        return PatchDetectorContrastOutput(
            loss=recon_loss + ratio * contrast_loss,
            recon_loss=recon_loss,
            contrast_loss=contrast_loss,
            over_output=over_output,
            sim_score=1 - over_similarity
        )

class PatchDenoise(nn.Module):
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

        self.encoder = PatchMixerEncoder(config)
        # self.head = ReconHead(in_features=config.d_model, out_features=config.patch_length, config=config)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.patch_length, bias=True),
        )
        self.loss = torch.nn.MSELoss(reduction="none")
        self.flatten = nn.Flatten(-2)

    def forward(self, inputs: Tensor) -> PatchDetectorOutput:
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
        over_head = odd_head * odd_mask + even_head * even_mask

        if self.norm is not None:
            odd_head = self.norm(odd_head, "denorm")
            even_head = self.norm(even_head, "denorm")

        # over_head = odd_head * odd_mask + even_head * even_mask
        odd_loss = self.loss(patches, odd_head).mean()
        even_loss = self.loss(patches, even_head).mean()
        total_loss = odd_loss + even_loss

        # odd_output, even_output, over_output: (batch_size,sequence_length,num_channels)
        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)

        return PatchDetectorOutput(loss=total_loss,
                                   odd_output=odd_output,
                                   even_output=even_output,
                                   over_output=over_output)


class PatchContrastDetector_2(nn.Module):
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
        self.transformation = Transformation(modes=('uniform', 'amplitude', "flip", "trend"))

    def forward(self, inputs: Tensor, epoch: int = 0) -> PatchDetectorOutput:

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
        odd_loss = self.loss(patches, odd_head)
        odd_loss = (odd_loss.mean(dim=-1) * odd_mask[:, :, :, 0]).sum() / (
                odd_mask[:, :, :, 0].sum() + 1e-10)
        even_loss = self.loss(patches, even_head)
        even_loss = (even_loss.mean(dim=-1) * even_mask[:, :, :, 0]).sum() / (
                even_mask[:, :, :, 0].sum() + 1e-10)
        # TODO: direct loss process
        recon_loss = (odd_loss + even_loss) / 2

        odd_head_embedding = self.encoder.patch_embedding(odd_head)
        even_head_embedding = self.encoder.patch_embedding(even_head)
        odd_embedding_mask = odd_mask[..., 0].unsqueeze(-1).repeat(1, 1, 1, odd_encoder_hidden.shape[-1])
        even_embedding_mask = even_mask[..., 0].unsqueeze(-1).repeat(1, 1, 1, even_encoder_hidden.shape[-1])
        over_head_embedding = odd_head_embedding * odd_embedding_mask + even_head_embedding * even_embedding_mask
        # over_similarity: batch_size x num_channels x num_patch x num_patch
        over_similarity = torch.matmul(over_head_embedding, initial_patch_embeddings.transpose(-1, -2))
        # over_similarity: batch_size x num_channels x num_patch
        over_similarity = torch.diagonal(over_similarity, dim1=-2, dim2=-1)
        # over_similarity: batch_size x num_channels x num_patch x patch_length
        over_similarity = over_similarity.unsqueeze(-1).repeat(1, 1, 1, patch_length)

        # obtain contrastive loss
        odd_contrast_loss = contrastive_loss(mask_embedding=odd_head_embedding,
                                             origin_embedding=initial_patch_embeddings,
                                             augmented_embedding=None,
                                             mask_pos=odd_mask_indices,
                                             batch_size=batch_size,
                                             num_channels=num_channels)
        even_contrast_loss = contrastive_loss(mask_embedding=even_head_embedding,
                                              origin_embedding=initial_patch_embeddings,
                                              augmented_embedding=None,
                                              mask_pos=even_mask_indices,
                                              batch_size=batch_size,
                                              num_channels=num_channels)
        contrast_loss = (odd_contrast_loss + even_contrast_loss) / 2

        # odd_output, even_output, over_output: (batch_size,sequence_length,num_channels)
        odd_output = self.flatten(odd_head).transpose(-1, -2)
        even_output = self.flatten(even_head).transpose(-1, -2)
        over_output = self.flatten(over_head).transpose(-1, -2)
        over_similarity = self.flatten(over_similarity).transpose(-1, -2)

        ratio = 0.01

        # final output
        final_output = PatchDetectorContrastOutput(
            loss=recon_loss + ratio * contrast_loss,
            recon_loss=recon_loss,
            contrast_loss=contrast_loss,
            over_output=over_output,
            sim_score=1 - over_similarity
        )
        # print(f"recon_loss: {recon_loss.item():.4f} \n "
        #       f"con_loss: {contrast_loss.item():.4f}")

        return final_output