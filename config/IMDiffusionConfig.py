from dataclasses import dataclass


@dataclass
class PatchDetectorConfig:
    num_epochs: int = 201
    learning_rate: float = 1e-3
    batch_size: int = 64
    patch_length: int = 16
    window_length: int = 512
    window_stride: int = 1
    d_model: int = 16
    num_channels: int = 9
    num_layers: int = 3
    device: str = "cuda"
    masking_mode: str = "random"  # "random" or "gating"
    forward_times: int = 1
    instance_normalization: bool = False
    mode: str = "mix_channel" # "mix_channel" or "common_channel"
    channel_consistent: bool = False
    mask_switching: bool = True
    mask_value: int = 0
    mask_ratio: float = 0.5
    use_position_encoder: bool = True
    positional_encoding_type: str = "sincos"
    gated_attn: bool = True
    norm_mlp: str = "LayerNorm"
    self_attn: bool = False
    self_attn_heads: int = 1
    expansion_factor: int = 6
    dropout: float = 0.2
    norm_eps: float = 1e-5
    weight: float = 0.01

    def __post_init__(self):
        self.stride = self.patch_length # patch_stride
        self.num_patches = (max(self.window_length, self.patch_length) - self.patch_length) // self.stride + 1
        if self.masking_mode == "gating":
            self.mask_ratio = 0.5
        # self.mode: str = "mix_channel" if not self.channel_consistent else "common_channel"
        if self.masking_mode == "random":
            self.forward_times = 10