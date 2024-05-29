from dataclasses import dataclass


@dataclass
class PatchDetectorConfig:
    num_epochs: int = 201
    learning_rate: float = 1e-3
    batch_size: int = 64
    window_length: int = 512
    patch_length: int = 16
    window_stride: int = 128
    d_model: int = 64
    num_channels: int = 9
    remove_anomaly: bool = False
    num_layers: int = 3
    device: str = "cuda"
    masking_mode: str = "random"  # "random" or "gating"
    forward_times: int = 1
    instance_normalization: bool = False
    mode: str = "common_channel"  # "mix_channel" or "common_channel"
    channel_consistent: bool = False
    mask_switching: bool = True
    mask_value: int = 0
    mask_ratio: float = 0.5
    use_position_encoder: bool = False
    positional_encoding_type: str = "sincos"  # "sincos" or "random"
    gated_attn: bool = True
    norm_mlp: str = "LayerNorm"
    self_attn: bool = False
    self_attn_heads: int = 1
    expansion_factor: int = 6
    dropout: float = 0.2
    norm_eps: float = 1e-5
    weight: float = 0.01
    anomaly_mode: str = "dynamic"  # "error" or "dynamic"

    def __post_init__(self):
        self.stride = self.patch_length  # patch_stride
        self.num_patches = (max(self.window_length, self.patch_length) - self.patch_length) // self.stride + 1
        if self.masking_mode == "gating":
            self.mask_ratio = 0.5
        # self.mode: str = "mix_channel" if not self.channel_consistent else "common_channel"
        if self.masking_mode == "random":
            self.forward_times = 5


@dataclass
class PatchDetectorAttentionConfig(PatchDetectorConfig):
    learning_rate: float = 2e-3
    instance_normalization: bool = False
    forward_times: int = 5
    d_model: int = 64
    remove_anomaly: bool = True
    d_ff: int = 16  # FeedForward: multiple of d_model
    self_attn_heads: int = 8  # MultiHeadAttention: number of heads
    use_position_encoder: bool = True
    attn_norm: str = "LayerNorm"  # Attention Norm: "LayerNorm" or "BatchNorm"
    attn_dropout: float = 0.2  # Attention Dropout
    res_attn: bool = True  # Residual Attention: True or False
    num_layers: int = 3 # number of layers in the transformer
    store_attn: bool = False  # Store Attention: True or False
    positional_encoding_type: str = "random"  # "sincos" or "random"

    def __post_init__(self):
        super().__post_init__()


@dataclass
class PatchDetectorGruConfig(PatchDetectorConfig):
    learning_rate: float = 1e-3
    instance_normalization: bool = False
    forward_times: int = 5
    d_model: int = 64
    num_layers: int = 3
    expansion_factor: int = 1



    def __post_init__(self):
        super().__post_init__()



if __name__ == '__main__':
    config = PatchDetectorAttentionConfig()
    print(config)
