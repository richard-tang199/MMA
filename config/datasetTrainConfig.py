from dataclasses import dataclass, field


@dataclass
class SateConfig:
    group_name: str
    num_epochs: int = 101
    window_size: int = 512
    patch_size: int = 16
    stride: int = 32
    d_model: int = 64
    mode: str = "common_channel"
    num_channels: int = 9

    def __post_init__(self):
        if self.group_name == "real_satellite_data_3":
            self.mode = "mix_channel"



@dataclass
class ASD_Config:
    group: int
    num_epochs: int = 101
    window_size: int = 1024
    patch_size: int = 16
    num_channels: int = 19
    stride: int = 128
    train_interval: list = field(default_factory=lambda: [0, 8000])
    d_model: int = 64
    mode: str = "common_channel"

@dataclass
class SMD_Config:
    group: str
    num_epochs: int = 101
    window_size: int = 1024
    patch_size: int = 16
    num_channels: int = 38
    stride: int = 250
    d_model: int = 64
    mode: str = "common_channel"

@dataclass
class synthetic_Config:
    num_epochs: int = 101
    window_size: int = 1024
    patch_size: int = 16
    num_channels: int = 5
    stride: int = 128
    d_model: int = 64
    mode: str = "common_channel"

@dataclass
class TELCO_Config:
    window_size: int = 2048
    num_epochs: int = 101
    patch_size: int = 32
    num_channels: int = 12
    stride: int = 250
    d_model: int = 64
    mode: str = "common_channel"

@dataclass
class UCR_Config:
    group: str
    num_epochs: int = 201
    window_size: int = 256
    patch_size: int = 8
    num_channels: int = 1
    stride: int = 256
    d_model: int = 128
    mode: str = "common_channel"