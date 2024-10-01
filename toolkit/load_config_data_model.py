from config.patchDetectorConfig import *
from config.datasetTrainConfig import *
from toolkit.load_dataset import SequenceWindowConversion
from torch.utils.data import TensorDataset, DataLoader
from model.patch_detector import *
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram


def find_length(data):
    if len(data.shape) > 1:
        data = data.flatten()
    data = data[:min(200000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 20 or local_max[max_local_max] > 1000:
            freq, power = periodogram(data)
            period = int(1 / freq[np.argmax(power)])
        else:
            period = local_max[max_local_max] + base
    except:
        freq, power = periodogram(data)
        period = int(1 / freq[np.argmax(power)])

    if period > 10000:
        period = 400

    return period


def determine_window_patch_size(train_data: np.ndarray, patch_num=8, multiple=32):
    if len(train_data.shape) > 1:
        input_data = train_data.squeeze(-1)
    else:
        input_data = train_data
    main_period = find_length(train_data)
    patch_size = int(np.ceil(main_period / patch_num))
    if patch_size == 0:
        patch_size = 1
    window_size = patch_size * multiple

    data_length = len(input_data)
    if data_length < window_size:
        window_size = data_length

    return window_size, patch_size, main_period


def load_train_config(args):
    if "sate" in args.data_name:
        data_config = SateConfig(group=args.group)
    elif args.data_name == "ASD":
        data_config = ASD_Config(group=args.group)
    elif args.data_name == "SMD":
        data_config = SMD_Config(group=args.group)
    elif args.data_name == "synthetic":
        data_config = synthetic_Config()
    elif args.data_name == "TELCO":
        data_config = TELCO_Config()
    elif args.data_name == "UCR":
        data_config = UCR_Config(group=args.group)
    else:
        raise NotImplementedError

    assert args.model_name in ["PatchDetector", "PatchAttention", "PatchDenoise", "PatchContrast", "PatchGru"]

    if args.model_name in ["PatchDetector", "PatchContrast"]:
        window_stride = (args.window_length // 4)
        if window_stride == 0:
            window_stride = 1

        train_config = PatchDetectorConfig(
            num_epochs=data_config.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            remove_anomaly=args.remove_anomaly,
            window_length=args.window_length,
            patch_length=data_config.patch_size,
            window_stride=window_stride,
            num_channels=data_config.num_channels,
            d_model=data_config.d_model,
            mode=data_config.mode,
            anomaly_mode=args.anomaly_mode,
        )
    elif args.model_name == "PatchAttention":
        train_config = PatchDetectorAttentionConfig(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            remove_anomaly=args.remove_anomaly,
            window_length=data_config.window_size,
            patch_length=data_config.patch_size,
            window_stride=data_config.stride,
            num_channels=data_config.num_channels,
            d_model=data_config.d_model,
            mode=data_config.mode,
            anomaly_mode=args.anomaly_mode
        )
    elif args.model_name == "PatchGru":
        train_config = PatchDetectorGruConfig(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            remove_anomaly=args.remove_anomaly,
            window_length=data_config.window_size,
            patch_length=data_config.patch_size,
            window_stride=data_config.stride,
            num_channels=data_config.num_channels,
            d_model=data_config.d_model,
            mode=data_config.mode,
            anomaly_mode=args.anomaly_mode
        )
    elif args.model_name == "PatchDenoise":
        train_config = PatchDetectorConfig(
            num_epochs=data_config.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            window_length=data_config.window_size,
            patch_length=data_config.patch_size,
            window_stride=data_config.stride,
            num_channels=data_config.num_channels,
            d_model=data_config.d_model,
            mode=data_config.mode,
            anomaly_mode=args.anomaly_mode,
        )
    else:
        raise NotImplementedError

    return train_config


def get_dataloader(data: np.ndarray, batch_size: int, window_length: int,
                   window_stride: int = None, mode="test", if_shuffle=True):
    assert mode in ["train", "test"]
    length = data.shape[0]
    if mode == "train":
        stride_size = 2 * (length // 4000)
        if stride_size == 0:
            stride_size = 2
        elif stride_size > window_length:
            stride_size = window_length
        if window_stride is not None:
            stride_size = window_stride
        batch_size = batch_size
    else:
        stride_size = window_stride
        batch_size = 2 * batch_size
        if_shuffle = False

    window_converter = SequenceWindowConversion(
        window_size=window_length,
        stride_size=stride_size,
        mode=mode
    )
    windows = window_converter.sequence_to_windows(data)
    windows = torch.tensor(windows)
    dataset = TensorDataset(windows)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=if_shuffle,
        drop_last=False
    )

    return data_loader, window_converter


def get_model(model_name: str, train_config):
    if model_name == "PatchDetector":
        model = PatchDetector(config=train_config)
    elif model_name == "PatchAttention":
        model = PatchDetectorAttention(config=train_config)
    elif model_name == "PatchContrast":
        model = PatchContrastDetector(config=train_config)
    elif model_name == "PatchGru":
        model = PatchDetectorGru(config=train_config)
    else:
        raise NotImplementedError
    model = model.to(train_config.device)
    return model
