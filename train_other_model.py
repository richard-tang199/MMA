import os.path
import sys
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import thop
import torch
from toolkit.result_plot import recon_plot
import argparse
from toolkit.load_dataset import load_dataset, load_pollute_dataset
from toolkit.load_config_data_model import get_dataloader, determine_window_patch_size
from toolkit.get_anomaly_score import AnomalyScoreCalculator
from evaluation.evaluate import evaluate, EfficiencyResult
import json

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="PatchAD")
parser.add_argument('--group', type=str, default="real_satellite_data_1", help='group number')
parser.add_argument("--learning_rate", type=float, default=2e-3, help="learning rate")
parser.add_argument('--data_name', type=str, default='synthetic', help='dataset name')
parser.add_argument("--window_length", type=int, default=100, help="window length")
parser.add_argument('--num_epochs', type=int, default=30, help="number of epochs")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--eval_gap", type=int, default=10, help="evaluation gap")
parser.add_argument('--figure_length', type=int, default=60, help="number of workers for dataloader")
parser.add_argument('--figure_width', type=int, default=20, help="number of workers for dataloader")
parser.add_argument('--anomaly_mode', type=str, default="error", help="anomaly mode")  # "error" or "dynamic"
parser.add_argument('--mode', type=str, default="normal", help="normal or robust verification")
parser.add_argument("--anomaly_ratio", type=float, default=2.0)

if __name__ == "__main__":
    sys.path.append("other_models")
    # assign params
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"\n model name: {args.model_name}, data name: {args.data_name}_{args.group}")
    now = datetime.now().strftime("%m-%d-%H-%M")
    model_name = args.model_name
    data_name = args.data_name
    args.smoother_window_size = None
    if data_name == "UCR":
        args.group = args.group.zfill(3)
    group = args.group
    raw_train_data, raw_test_data, raw_test_labels = load_dataset(data_name, args.group)

    if model_name == "TranAD":
        args.window_length = 10
    elif model_name == "mtad_gat":
        args.window_length = 100
    elif model_name == "gdn":
        args.window_length = 5
    elif model_name == "mtgflow":
        args.window_length = 60
        args.num_epochs = 50
    elif model_name == "NormFAAE":
        args.window_length = 128
        args.num_epochs = 300
    elif model_name == "FGANomaly":
        args.window_length = 120
    elif model_name == "MAUT":
        args.window_length = 100
        args.num_epochs = 100
    elif model_name == "usad":
        args.window_length = 5
        args.num_epochs = 100
    elif model_name == "cad":
        args.window_length = 20
        args.num_epochs = 30
    elif model_name == "PatchAD":
        args.window_length = 105
        args.num_epochs = 20

    if args.data_name == "SMD":
        args.figure_width = 40
    if args.data_name == "UCR":
        args.figure_length, args.figure_width = 160, 20
        _, _, main_period = determine_window_patch_size(raw_train_data)
        if args.group == "220":
            main_period = 400
        args.smoother_window_size = main_period

    if model_name == "MP" or model_name == "DAMP" or model_name == "KMeans":
        if args.data_name == "UCR":
            args.window_length = main_period
        else:
            args.window_length = 20

    if args.mode != "normal":
        if args.mode == "realistic":
            args.anomaly_ratio = int(args.anomaly_ratio)
        raw_train_data, raw_test_data, raw_train_labels, raw_test_labels = load_pollute_dataset(
            data_name=args.data_name,
            group=args.group,
            mode=args.mode,
            ratio=args.anomaly_ratio
        )
    else:
        args.anomaly_ratio = 0
        raw_train_labels = None

    if len(raw_train_data.shape) == 1:
        raw_train_data = raw_train_data[:, None]
        raw_test_data = raw_test_data[:, None]

    output_dir = (f"output/{model_name}/{data_name}/{args.mode}/"
                  f"{args.mode}_{args.anomaly_ratio}/window_length_{args.window_length}")

    if data_name in ["ASD", "SMD", "UCR", "sate"]:
        output_dir = (f"output/{model_name}/{data_name}/{args.mode}/"
                      f"{args.mode}_{args.anomaly_ratio}/{data_name}_{group}/window_length_{args.window_length}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_name == "FGANomaly":
        from FGANomaly_main.main import params, FGANomalyModel, RNNAutoEncoder, MLPDiscriminator

        if "sate" in args.data_name:
            params["epoch"] = 50
        elif args.data_name == "synthetic":
            params["epoch"] = 100

        params["epoch"] = 1
        train_data = raw_train_data
        valid_data = raw_train_data
        _, num_channels = train_data.shape
        train_loader, _ = get_dataloader(train_data, batch_size=params["batch_size"],
                                         window_length=args.window_length,
                                         window_stride=params["stride"],
                                         mode="train",
                                         if_shuffle=False)
        data = {"train": train_loader, "val": valid_data,
                "test": (raw_test_data, raw_test_labels), "nc": num_channels}

        params["best_model_path"] = output_dir
        model = FGANomalyModel(ae=RNNAutoEncoder(inp_dim=data['nc'],
                                                 z_dim=params['z_dim'],
                                                 hidden_dim=params['hidden_dim'],
                                                 rnn_hidden_dim=params['rnn_hidden_dim'],
                                                 num_layers=params['num_layers'],
                                                 bidirectional=params['bidirectional'],
                                                 cell=params['cell']),
                               dis_ar=MLPDiscriminator(inp_dim=data['nc'],
                                                       hidden_dim=params['hidden_dim']),
                               data_loader=data, **params)
        model.train()
        recon_train = model.test(train_data)

        start_time = time.time()
        recon_test = model.test(raw_test_data)
        duration = time.time() - start_time

        train_data = train_data[:len(recon_train)]
        test_data = raw_test_data[:len(recon_test)]

        # recon_train[::params["window_size"], :] = train_data[::params["window_size"], :]
        # recon_test[::params["window_size"], :] = test_data[::params["window_size"], :]

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=train_data,
            raw_test_data=test_data,
            recon_train_data=recon_train,
            recon_test_data=recon_test
        )

        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all
        test_result = evaluate(test_anomaly_score, raw_test_labels[:len(recon_test)], pa=True)
        threshold = test_result.best_threshold_wo_pa

        sample_input = torch.randn(1, 1024, data['nc'], device=model.device)
        flops, params = thop.profile(model.best_ae, inputs=(sample_input,))
        flops, params = thop.clever_format([flops, params])

    elif model_name == "TranAD":
        from TranAD.tran_ad import TranAD

        if "sate" in args.data_name:
            args.num_epochs = 20
        elif args.data_name == "synthetic":
            args.num_epochs = 100
        _, num_channels = raw_train_data.shape
        model = TranAD(feats=num_channels, window_length=args.window_length)
        model = model.to(device=device)
        window_pad = np.ones([model.n_window - 1, num_channels])
        new_train_data = np.concatenate((window_pad, raw_train_data), axis=0)
        new_test_data = np.concatenate((window_pad, raw_test_data), axis=0)
        train_loader, _ = get_dataloader(new_train_data,
                                         batch_size=args.batch_size,
                                         window_length=model.n_window,
                                         window_stride=1,
                                         mode="train",
                                         if_shuffle=True)
        val_loader, _ = get_dataloader(new_train_data,
                                       batch_size=args.batch_size,
                                       window_length=model.n_window,
                                       window_stride=1,
                                       mode="test",
                                       if_shuffle=False)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=args.batch_size,
                                        window_length=model.n_window,
                                        window_stride=1,
                                        mode="test",
                                        if_shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        model.fit(train_loader=train_loader, epochs=args.num_epochs,
                  optimizer=optimizer, scheduler=scheduler)
        recon_train = model.predict(val_loader)

        start_time = time.time()
        recon_test = model.predict(test_loader)
        duration = time.time() - start_time

        train_data = raw_train_data
        test_data = raw_test_data

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=train_data,
            raw_test_data=test_data,
            recon_train_data=recon_train,
            recon_test_data=recon_test
        )

        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all
        test_result = evaluate(test_anomaly_score, raw_test_labels[:len(recon_test)], pa=True)
        threshold = test_result.best_threshold_wo_pa

        sample_input = (torch.randn(10, 1, num_channels, device=device),
                        torch.randn(1, 1, num_channels, device=device))
        flops, params = thop.profile(model, inputs=sample_input)
        flops, params = thop.clever_format([flops, params])

    elif model_name == "mtad_gat":
        from mtad_gat.mtad_gat import MTAD_GAT
        import yaml
        from types import SimpleNamespace

        with open("other_models/mtad_gat/configs.yaml", "r") as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
        configs = SimpleNamespace(**configs)
        _, num_channels = raw_train_data.shape

        configs.data_set = data_name
        configs.n_features = num_channels
        configs.out_dim = num_channels
        configs.window_size = args.window_length - 1

        window_pad = np.ones([configs.window_size, num_channels])
        new_train_data = np.concatenate((window_pad, raw_train_data), axis=0)
        new_test_data = np.concatenate((window_pad, raw_test_data), axis=0)

        train_loader, _ = get_dataloader(new_train_data,
                                         batch_size=args.batch_size,
                                         window_length=args.window_length,
                                         mode="train",
                                         window_stride=1,
                                         if_shuffle=True)
        val_loader, _ = get_dataloader(new_train_data,
                                       batch_size=args.batch_size,
                                       window_length=args.window_length,
                                       window_stride=1,
                                       mode="test",
                                       if_shuffle=False)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=args.batch_size,
                                        window_length=args.window_length,
                                        window_stride=1,
                                        mode="test",
                                        if_shuffle=False)

        model = MTAD_GAT(**configs.__dict__)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.init_lr)
        duration_list = model.fit(train_loader=train_loader, epochs=args.num_epochs, optimizer=optimizer)
        predict_train, recon_train_0 = model.predict(val_loader)

        start_time = time.time()
        predict_test, recon_test_0 = model.predict(test_loader)
        duration = time.time() - start_time

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        predict_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            recon_train_data=predict_train,
            recon_test_data=predict_test
        )
        recon_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            recon_train_data=recon_train_0,
            recon_test_data=recon_test_0
        )

        recon_train = (predict_train + recon_train_0) / 2
        recon_test = (predict_test + recon_test_0) / 2
        test_anomaly_score = predict_score.test_score_all + recon_score.test_score_all
        train_anomaly_score = predict_score.train_score_all + recon_score.train_score_all
        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)

        sample_input = torch.randn(1, configs.window_size, num_channels, device=device)
        flops, params = thop.profile(model, inputs=(sample_input,))
        flops, params = thop.clever_format([flops, params])

        test_data = raw_test_data
        train_data = raw_train_data
        threshold = test_result.best_threshold_wo_pa

    elif model_name == "gdn":
        from GDN_main.gdn import GDN

        _, num_channels = raw_train_data.shape
        window_length = args.window_length - 1

        model = GDN(node_num=num_channels, input_dim=window_length, topk=int(num_channels * 0.3))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

        window_pad = np.ones([window_length, num_channels])
        new_train_data = np.concatenate((window_pad, raw_train_data), axis=0)
        new_test_data = np.concatenate((window_pad, raw_test_data), axis=0)

        train_loader, _ = get_dataloader(new_train_data,
                                         batch_size=args.batch_size,
                                         window_length=args.window_length,
                                         mode="train",
                                         window_stride=1,
                                         if_shuffle=True)
        val_loader, _ = get_dataloader(new_train_data,
                                       batch_size=args.batch_size,
                                       window_length=args.window_length,
                                       window_stride=1,
                                       mode="test",
                                       if_shuffle=False)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=args.batch_size,
                                        window_length=args.window_length,
                                        window_stride=1,
                                        mode="test",
                                        if_shuffle=False)

        model.fit(train_loader=train_loader, epochs=args.num_epochs, optimizer=optimizer)
        recon_train = model.predict(val_loader)

        start_time = time.time()
        recon_test = model.predict(test_loader)
        duration = time.time() - start_time

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            recon_train_data=recon_train,
            recon_test_data=recon_test
        )
        test_score_channels = test_score.test_channel_score
        train_score_channels = test_score.train_channel_score
        train_anomaly_score = np.max(train_score_channels, axis=1)
        test_anomaly_score = np.max(test_score_channels, axis=1)
        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)

        sample_input = torch.randn(1, window_length, num_channels, device=device)
        flops, params = thop.profile(model, inputs=(sample_input,))
        flops, params = thop.clever_format([flops, params])

        test_data = raw_test_data
        train_data = raw_train_data
        threshold = test_result.best_threshold_wo_pa

    elif model_name == "mtgflow":
        from mtgflow_main.mtgflow import MTGFLOW, configs

        _, num_channels = raw_train_data.shape
        configs["n_sensor"] = num_channels
        model = MTGFLOW(**configs)
        model = model.to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'weight_decay': configs['weight_decay']}],
            lr=configs["lr"], weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        window_pad = np.ones([configs["window_size"] - 1, num_channels])
        new_train_data = np.concatenate((window_pad, raw_train_data), axis=0)
        new_test_data = np.concatenate((window_pad, raw_test_data), axis=0)

        train_loader, _ = get_dataloader(new_train_data,
                                         batch_size=args.batch_size,
                                         window_length=configs["window_size"],
                                         mode="train",
                                         window_stride=1,
                                         if_shuffle=True)
        val_loader, _ = get_dataloader(new_train_data,
                                       batch_size=args.batch_size,
                                       window_length=configs["window_size"],
                                       window_stride=1,
                                       mode="test",
                                       if_shuffle=False)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=args.batch_size,
                                        window_length=configs["window_size"],
                                        window_stride=1,
                                        mode="test",
                                        if_shuffle=False)

        model.fit(data_loader=train_loader, epochs=args.num_epochs, optimizer=optimizer, scheduler=scheduler)
        train_density = model.predict(val_loader)
        start_time = time.time()
        test_density = model.predict(test_loader)
        duration = time.time() - start_time

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=np.zeros_like(train_density),
            raw_test_data=np.zeros_like(test_density),
            recon_train_data=train_density,
            recon_test_data=test_density
        )

        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all
        train_anomaly_score = MinMaxScaler().fit_transform(train_anomaly_score.reshape(-1, 1)).flatten()
        test_anomaly_score = MinMaxScaler().fit_transform(test_anomaly_score.reshape(-1, 1)).flatten()

        train_anomaly_score[:len(window_pad)] = 0
        test_anomaly_score[:len(window_pad)] = 0

        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)

        sample_input = torch.randn(1, configs["window_size"], num_channels, device=device)
        flops, params = thop.profile(model, inputs=(sample_input,))
        flops, params = thop.clever_format([flops, params])

        test_data = raw_test_data
        train_data = raw_train_data
        threshold = test_result.best_threshold_wo_pa
        recon_test = None
        recon_train = None

    elif model_name == "NormFAAE":
        from NormFAAE_main.data_loader import get_statistics, delete_unique, SegDataLoader
        from NormFAAE_main.main import train_test
        from torch.utils.data import random_split
        from torch.utils.data import DataLoader

        raw_train_data, raw_test_data, num_channels = delete_unique(raw_train_data, raw_test_data)
        dis, mins, mea, std, con = get_statistics(raw_train_data)
        train_set = SegDataLoader(data=raw_train_data, win_size=args.window_length, step=8)
        test_data = SegDataLoader(data=raw_test_data, win_size=args.window_length,
                                  step=args.window_length, label=raw_test_labels)
        train_size = int(len(train_set) * 0.8)
        valid_size = len(train_set) - train_size
        train_data_input, valid_data_input = random_split(train_set, [train_size, valid_size])
        train_loader = DataLoader(train_data_input, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        valid_loader = DataLoader(valid_data_input, args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        raw_test_labels, test_anomaly_score, test_data, recon_test, flops, params, duration = train_test(
            n_features=num_channels, num_hiddens=128, num_epochs=args.num_epochs,
            lr1=1e-4, lr2=1e-4, weight_decay=1e-4, patience=5, data_name=args.data_name,
            model_path=output_dir, train_data=train_loader, valid_data=valid_loader, test_data=test_loader,
            mea_=mea, std_=std, dis_=dis, min_=mins, con_=con, alpha=1, Lambda=1, device="cuda")
        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        threshold = test_result.best_threshold_wo_pa
        recon_train = None
        train_data = raw_train_data
        train_anomaly_score = None

    elif model_name == "MAUT":
        from MemAugUTransAD_main.model_pyramid_trans_mem import PYRAMID_TRANS_MEM

        _, num_channels = raw_train_data.shape
        model = PYRAMID_TRANS_MEM(n_features=num_channels, window_size=args.window_length, out_dim=num_channels)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        train_loader, train_window_converter = get_dataloader(raw_train_data,
                                                              batch_size=args.batch_size,
                                                              window_length=args.window_length,
                                                              mode="train")

        valid_loader, valid_window_converter = get_dataloader(raw_train_data,
                                                              batch_size=args.batch_size,
                                                              window_length=args.window_length,
                                                              window_stride=args.window_length,
                                                              mode="test")

        test_loader, test_window_converter = get_dataloader(raw_test_data,
                                                            batch_size=args.batch_size,
                                                            window_length=args.window_length,
                                                            window_stride=args.window_length,
                                                            mode="test")
        duration_list = model.fit(train_loader=train_loader, epochs=args.num_epochs, optimizer=optimizer)
        recon_train = model.predict(valid_loader)
        recon_train = valid_window_converter.windows_to_sequence(recon_train)

        start_time = time.time()
        recon_test = model.predict(test_loader)
        recon_test = test_window_converter.windows_to_sequence(recon_test)
        duration = time.time() - start_time

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            recon_train_data=recon_train,
            recon_test_data=recon_test
        )
        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all

        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        test_data = raw_test_data
        train_data = raw_train_data
        threshold = test_result.best_threshold_wo_pa

        sample_input = torch.randn(1, args.window_length, num_channels, device=device)
        flops, params = thop.profile(model, inputs=(sample_input,))
        flops, params = thop.clever_format([flops, params])

    elif model_name == "usad":
        from usad_main.usad import UsadModel

        _, num_channels = raw_train_data.shape
        model = UsadModel(w_size=args.window_length * num_channels, z_size=args.window_length * 40)
        model = model.to(device)
        train_loader, _ = get_dataloader(raw_train_data,
                                         batch_size=args.batch_size,
                                         window_length=args.window_length,
                                         mode="train")

        valid_loader, valid_window_converter = get_dataloader(raw_train_data,
                                                              batch_size=args.batch_size,
                                                              window_length=args.window_length,
                                                              window_stride=args.window_length,
                                                              mode="test")

        test_loader, test_window_converter = get_dataloader(raw_test_data,
                                                            batch_size=args.batch_size,
                                                            window_length=args.window_length,
                                                            window_stride=args.window_length,
                                                            mode="test")
        model.fit(epochs=args.num_epochs, train_loader=train_loader)
        recon_train = model.predict(valid_loader)
        recon_train = valid_window_converter.windows_to_sequence(recon_train)

        start_time = time.time()
        recon_test = model.predict(test_loader)
        duration = time.time() - start_time
        recon_test = test_window_converter.windows_to_sequence(recon_test)

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            recon_train_data=recon_train,
            recon_test_data=recon_test
        )
        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all

        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        test_data = raw_test_data
        train_data = raw_train_data
        threshold = test_result.best_threshold_wo_pa

        sample_input = torch.randn(1, args.window_length, num_channels, device=device)
        sample_input = sample_input.view(1, -1)
        flop, param = thop.profile(model, inputs=(sample_input, 1))
        flops, params = thop.clever_format([flop, param])

    elif model_name == "cad":
        from CAD_main.cad import MMoE

        _, num_channels = raw_train_data.shape
        window_size = args.window_length - 1
        window_pad = np.ones([window_size, num_channels])
        new_train_data = np.concatenate((window_pad, raw_train_data), axis=0)
        new_test_data = np.concatenate((window_pad, raw_test_data), axis=0)

        train_loader, _ = get_dataloader(new_train_data,
                                         batch_size=args.batch_size,
                                         window_length=args.window_length,
                                         mode="train",
                                         window_stride=1,
                                         if_shuffle=True)
        val_loader, _ = get_dataloader(new_train_data,
                                       batch_size=args.batch_size,
                                       window_length=args.window_length,
                                       window_stride=1,
                                       mode="test",
                                       if_shuffle=False)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=args.batch_size,
                                        window_length=args.window_length,
                                        window_stride=1,
                                        mode="test",
                                        if_shuffle=False)

        model = MMoE(n_multiv=num_channels, window_size=args.window_length - 3)
        model = model.to(device)

        duration_list = model.fit(train_loader=train_loader, epochs=args.num_epochs)

        recon_train = model.predict(val_loader)
        start_time = time.time()
        recon_test = model.predict(test_loader)
        duration = time.time() - start_time

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error", average_window=args.smoother_window_size)
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            recon_train_data=recon_train,
            recon_test_data=recon_test
        )

        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all

        test_anomaly_score[:window_size] = 0
        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        sample_input = torch.randn(1, args.window_length - 3, num_channels, device=device)
        flops, params = thop.profile(model, inputs=(sample_input,))
        flops, params = thop.clever_format([flops, params])

        test_data = raw_test_data
        train_data = raw_train_data
        threshold = test_result.best_threshold_wo_pa

    elif model_name == "PatchAD":
        from PatchAD.patch_ad import Solver

        _, num_channels = raw_train_data.shape

        train_loader, _ = get_dataloader(raw_train_data,
                                         batch_size=args.batch_size,
                                         window_length=args.window_length,
                                         mode="train",
                                         if_shuffle=True)
        val_loader, valid_window_converter = get_dataloader(raw_train_data,
                                                            batch_size=args.batch_size,
                                                            window_length=args.window_length,
                                                            window_stride=args.window_length,
                                                            mode="test",
                                                            if_shuffle=False)
        test_loader, test_window_converter = get_dataloader(raw_test_data,
                                                            batch_size=args.batch_size,
                                                            window_length=args.window_length,
                                                            window_stride=args.window_length,
                                                            mode="test",
                                                            if_shuffle=False)

        model = Solver(epochs=args.num_epochs, window_size=args.window_length, channels=num_channels)
        model.fit(train_loader=train_loader)

        train_anomaly_score = model.test(test_loader=val_loader)
        train_anomaly_score = valid_window_converter.windows_to_sequence(train_anomaly_score)
        test_anomaly_score = model.test(test_loader=test_loader)
        test_anomaly_score = test_window_converter.windows_to_sequence(test_anomaly_score)
        # train_anomaly_score = train_anomaly_score.squeeze(-1)
        # test_anomaly_score = test_anomaly_score.squeeze(-1)

        anomaly_score_calculator = AnomalyScoreCalculator(mode="error")
        test_score = anomaly_score_calculator.calculate_anomaly_score(
            raw_train_data=np.zeros_like(train_anomaly_score),
            raw_test_data=np.zeros_like(test_anomaly_score),
            recon_train_data=train_anomaly_score,
            recon_test_data=test_anomaly_score
        )
        test_anomaly_score = test_score.test_score_all
        train_anomaly_score = test_score.train_score_all

        train_anomaly_score = MinMaxScaler().fit_transform(train_anomaly_score.reshape(-1, 1)).flatten()
        test_anomaly_score = MinMaxScaler().fit_transform(test_anomaly_score.reshape(-1, 1)).flatten()

        test_data = raw_test_data
        train_data = raw_train_data
        recon_train = None
        recon_test = None
        flops = 0
        params = 0
        duration_list = [0]
        test_result = evaluate(test_anomaly_score, raw_test_labels[:len(test_anomaly_score)], pa=True)
        threshold = test_result.best_threshold_wo_pa

    elif model_name == "MP":
        import stumpy
        from numba import cuda

        all_gpu_devices = [device.id for device in cuda.list_devices()]

        data = np.concatenate((raw_train_data, raw_test_data), axis=0)
        # covert to float64
        data = data.astype(np.float64)
        data = np.squeeze(data)
        start_time = time.time()
        matrix_profile = stumpy.gpu_stump(data, m=args.window_length, device_id=all_gpu_devices)[:, 0]
        duration = time.time() - start_time
        test_anomaly_score = matrix_profile[raw_train_data.shape[0]:]
        test_anomaly_score = (test_anomaly_score - test_anomaly_score.min()) / (
                test_anomaly_score.max() - test_anomaly_score.min())
        test_result = evaluate(test_anomaly_score, raw_test_labels[:len(test_anomaly_score)], pa=True)
        recon_train = None
        recon_test = None
        flops = 0
        params = 0
        train_data = raw_train_data
        test_data = raw_test_data[:len(test_anomaly_score)]
        threshold = test_result.best_threshold_wo_pa
        train_anomaly_score = None

    elif model_name == "DAMP":
        from DAMP_main.damp import DAMP_2_0

        data = np.concatenate((raw_train_data, raw_test_data), axis=0)
        # covert to float64
        data = data.astype(np.float64)
        data = np.squeeze(data)
        start_time = time.time()
        left_mp, _, _ = DAMP_2_0(time_series=data,
                                 subsequence_length=args.window_length,
                                 stride=1,
                                 location_to_start_processing=len(raw_train_data),
                                 lookahead=None
                                 )
        duration = time.time() - start_time
        test_anomaly_score = left_mp[len(raw_train_data):]
        test_anomaly_score = (test_anomaly_score - test_anomaly_score.min()) / (
                test_anomaly_score.max() - test_anomaly_score.min())
        test_result = evaluate(test_anomaly_score, raw_test_labels[:len(test_anomaly_score)], pa=True)
        recon_train = None
        recon_test = None
        flops = 0
        params = 0
        train_data = raw_train_data
        test_data = raw_test_data
        threshold = test_result.best_threshold_wo_pa
        train_anomaly_score = None

    elif model_name == "KMeans":
        from KMeans.ano_kmeans import KMeansAD

        data = np.concatenate((raw_train_data, raw_test_data), axis=0)
        model = KMeansAD(k=20, window_size=args.window_length, stride=1)
        start_time = time.time()
        anomaly_scores = model.fit_predict(data)
        assert len(anomaly_scores) == len(data)
        duration = time.time() - start_time

        train_anomaly_score = anomaly_scores[:len(raw_train_data)]
        test_anomaly_score = anomaly_scores[-len(raw_test_data):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_anomaly_score.reshape(-1, 1))
        train_anomaly_score = scaler.transform(train_anomaly_score.reshape(-1, 1)).flatten()
        test_anomaly_score = scaler.fit_transform(test_anomaly_score.reshape(-1, 1)).flatten()

        test_result = evaluate(test_anomaly_score, raw_test_labels, pa=True)
        recon_train = None
        recon_test = None
        flops = 0
        params = 0
        duration_list = [duration]
        train_data = raw_train_data
        test_data = raw_test_data
        threshold = test_result.best_threshold_wo_pa
    else:
        raise ValueError("Invalid model name")

    if recon_train is not None:
        if len(recon_train.shape) == 1:
            recon_train = recon_train[:, np.newaxis]

    if recon_test is not None:
        if len(recon_test.shape) == 1:
            recon_test = recon_test[:, np.newaxis]

    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        json.dump(test_result.__dict__, f, indent=4)

    efficiency_result = EfficiencyResult(test_time=duration, flops=flops, params=params,
                                         average_epoch_time=np.median(duration_list),
                                         all_training_time=np.sum(duration_list))
    efficiency_result = efficiency_result.__dict__
    with open(os.path.join(output_dir, f"efficiency_result.json"), "w") as f:
        json.dump(efficiency_result, f, indent=4)

    figure_save_path = os.path.join(output_dir, f"result.png")
    if recon_train is not None and recon_test is not None:
        gap = (recon_train.shape[0] + recon_test.shape[0]) // 80
    else:
        gap = 400

    recon_plot(
        save_path=figure_save_path,
        gap=gap,
        figure_length=args.figure_length,
        figure_width=args.figure_width,
        font_size=4,
        test_data=test_data,
        test_label=raw_test_labels,
        recon_test_data=recon_test,
        test_anomaly_score=test_anomaly_score,
        train_data=train_data,
        train_label=raw_train_labels,
        recon_train_data=recon_train,
        train_anomaly_score=train_anomaly_score,
        threshold=threshold,
        plot_diff=True
    )

    # save recon and anomaly_score
    np.save(os.path.join(output_dir, f"raw_test_data.npy"), test_data)
    np.save(os.path.join(output_dir, f"raw_test_labels.npy"), raw_test_labels)
    np.save(os.path.join(output_dir, f"test_anomaly_score.npy"), test_anomaly_score)
    np.save(os.path.join(output_dir, f"train_anomaly_score.npy"), train_anomaly_score)
    np.save(os.path.join(output_dir, f"recon_test_data.npy"), recon_test)

    # save model
    torch.save(model, os.path.join(output_dir, f"model.pth"))
