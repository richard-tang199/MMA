import os.path
import time
from datetime import datetime
import thop
import torch
import tqdm
import numpy as np
from train import train_one_epoch
from toolkit.load_config_data_model import load_train_config, get_dataloader, get_model, determine_window_patch_size
import argparse
from test import test, get_threshold, Test_output
from toolkit.load_dataset import load_dataset, load_pollute_dataset
from toolkit.result_plot import recon_plot, score_plot
from toolkit.get_anomaly_score import AnomalyScoreCalculator
from evaluation.evaluate import evaluate
from evaluation.evaluate import EfficiencyResult
import matplotlib.pyplot as plt
import json
from torch.utils.tensorboard import SummaryWriter

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="PatchAttention")  # PatchDetector or PatchAttention or PatchDenoise or PatchContrast
parser.add_argument('--group', type=str, default="013", help='group number')
parser.add_argument("--learning_rate", type=float, default=2e-3, help="learning rate")
parser.add_argument('--data_name', type=str, default='UCR', help='dataset name')
parser.add_argument('--num_epochs', type=int, default=201, help="number of epochs")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--eval_gap", type=int, default=200, help="evaluation gap")
parser.add_argument('--figure_length', type=int, default=60, help="number of workers for dataloader")
parser.add_argument('--figure_width', type=int, default=20, help="number of workers for dataloader")
parser.add_argument('--remove_anomaly', type=int, default=1, help="whether to remove anomaly or not")
parser.add_argument('--anomaly_mode', type=str, default="error", help="anomaly mode")  # "error" or "dynamic"
parser.add_argument('--plot', type=bool, default=True, help="plot the result or not")
parser.add_argument('--mode', type=str, default="normal", help="normal or robust verification")
parser.add_argument("--anomaly_ratio", type=float, default=2)
parser.add_argument("--multiple", type=int, default=32)
parser.add_argument("--window_length", type=int, default=1024, help="window length")
parser.add_argument("--use_tensorboard", type=bool, default=True, help="whether to use tensorboard or not")
parser.add_argument("--weight", type=float, default=0.005, help="contrastive loss weight")

if __name__ == "__main__":
    # assign params
    args = parser.parse_args()
    print(f"\n model name: {args.model_name}, data name: {args.data_name}_{args.group}")
    now = datetime.now().strftime("%m-%d-%H-%M")
    model_name = args.model_name
    data_name = args.data_name
    print_gap = args.eval_gap

    args.remove_anomaly = int(args.remove_anomaly)
    if args.remove_anomaly == 1:
        args.remove_anomaly = True
    else:
        args.remove_anomaly = False

    if args.data_name == "SMD":
        # print_gap = 100
        args.figure_length, args.figure_width = 60, 40
    elif args.data_name == "synthetic":
        # print_gap = 100
        args.figure_length, args.figure_width = 60, 20
    elif args.data_name == "ASD":
        # print_gap = 100
        args.figure_length, args.figure_width = 40, 20
    elif "sate" in args.data_name:
        args.figure_length, args.figure_width = 20, 20
    elif "UCR" in args.data_name:
        args.figure_length, args.figure_width = 160, 20
        args.group = args.group.zfill(3)

    group = args.group
    train_config = load_train_config(args)
    raw_train_data, raw_test_data, raw_test_labels = load_dataset(data_name, args.group)
    raw_train_labels = None

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

    if args.data_name == "UCR":
        train_config.window_length, train_config.patch_length, main_period = determine_window_patch_size(raw_train_data,
                                                                                                         multiple=args.multiple)
        if train_config.window_stride == 0:
            train_config.window_stride = 1
        train_config.stride = train_config.patch_length
        train_config.window_stride = train_config.patch_length // 2
        train_config.num_patches = (max(train_config.window_length,
                                        train_config.patch_length) - train_config.patch_length) // train_config.stride + 1
        train_subsequence_num = int(len(raw_train_data) // main_period)
        test_subsequence_num = int(len(raw_test_data) // main_period)
        train_config.train_subsequence_num = train_subsequence_num
        train_config.test_subsequence_num = test_subsequence_num

    print(train_config.__dict__)
    print(args.__dict__)

    # TODO: add anomaly mode
    output_dir = (
        f"output/{model_name}/{data_name}/{args.mode}/{args.mode}_{args.anomaly_ratio}/"
        f"window_len_{train_config.window_length}"
        f"-d_model_{train_config.d_model}-patch_len_{train_config.patch_length}"
        f"_remove_anomaly_{train_config.remove_anomaly}"
        f"-mode_{train_config.masking_mode}-weight_{args.weight}")

    if data_name in ["ASD", "SMD", "sate"]:
        output_dir = (
            f"output/{model_name}/{data_name}/{args.mode}/{args.mode}_{args.anomaly_ratio}/{data_name}_{group}/"
            f"window_len_{train_config.window_length}-d_model_{train_config.d_model}-patch_len_{train_config.patch_length}"
            f"-remove_anomaly_{train_config.remove_anomaly}"
            f"-mode_{train_config.masking_mode}-weight_{args.weight}")

    if data_name == "UCR":
        output_dir = (
            f"output/{model_name}/{data_name}/{args.mode}/{args.mode}_{args.anomaly_ratio}/{data_name}_{group}/"
            f"{args.multiple}/d_model_{train_config.d_model}-patch_len_{train_config.patch_length}"
            f"-remove_anomaly_{train_config.remove_anomaly}"
            f"-mode_{train_config.masking_mode}-weight_{args.weight}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_params = train_config.__dict__
    param_save_path = os.path.join(output_dir, f"params.json")
    with open(param_save_path, "w") as file:
        json.dump(train_params, file, indent=4)

    # initial tensorboard
    now = datetime.now().strftime("%m-%d-%H-%M")
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs", now))

    train_labels = np.zeros(raw_train_data.shape[0])
    if len(raw_train_data.shape) == 1:
        raw_train_data = raw_train_data[:, None]
        raw_test_data = raw_test_data[:, None]

    # get dataloader and window_converter
    train_loader, train_window_converter = get_dataloader(raw_train_data,
                                                          batch_size=train_config.batch_size,
                                                          window_length=train_config.window_length,
                                                          window_stride=train_config.window_stride if data_name == "UCR" else None,
                                                          mode="train")

    valid_loader, valid_window_converter = get_dataloader(raw_train_data,
                                                          batch_size=train_config.batch_size,
                                                          window_length=train_config.window_length,
                                                          window_stride=main_period if data_name == "UCR" else train_config.window_stride,
                                                          mode="test")

    test_loader, test_window_converter = get_dataloader(raw_test_data,
                                                        batch_size=train_config.batch_size,
                                                        window_length=train_config.window_length,
                                                        window_stride=main_period if data_name == "UCR" else train_config.window_stride,
                                                        mode="test")

    # get model
    model = get_model(args.model_name, train_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(train_loader), 0.95)

    total_train_loss = []
    total_valid_loss = []
    duration_list = []
    flops = None
    params = None
    latency = None
    if print_gap > train_config.num_epochs:
        print_gap = train_config.num_epochs - 1
    for epoch in tqdm.trange(train_config.num_epochs):
        epoch_output = train_one_epoch(current_epoch=epoch, weight=args.weight,
                                       model=model, train_loader=train_loader,
                                       optimizer=optimizer, scheduler=scheduler,
                                       device=train_config.device, save_dir=output_dir)
        epoch_loss = epoch_output["epoch_loss"]
        duration = epoch_output["epoch_duration"]
        total_train_loss.append([epoch, epoch_loss])
        duration_list.append(duration)

        if args.use_tensorboard:
            writer.add_scalar("train_loss", epoch_loss, epoch)
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("contrast_loss", epoch_output["epoch_contrast_loss"], epoch)
            writer.add_scalar("recon_loss", epoch_output["epoch_recon_loss"], epoch)

        if epoch == train_config.num_epochs - 1:
            # set batch size to 1, evaluate efficiency
            sample_input = torch.randn([1, train_config.window_length, train_config.num_channels],
                                       device=train_config.device)
            flops, params = thop.profile(model=model, inputs=(sample_input,))
            flops, params = thop.clever_format([flops, params], "%.4f")

            start_time = time.time()
            _ = model(sample_input)
            latency = time.time() - start_time

        if epoch % print_gap == 0 and epoch != 0:
            # test for each epoch, test_recon:
            valid_test_out: Test_output = test(test_loader=valid_loader,
                                               model=model,
                                               window_converter=valid_window_converter,
                                               forward_times=train_config.forward_times,
                                               mode="valid")
            valid_recon = valid_test_out.recon_out
            valid_loss = valid_test_out.loss
            valid_sim = valid_test_out.sim_out

            window_threshold = get_threshold(raw_data=raw_train_data,
                                             recon_data=valid_recon,
                                             average_length=train_config.window_length,
                                             ratio=3)

            patch_threshold = get_threshold(raw_data=raw_train_data,
                                            recon_data=valid_recon,
                                            average_length=train_config.patch_length,
                                            ratio=3)

            Test_output_test: Test_output = test(test_loader=test_loader,
                                                 model=model,
                                                 window_converter=test_window_converter,
                                                 forward_times=train_config.forward_times,
                                                 mode="test",
                                                 window_threshold=window_threshold,
                                                 patch_threshold=patch_threshold)

            test_recon = Test_output_test.recon_out
            test_loss = Test_output_test.loss
            test_duration = Test_output_test.duration
            test_sim = Test_output_test.sim_out

            total_valid_loss.append([epoch, valid_loss])

            # evaluate the test result
            if data_name == "UCR":
                average_window_length = int(main_period)
            else:
                average_window_length = None
            anomaly_score_cal = AnomalyScoreCalculator(mode=train_config.anomaly_mode,
                                                       average_window=average_window_length)
            test_anomaly_score = anomaly_score_cal.calculate_anomaly_score(
                raw_train_data=raw_train_data,
                raw_test_data=raw_test_data,
                recon_test_data=test_recon,
                recon_train_data=valid_recon,
            )
            test_anomaly_score_final = test_anomaly_score.test_score_all
            train_anomaly_score_final = test_anomaly_score.train_score_all

            if args.model_name == "PatchContrast":
                sim_anomaly_score = anomaly_score_cal.calculate_anomaly_score(
                    raw_train_data=np.zeros_like(valid_sim),
                    raw_test_data=np.zeros_like(test_sim),
                    recon_train_data=valid_sim,
                    recon_test_data=test_sim
                )
                test_sim_socre = sim_anomaly_score.test_score_all
                test_sim_socre[:test_window_converter.pad_length] = 0
                test_anomaly_score_final = test_anomaly_score.test_score_all + test_sim_socre
                train_anomaly_score_final = test_anomaly_score.train_score_all + sim_anomaly_score.train_score_all

            # save test result
            test_result = evaluate(scores=test_anomaly_score_final, targets=raw_test_labels, pa=True)
            eval_result_save_path = os.path.join(output_dir, f"{data_name}_{group}_test_result_{epoch}.json")
            with open(eval_result_save_path, "w") as file:
                json.dump(test_result.__dict__, file, indent=4)

            # evaluate efficiency
            efficiency_result = EfficiencyResult(test_time=test_duration, flops=flops, params=params,
                                                 average_epoch_time=np.median(duration_list),
                                                 all_training_time=np.sum(duration_list),
                                                 latency=latency)
            efficiency_result = efficiency_result.__dict__
            efficiency_result_save_path = os.path.join(output_dir, f"efficiency_result.json")
            with open(efficiency_result_save_path, "w") as file:
                json.dump(efficiency_result, file, indent=4)

            # save model
            torch.save(model, os.path.join(output_dir, f"model_{epoch}.pth"))
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_state_dict_{epoch}.pth"))

            # save recon and anomaly_score
            np.save(os.path.join(output_dir, f"recon_train_{epoch}.npy"), valid_recon)
            np.save(os.path.join(output_dir, f"recon_test_{epoch}.npy"), test_recon)
            np.save(os.path.join(output_dir, f"test_anomaly_score_{epoch}.npy"), test_anomaly_score_final)
            np.save(os.path.join(output_dir, f"train_anomaly_score_{epoch}.npy"), train_anomaly_score_final)

            tqdm.tqdm.write(f'Epoch {epoch},\t'
                            f'train_loss = {epoch_loss}\t'
                            f'valid_loss = {valid_loss},\t'
                            f'test_loss = {test_loss}')

            if args.plot:
                figure_save_path = os.path.join(output_dir, f"epoch_{epoch}.png")
                recon_plot(
                    save_path=figure_save_path,
                    gap=400,
                    figure_length=args.figure_length,
                    figure_width=args.figure_width,
                    font_size=4,
                    test_data=raw_test_data,
                    test_label=raw_test_labels,
                    train_label=raw_train_labels,
                    test_anomaly_score=test_anomaly_score_final,
                    train_anomaly_score=train_anomaly_score_final,
                    train_data=raw_train_data,
                    recon_test_data=test_recon,
                    recon_train_data=valid_recon,
                    threshold=test_result.best_threshold_wo_pa,
                    plot_diff=True
                )

    # plot loss
    total_train_loss = np.array(total_train_loss)
    total_valid_loss = np.array(total_valid_loss)
    plt.plot(total_train_loss[20:, 0], total_train_loss[20:, 1])
    plt.plot(total_valid_loss[20:, 0], total_valid_loss[20:, 1])
    plt.scatter(total_train_loss[20:, 0], total_train_loss[20:, 1])
    plt.scatter(total_valid_loss[20:, 0], total_valid_loss[20:, 1])
    # plt.ylim(0, 0.05)
    plt.savefig(os.path.join(output_dir, f"loss.png"), format="png", dpi=200)
