from toolkit.load_dataset import load_explain_dataset
from toolkit.load_config_data_model import get_dataloader
import argparse
import torch
import os
import sys
from test import test, get_threshold
from toolkit.explainability import explainable_scores
import json
import numpy as np
from toolkit.result_plot import recon_plot
from toolkit.load_config_data_model import find_length

sys.path.append("other_models")
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="MAUT", help='model name')
parser.add_argument('--group', type=str, default="real_satellite_data_2", help='group number')
parser.add_argument('--data_name', type=str, default='synthetic', help='dataset name')
parser.add_argument('--figure_length', type=int, default=60)
parser.add_argument('--figure_width', type=int, default=20)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model_name = args.model_name
    group = args.group
    data_name = args.data_name
    train_data, test_data, raw_data, explain_labels = load_explain_dataset(data_name=data_name, group=group)

    output_dir = os.path.join("explain_results", f"{model_name}", f"{data_name}")

    if len(train_data.shape) == 1:
        train_data = train_data[:, np.newaxis]

    print(f"\n Model: {model_name}, Data: {data_name}, Group: {group},")

    if data_name == "synthetic":
        num_channels = 5
        group = 0
        main_period = 10
    elif data_name == "sate":
        num_channels = 9
        main_period = find_length(train_data[:, 0])
    elif data_name == "ASD":
        num_channels = 19
        main_period = find_length(train_data[:,11])
    elif data_name == "UCR":
        num_channels = 1
        args.figure_length, args.figure_width = 160, 20
        main_period = find_length(train_data[:, 0])
    else:
        raise NotImplementedError("Invalid data name")

    if model_name == "PatchContrast":
        # model save path need to be modified
        # model_path = 'output\\PatchContrast\\real_satellite_data_1\\window_len_512-d_model_64-patch_len_16_remove_anomaly_True-mode_common_channel-04-02-12-35'
        if data_name == "ASD":
            model_dir = os.path.join("output", "PatchContrast", "ASD_output", f"ASD_{group}")
            model_path = os.listdir(model_dir)[0]
            model_path = os.path.join(model_dir, model_path)
            model = torch.load(os.path.join(model_path, "model_100.pth")).to(device)
        elif data_name == "UCR":
            model_dir = os.path.join("output", "PatchContrast", "UCR", f"UCR_{group}")
            model_path = os.listdir(model_dir)[0]
            model_path = os.path.join(model_dir, model_path)
            model = torch.load(os.path.join(model_path, "model_200.pth")).to(device)
        elif data_name == "synthetic":
            model_path = "output/PatchContrast/synthetic/window_len_1024-d_model_64-patch_len_16_remove_anomaly_True-mode_common_channel-03-26-10-29"
            model = torch.load(os.path.join(model_path, "model_100.pth")).to(device)
        elif data_name == "sate":
            if group == "real_satellite_data_1":
                model_path = "output/PatchContrast/real_satellite_data_1/window_len_512-d_model_64-patch_len_16_remove_anomaly_True-mode_common_channel-04-02-12-35"
            elif group == "real_satellite_data_2":
                model_path = "output/PatchContrast/real_satellite_data_2/window_len_512-d_model_64-patch_len_16_remove_anomaly_True-mode_common_channel-04-02-12-38"
            else:
                raise ValueError("Invalid group number")
            model = torch.load(os.path.join(model_path, "model_100.pth")).to(device)

        val_loader, val_window_converter = get_dataloader(train_data,
                                                          batch_size=64,
                                                          window_length=model.patcher.sequence_length,
                                                          window_stride=model.patcher.sequence_length,
                                                          mode="test")

        test_loader, test_window_converter = get_dataloader(test_data,
                                                            batch_size=64,
                                                            window_length=model.patcher.sequence_length,
                                                            window_stride=model.patcher.sequence_length,
                                                            mode="test")
        val_recon = test(test_loader=val_loader,
                         model=model,
                         window_converter=val_window_converter,
                         forward_times=5,
                         mode="valid").recon_out

        window_threshold = get_threshold(raw_data=train_data,
                                         recon_data=val_recon,
                                         average_length=model.patcher.sequence_length,
                                         ratio=5)

        patch_threshold = get_threshold(raw_data=train_data,
                                        recon_data=val_recon,
                                        average_length=model.patcher.patch_length,
                                        ratio=5)

        recon_test = test(test_loader=test_loader,
                          model=model,
                          window_converter=test_window_converter,
                          forward_times=5,
                          mode="valid",
                          window_threshold=window_threshold,
                          patch_threshold=patch_threshold).recon_out

    elif model_name == "cad":
        model_path = os.path.join("output", model_name, data_name, "normal_0", f"{data_name}_{group}",
                                  "window_length_20")
        model = torch.load(os.path.join(model_path, "model.pth")).to(device)
        window_length = 20
        window_pad = np.ones([window_length - 1, num_channels])
        new_test_data = np.concatenate([window_pad, test_data], axis=0)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=64,
                                        window_length=window_length,
                                        window_stride=1,
                                        mode="test")
        recon_test = model.predict(test_loader)
        if len(recon_test.shape) == 1:
            recon_test = recon_test[:, np.newaxis]
    elif model_name == "mtad_gat":
        model_path = os.path.join("output", model_name, data_name, "normal_0", f"{data_name}_{group}",
                                  "window_length_100")
        window_length = 100
        model = torch.load(os.path.join(model_path, "model.pth")).to(device)

        window_pad = np.ones([window_length - 1, num_channels])
        new_test_data = np.concatenate([window_pad, test_data], axis=0)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=64,
                                        window_length=100,
                                        window_stride=1,
                                        mode="test",
                                        if_shuffle=False)

        recon_test = model.predict(test_loader)
        recon_test = (recon_test[0] + recon_test[1]) / 2
    elif model_name == "TranAD":
        model_path = os.path.join("output", model_name, data_name, "normal_0", f"{data_name}_{group}",
                                  "window_length_10")
        model = torch.load(os.path.join(model_path, "model.pth")).to(device)
        window_length = 10
        window_pad = np.ones([window_length - 1, num_channels])
        new_test_data = np.concatenate([window_pad, test_data], axis=0)
        test_loader, _ = get_dataloader(new_test_data,
                                        batch_size=64,
                                        window_length=window_length,
                                        window_stride=1,
                                        mode="test")
        recon_test = model.predict(test_loader)
    elif model_name == "MAUT":
        model_path = os.path.join("output", model_name, data_name, "normal_0", f"{data_name}_{group}",
                                  "window_length_100")
        model = torch.load(os.path.join(model_path, "model.pth")).to(device)
        test_loader, test_window_converter = get_dataloader(test_data,
                                                            batch_size=64,
                                                            window_length=100,
                                                            window_stride=100,
                                                            mode="test")
        recon_test = model.predict(test_loader)
        recon_test = test_window_converter.windows_to_sequence(recon_test)
    else:
        raise ValueError("Invalid model name")

    explain_scores = explainable_scores(target=raw_data, reconstructed=recon_test,
                                        train_data=train_data, labels=explain_labels, main_period=main_period)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join("explain_results", f"{model_name}", f"{data_name}", f"group_{group}.json")
    with open(save_path, "w") as f:
        json.dump(explain_scores, f, indent=4)

    figure_save_path = os.path.join(output_dir, f"results_group_{group}.png")
    recon_plot(
        save_path=figure_save_path,
        gap=400,
        figure_length=args.figure_length,
        figure_width=args.figure_width,
        font_size=4,
        test_data=raw_data,
        test_label=explain_labels,
        train_data=test_data,
        recon_test_data=recon_test,
        plot_diff=False
    )

    # save raw_data, test_data, recon_test_data, explain_labels
    np.save(os.path.join(output_dir, f"raw_data_group_{group}.npy"), raw_data)
    np.save(os.path.join(output_dir, f"test_data_group_{group}.npy"), test_data)
    np.save(os.path.join(output_dir, f"recon_test_data_group_{group}.npy"), recon_test)
    np.save(os.path.join(output_dir, f"explain_labels_group_{group}.npy"), explain_labels)
