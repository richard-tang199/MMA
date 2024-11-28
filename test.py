from torch.utils.data import DataLoader
from model.patch_detector import PatchDetector, PatchDetectorOutput, PatchDetectorContrastOutput
import numpy as np
from toolkit.load_dataset import SequenceWindowConversion
import torch
import time
from dataclasses import dataclass


@dataclass
class Test_output:
    recon_out: np.ndarray
    loss: float
    sim_out: np.ndarray
    duration: float


def sum_every_length(arr: np.ndarray, length: int):
    result = np.add.reduceat(arr, np.arange(0, len(arr), length))
    if len(arr) % length != 0:
        result = result[:-1]
    result = result / length

    return result


def get_threshold(raw_data: np.ndarray,
                  recon_data: np.ndarray,
                  average_length: int,
                  ratio: int = 3):
    """
    @param average_length: window_length or patch_size
    @param raw_data: sequence_len, num_channels
    @param recon_data: sequence_len, num_channels
    @return:
    """
    # TODO: change to mean
    diff = np.abs(raw_data - recon_data).mean(axis=1)
    average_length_diff = sum_every_length(diff, average_length)
    average_length_mean = average_length_diff.mean()
    std_length_diff = average_length_diff.std()
    threshold = average_length_mean + ratio * std_length_diff
    return threshold


def test(model: PatchDetector,
         test_loader: DataLoader,
         window_converter: SequenceWindowConversion,
         forward_times: int = 5,
         mode: str = "valid",
         window_threshold: float = 0.0,
         patch_threshold: float = 0.0,
         device: str = "cuda"):
    """
    @param test_loader:
    @param model:
    @param stride_size:
    @param window_size:
    @param forward_times:
    @return: recon_out: num_channels X total_sequence_length
    """
    model.eval()
    all_loss = 0.0
    recon_out = None
    sim_out = None
    start_time = time.time()
    with torch.no_grad():
        for (test_data,) in test_loader:
            test_data = test_data.to(device)
            test_loss = torch.zeros(1, 1, device=device)
            output = torch.zeros_like(test_data, device=device)
            sim_score = torch.zeros_like(test_data, device=device)

            for i in range(forward_times):
                test_output: PatchDetectorContrastOutput = model(test_data,
                                                                 mode=mode,
                                                                 window_threshold=window_threshold,
                                                                 patch_threshold=patch_threshold)
                # output: (batch_size, num_channels, sequence_length)
                test_loss += test_output.loss
                output += test_output.over_output
                if test_output.sim_score is not None:
                    sim_score += test_output.sim_score

            test_loss = test_loss / forward_times
            output = output / forward_times

            if test_output.sim_score is not None:
                sim_score = sim_score / forward_times
                if sim_out is None:
                    sim_out = sim_score
                else:
                    sim_out = torch.concat([sim_out, sim_score], dim=0)

            all_loss += test_loss.item()
            if recon_out is None:
                recon_out = output
            else:
                recon_out = torch.concat([recon_out, output], dim=0)

        duration = time.time() - start_time
        recon_out = recon_out.detach().cpu().numpy()
        recon_out = window_converter.windows_to_sequence(recon_out)

        if sim_out is not None:
            sim_out = sim_out.detach().cpu().numpy()
            sim_out = window_converter.windows_to_sequence(sim_out)

        return Test_output(
            recon_out=recon_out,
            loss=all_loss / len(test_loader),
            sim_out=sim_out,
            duration=duration
        )
