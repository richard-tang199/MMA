import torch
import numpy as np
from torch.utils.data import DataLoader
from model.patch_detector import PatchDetector, PatchDetectorOutput
import os


def train_one_epoch(current_epoch: int,
                    model: PatchDetector,
                    train_loader: DataLoader,
                    optimizer,
                    save_dir: str,
                    scheduler=None,
                    device="cuda") -> float:
    """
    @param optimizer:
    @param device:
    @param scheduler:
    @param current_epoch:
    @param model:
    @param save_dir:
    @type train_loader: DataLoader
    """
    model.train()
    # running_loss: store current epoch loss
    running_loss = 0.0
    for batch_idx, (train_batch,) in enumerate(train_loader):
        train_batch = train_batch.to(device)
        optimizer.zero_grad()
        train_output = model(train_batch, epoch=current_epoch)
        loss: torch.Tensor = train_output.loss
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # if batch_idx % 50 == 0:
        #     print(f'Train Epoch: {current_epoch} [{batch_idx * len(train_batch)}/{len(train_loader.dataset)} '
        #           f'({100 * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / (batch_idx + 1):.6f}')
        #     running_loss = 0.0

    # print(f'Train Epoch: {current_epoch} \tLoss: {running_loss / len(train_loader):.6f}')
    return running_loss / len(train_loader)
