import os
from os import symlink, remove
from os.path import dirname, join, exists, basename

import torch
from torch.optim import Optimizer


def save_checkpoint(model: torch.nn.Module,
                    optimizer: Optimizer,
                    filepath: str,
                    scheduler=None,
                    meta=None) -> None:
    """ save model parameters

    Args:
        model (torch.nn.Module): The model that needs to save the parameters
        optimizer (Optimizer): The optimizer that needs to save the parameters.
        filepath (str): Save path
        scheduler (_LRScheduler): LR Scheduler
        meta (dict): Other meta information to be saved

    """
    if meta is None:
        meta = {}

    file_dir = os.path.dirname(filepath)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    checkpoint = {
        'meta': meta,
        'state_dict': model.state_dict(),
    }

    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save scheduler state dict in the checkpoint
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    # save
    torch.save(checkpoint, filepath)


def load_checkpoint(filename: str,
                    model: torch.nn.Module,
                    optimizer: Optimizer,
                    scheduler=None,) -> dict:
    """

    Args:
        filename (str): Path of model storage
        model (torch.nn.Module): The model with random initialization parameters
        optimizer (Optimizer): The optimizer with random initialization parameters
        scheduler (_LRScheduler): LR Scheduler

    Returns:
        dict: Dict containing model/optimizer parameters and other meta information

    """
    # load
    checkpoint = torch.load(filename, map_location='cpu')

    # meta
    meta = checkpoint['meta']

    # uchiha & load params
    model.load_state_dict(checkpoint['state_dict'])

    # optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # scheduler
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # clean memory
    del checkpoint
    torch.cuda.empty_cache()

    return meta


def auto_resume_helper(output_dir):
    """ find the latest saved model

    Args:
        output_dir (str): Paths for saved models

    Returns:
        str: The path of the latest saved model
    """
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file