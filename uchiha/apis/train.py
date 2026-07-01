import random

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from ..utils import print_log, get_root_logger


def train_by_epoch(cfg, epoch, dataloader, model, optimizer, scheduler, criterion, writer,
                   eta_calculator, device):
    """ train for one epoch

    Prints logs based on the configured frequency (based on the number of iterations)

    Args:
        cfg (class): Config class
        epoch (int): the number of epoch trained
        dataloader (torch.utils.data.Dataloader): training set's dataloader
        model (torch.nn.Module): model built from configuration file
        optimizer (class): optimizer built from configuration file
        scheduler (class): lr scheduler built from configuration file
        criterion (class): loss function built from configuration file
        writer (SummaryWriter): tensorboard-based loggers currently support tensorboardX
        eta_calculator (class): ETA (Estimated Time) Calculator
        device (torch.device): device to run the model

    Returns:
        writer (dict): The updated logger, also return the updated model, optimizer and scheduler.

    """
    print_freq = cfg.train.print_freq
    total_epoch = cfg.train.total_epoch
    use_grad_clip = cfg.train.use_grad_clip

    model.train()
    for idx, data in enumerate(dataloader):
        # data
        sample = data['sample'].to(device, non_blocking=True)
        target = data['target'].to(device, non_blocking=True)

        # forward & loss
        pred = model(sample)
        loss = criterion(pred, target)

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        eta = eta_calculator.update()

        # log
        if (idx + 1) % print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print_log(
                f'epoch:[{epoch + 1}/{total_epoch}]\titer:[{idx + 1}/{len(dataloader)}]\tloss:{loss:.6f}\t'
                f'lr:{current_lr:6e}\teta:{eta_calculator.format_eta(eta)}',
                get_root_logger())

        writer.add_scalar('loss', loss.item(), epoch * len(dataloader) + idx)

    # 在你的训练循环开始前，初始化profiler
    # with profile(
    #         activities=[
    #             ProfilerActivity.CPU,
    #             ProfilerActivity.CUDA,  # 如果使用GPU
    #         ],
    #         schedule=torch.profiler.schedule(
    #             wait=1,  # 跳过前1个step
    #             warmup=1,  # 预热1个step（不记录）
    #             active=3,  # 记录接下来的3个step
    #             repeat=1  # 只重复1轮
    #         ),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/DRS'),  # 保存文件供TensorBoard使用
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=False,  # 可以查看调用栈，但会慢一些
    # ) as prof:
    #     for idx, data in enumerate(dataloader):
    #         if idx >= (1 + 1 + 3):  # 对应 wait + warmup + active
    #             break
    #         # data
    #         sample = data['sample'].to(device, non_blocking=True)
    #         target = data['target'].to(device, non_blocking=True)
    #
    #         # forward & loss
    #         with record_function("forward_pass"):
    #             pred = model(sample)
    #             loss = criterion(pred, target)
    #
    #         # backward & optimize
    #         with record_function("backward_pass"):
    #             optimizer.zero_grad()
    #             loss.backward()
    #             if use_grad_clip:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    #             optimizer.step()
    #         # 告诉profiler一个step结束了
    #         prof.step()
    #
    # # 在控制台打印摘要
    # print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
    #                                 row_limit=20))

    scheduler.step()

    return writer, model, optimizer, scheduler


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
