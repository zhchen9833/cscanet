# train_by_epoch_ddp.py (示例函数)
import torch
from uchiha.utils import print_log, get_root_logger
from torch.profiler import profile, record_function, ProfilerActivity

def train_by_epoch_ddp(cfg, epoch, dataloader, model, optimizer, scheduler, criterion, writer,
                       eta_calculator, device, local_rank=0, rank=0):
    print_freq = cfg.train.print_freq
    total_epoch = cfg.train.total_epoch
    use_grad_clip = cfg.train.use_grad_clip

    model.train()
    for idx, data in enumerate(dataloader):
        # data -> move to local device
        sample = data['sample'].to(device, non_blocking=True)
        target = data['target'].to(device, non_blocking=True)

        # forward & loss
        pred = model(sample)
        loss = criterion(pred, target)

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     for name, param in model.named_parameters():
        #         if param.grad is None:
        #             print("[UNUSED]", name)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        # only rank 0 compute ETA and log
        if rank == 0 and eta_calculator is not None:
            eta = eta_calculator.update()
            if (idx + 1) % print_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print_log(
                    f'epoch:[{epoch + 1}/{total_epoch}]\titer:[{idx + 1}/{len(dataloader)}]\t'
                    f'loss:{loss.item():.6f}\tlr:{current_lr:6e}\teta:{eta_calculator.format_eta(eta)}',
                    get_root_logger())

            if writer is not None:
                writer.add_scalar('loss', loss.item(), epoch * len(dataloader) + idx)

    scheduler.step()
    return writer, model, optimizer, scheduler
