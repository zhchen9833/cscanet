import numpy as np
import torch
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from uchiha.utils import get_root_logger


def validate_ddp(epoch, dataloader, model, writer, metric, device, rank, world_size):
    model.eval()
    dataset = dataloader.dataset
    logger = get_root_logger()
    logger.info('start evaluating...')

    # 每个 rank 本地累积
    local_psnr_sum = 0.0
    local_ssim_sum = 0.0
    local_count = 0

    # 如果你需要 by_level
    local_by_level = {lvl: {'psnr': 0.0, 'ssim': 0.0, 'count': 0} for lvl in dataset.NOISE_LEVELS}

    with torch.no_grad():
        for data in dataloader:
            sample = data['sample'].to(device, non_blocking=True)
            pred = model(sample)
            pred = pred.cpu().numpy()
            target = data['target'].numpy()
            index = data['index'].numpy()

            B, C, H, W = pred.shape
            for i in range(B):
                pred_img = pred[i]
                target_img = target[i]
                idx = index[i]
                noise_level = dataset.NOISE_LEVELS[idx % len(dataset.NOISE_LEVELS)]

                # HWC 转换
                if C > 1:
                    pred_img = np.transpose(pred_img, (1, 2, 0))
                    target_img = np.transpose(target_img, (1, 2, 0))
                else:
                    pred_img = pred_img[0]
                    target_img = target_img[0]

                data_range = target_img.max() - target_img.min()
                psnr_val = psnr(target_img, pred_img, data_range=data_range)
                ssim_val = ssim(target_img, pred_img,
                                multichannel=(C > 1),
                                channel_axis=2 if C > 1 else None,
                                data_range=data_range)

                # 累积
                local_psnr_sum += psnr_val
                local_ssim_sum += ssim_val
                local_count += 1

                # by_level
                local_by_level[noise_level]['psnr'] += psnr_val
                local_by_level[noise_level]['ssim'] += ssim_val
                local_by_level[noise_level]['count'] += 1

    # ----------- 使用 all_reduce 聚合标量（极省内存） ----------
    local_tensor = torch.tensor(
        [local_psnr_sum, local_ssim_sum, local_count],
        dtype=torch.float64, device=device
    )
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)

    global_psnr_mean = (local_tensor[0] / local_tensor[2]).item()
    global_ssim_mean = (local_tensor[1] / local_tensor[2]).item()

    if rank == 0:
        logger.info(f"{' Evaluation Results ':=^60}")
        logger.info(f"Mean PSNR: {global_psnr_mean:.2f} dB")
        logger.info(f"Mean SSIM: {global_ssim_mean:.4f}")
        logger.info("=" * 60)
        writer.add_scalar("PSNR", global_psnr_mean, epoch)
        writer.add_scalar("SSIM", global_ssim_mean, epoch)

    return {"psnr": global_psnr_mean, "ssim": global_ssim_mean}
