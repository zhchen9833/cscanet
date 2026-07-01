import time

import numpy as np
import torch
from tqdm import tqdm

from uchiha.utils import get_root_logger
from uchiha.utils.metrics import (
    calculate_ag,
    calculate_brisque,
    calculate_il_niqe,
    calculate_niqe,
    calculate_piqe,
    calculate_psnr_ssim,
    calculate_sam,
    calculate_uqi,
)


def simple_test(dataloader, model, metric):
    """ simple test

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        metric (str): model's performance metrics for specific tasks

    """
    dataset = dataloader.dataset
    targets = []
    preds = []
    indexes = []
    model.eval()

    pbar = tqdm(dataloader, desc="testing", total=len(dataloader))

    with torch.no_grad():
        for idx, data in enumerate(pbar):
            # data
            sample = data['sample'].cuda()
            # forward
            pred = model(sample)

            targets.append(data['target'].numpy())
            indexes.append(data['index'].numpy())
            preds.append(pred.cpu().numpy())

            pbar.set_postfix(iter=f"{idx + 1}/{len(dataloader)}")

    # evaluate
    results = dataset.evaluate(preds, targets, metric, indexes)

    pbar.close()

    return results


def complex_test(dataloader, model, device, no_reference=False, include_no_reference=False):
    """ test: metric speed

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        device (torch.device): device to run the model
        no_reference(bool): only no-reference metrics: NIQE AG BRISQUE PIQE IL-NIQE
        include_no_reference(bool): also report no-reference metrics when full-reference metrics are enabled

    """
    logger = get_root_logger()
    logger.info('start testing...')

    psnrs, ssims, uqis, sams = [], [], [], []
    niqes, ags, brisques, piqes, il_niqes = [], [], [], [], []
    model.eval()

    # 时间统计容器
    total_inference_time = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="testing", total=len(dataloader))

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(pbar):
            # data
            sample = data['sample'].to(device, non_blocking=True)
            target = data['target'].numpy()
            # forward
            batch_size = sample.shape[0]

            # 推理与精确计时
            if 'cuda' in str(device):
                torch.cuda.synchronize()  # 等待数据加载完成

            start_time = time.time()

            # --- 推理开始 ---
            pred = model(sample)
            # --- 推理结束 ---

            if 'cuda' in str(device):
                torch.cuda.synchronize()  # 等待模型计算完成

            end_time = time.time()

            # 累加时间与样本数
            total_inference_time += (end_time - start_time)
            total_samples += batch_size

            pred = pred.cpu().numpy()

            # test
            for i in range(batch_size):
                # 提取单张图像 (C, H, W) -> 转置为 (H, W, C)
                pred_img = pred[i].transpose(1, 2, 0)
                target_img = target[i].transpose(1, 2, 0)

                if no_reference or include_no_reference:
                    niqes.append(calculate_niqe(pred_img))
                    ags.append(calculate_ag(pred_img))
                    brisques.append(calculate_brisque(pred_img))
                    piqes.append(calculate_piqe(pred_img))
                    il_niqes.append(calculate_il_niqe(pred_img))

                if not no_reference:
                    p, s = calculate_psnr_ssim(target_img, pred_img)
                    psnrs.append(p)
                    ssims.append(s)
                    uqis.append(calculate_uqi(target_img, pred_img))
                    sams.append(calculate_sam(target_img, pred_img))

            pbar.set_postfix(iter=f"{idx + 1}/{len(dataloader)}")

    pbar.close()

    # 计算平均推理时间 (秒/张)
    avg_inference_time = total_inference_time / total_samples
    logger.info('--- Evaluation Results ---')
    logger.info(f'Total Samples: {total_samples}')
    logger.info(f'Inference Speed: {avg_inference_time * 1000:.2f} ms/sample')  # 转换为毫秒更直观
    logger.info(f'FPS: {1.0 / avg_inference_time:.2f}')
    
    if psnrs:
        logger.info(f'Mean PSNR: {np.mean(psnrs):.4f} dB')
        logger.info(f'Mean SSIM: {np.mean(ssims):.4f}')
        logger.info(f'Mean UQI: {np.mean(uqis):.4f}')
        logger.info(f'Mean SAM (°): {np.mean(sams):.4f}')
    if niqes:
        logger.info(f'Mean NIQE: {np.mean(niqes):.4f}')
        logger.info(f'Mean AG: {np.mean(ags):.4f}')
        logger.info(f'Mean BRISQUE: {np.mean(brisques):.4f}')
        logger.info(f'Mean PIQE: {np.mean(piqes):.4f}')
        logger.info(f'Mean IL-NIQE: {np.mean(il_niqes):.4f}')
