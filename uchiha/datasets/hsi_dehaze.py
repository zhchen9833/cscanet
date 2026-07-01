import glob
import os
from functools import partial
from typing import List

import numpy as np
import tifffile as tiff
import torch
from scipy import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset

from .builder import DATASET
from .pipelines import Compose
from ..utils import get_root_logger
from ..utils.metrics import calculate_sam, calculate_uqi


# ================= band score and exchange function ===================
def piecewise_reflectance_penalty(x, lower=0.01, upper=2.0):
    if lower >= upper:
        raise ValueError(f'lower threshold must be smaller than upper threshold, got {lower} >= {upper}')

    return torch.where(
        x < lower,
        0.1 * (x + upper - lower),
        torch.where(
            x > upper,
            0.1 * (x - lower),
            0.01 * (x - lower),
        ),
    )


def smooth_reflectance_penalty(x, lower=0.01, upper=2.0, tau_ratio=0.05):
    if lower >= upper:
        raise ValueError(f'lower threshold must be smaller than upper threshold, got {lower} >= {upper}')
    tau = tau_ratio * (upper - lower)
    return (
        0.01 * (x - lower)
        + (0.09 * x + 0.1 * upper - 0.09 * lower) * torch.sigmoid((lower - x) / tau)
        + 0.09 * (x - lower) * torch.sigmoid((x - upper) / tau)
    )


def compute_scores(band_data, wavelengths, a=0.01, b=2, score_mode='legacy',
                   lower=0.01, upper=2.0, tau_ratio=0.05):
    # band_data: [B, C, H, W], wavelengths: [C]
    spatial_std = band_data.std(dim=[2, 3])  # [B, C]
    wave_term = a * torch.exp(-b * (wavelengths.unsqueeze(0) - 400) / (2500 - 400))  # [1, C]
    scores = spatial_std + wave_term.to(spatial_std.device)  # [B, C]

    if score_mode == 'legacy':
        return scores
    if score_mode == 'piecewise':
        band_mean = band_data.mean(dim=[2, 3])
        penalty = piecewise_reflectance_penalty(band_mean, lower=lower, upper=upper)
        return scores - penalty.to(scores.device)
    if score_mode == 'smooth':
        band_mean = band_data.mean(dim=[2, 3])
        penalty = smooth_reflectance_penalty(
            band_mean, lower=lower, upper=upper, tau_ratio=tau_ratio)
        return scores - penalty.to(scores.device)

    raise ValueError(f'Invalid band score mode: {score_mode}')


def get_topk_indices(scores, ratio=0.1, largest=True):
    if ratio <= 0 or ratio > 1:
        raise ValueError(f'exchange ratio must be in (0, 1], got {ratio}')
    k = max(1, int(scores.size(1) * ratio))
    values, indices = torch.topk(scores, k=k, dim=1, largest=largest)
    return indices


def exchange_band_values(group_a, group_b, indices_a, indices_b):
    group_a_new = group_a.clone()
    group_b_new = group_b.clone()
    for b in range(group_a.size(0)):
        a_idx = indices_a[b]
        b_idx = indices_b[b]
        temp = group_a[b, a_idx].clone()
        group_a_new[b, a_idx] = group_b[b, b_idx]
        group_b_new[b, b_idx] = temp
    return group_a_new, group_b_new


def _as_chw_tensor(hsi, channel_hint=305):
    tensor = hsi if torch.is_tensor(hsi) else torch.as_tensor(hsi)
    if tensor.ndim != 3:
        raise ValueError(f'Expected a 3D HSI tensor, got shape {tuple(tensor.shape)}')

    if channel_hint is not None:
        if tensor.shape[-1] == channel_hint and tensor.shape[0] != channel_hint:
            return tensor.permute(2, 0, 1).contiguous(), 'HWC'
        if tensor.shape[0] == channel_hint and tensor.shape[-1] != channel_hint:
            return tensor.contiguous(), 'CHW'
    if tensor.shape[-1] > tensor.shape[0] and tensor.shape[-1] > tensor.shape[1]:
        return tensor.permute(2, 0, 1).contiguous(), 'HWC'
    return tensor.contiguous(), 'CHW'


def _restore_hsi_layout(hsi_chw, layout):
    if layout == 'HWC':
        return hsi_chw.permute(1, 2, 0).contiguous()
    return hsi_chw


def compute_band_exchange_indices(hsi, first_branch_channel=102, a=0.01, b=2,
                                  score_mode='legacy', lower=0.01, upper=2.0,
                                  tau_ratio=0.05, exchange_ratio=0.1):
    hsi, _ = _as_chw_tensor(hsi, channel_hint=first_branch_channel * 3 - 1)
    hsi = hsi.unsqueeze(0)
    C = hsi.shape[1]
    f = first_branch_channel
    device = hsi.device

    group1 = hsi[:, 0:f]
    group2 = hsi[:, f:2 * f]
    group3 = hsi[:, 2 * f:]

    wavelengths = torch.linspace(400, 2500, C, device=device)
    w1 = wavelengths[0:f]
    w2 = wavelengths[f:2 * f]
    w3 = wavelengths[2 * f:]

    s1 = compute_scores(group1, w1, a, b, score_mode=score_mode,
                        lower=lower, upper=upper, tau_ratio=tau_ratio)
    s2 = compute_scores(group2, w2, a, b, score_mode=score_mode,
                        lower=lower, upper=upper, tau_ratio=tau_ratio)
    s3 = compute_scores(group3, w3, a, b, score_mode=score_mode,
                        lower=lower, upper=upper, tau_ratio=tau_ratio)

    top1 = get_topk_indices(s1, exchange_ratio, largest=True)
    low2 = get_topk_indices(s2, exchange_ratio, largest=False)
    top2 = get_topk_indices(s2, exchange_ratio, largest=True)
    low3 = get_topk_indices(s3, exchange_ratio, largest=False)

    return top1, low2, top2, low3


def apply_band_exchange_with_indices(hsi, indices, first_branch_channel=102):
    input_is_tensor = torch.is_tensor(hsi)
    hsi, layout = _as_chw_tensor(hsi, channel_hint=first_branch_channel * 3 - 1)
    hsi = hsi.unsqueeze(0)
    f = first_branch_channel

    group1 = hsi[:, 0:f]
    group2 = hsi[:, f:2 * f]
    group3 = hsi[:, 2 * f:]

    top1, low2, top2, low3 = indices

    group1, group2 = exchange_band_values(group1, group2, top1, low2)
    group2, group3 = exchange_band_values(group2, group3, top2, low3)

    out = torch.cat([group1, group2, group3], dim=1).squeeze(0)
    out = _restore_hsi_layout(out, layout)
    return out if input_is_tensor else out.cpu().numpy()


def get_clean_of_hd(haze_path, dataset_name):
    clean_path = haze_path.replace("haze", "clean")
    if dataset_name == 'HDD':
        return clean_path
    clean_path = list(clean_path)
    len1 = len(clean_path)
    if clean_path[len1 - 7] == "_":
        clean_path[len1 - 7:len1 - 4] = ""
    else:
        clean_path[len1 - 6:len1 - 4] = ""
    clean_path = "".join(clean_path)
    return clean_path


def get_clean_of_hyperhazeoff(haze_path, dataset_name):
    if dataset_name == 'HyperHazeOffSyn':
        split_dir = os.path.dirname(haze_path)
        root_dir = os.path.dirname(split_dir)
        clean_id = os.path.splitext(os.path.basename(haze_path))[0].split('_')[0]
        return os.path.join(root_dir, 'clear', f'{clean_id}.npy').replace('\\', '/')

    if dataset_name == 'HyperHazeOffReal':
        same_stem_clean = haze_path.replace('_hazed.npy', '_clean.npy')
        if same_stem_clean != haze_path:
            clean_candidates = sorted(glob.glob(os.path.join(os.path.dirname(haze_path), '*_clean.npy')))
            if clean_candidates and not os.path.exists(same_stem_clean):
                return clean_candidates[0].replace('\\', '/')
            return same_stem_clean.replace('\\', '/')

    raise ValueError(f'Invalid HyperHazeOff dataset name: {dataset_name}')


def _to_hwc_array(array, num_bands):
    array = np.asanyarray(array, dtype="float32")
    if array.ndim != 3:
        raise ValueError(f'Expected a 3D HSI array, got shape {array.shape}')

    if num_bands is not None:
        if array.shape[-1] == num_bands:
            return array
        if array.shape[0] == num_bands:
            return np.transpose(array, (1, 2, 0))

    if array.shape[-1] <= array.shape[0] and array.shape[-1] <= array.shape[1]:
        return array
    if array.shape[0] <= array.shape[1] and array.shape[0] <= array.shape[2]:
        return np.transpose(array, (1, 2, 0))

    raise ValueError(f'Cannot infer HSI channel axis for shape {array.shape}')


def _normalize_hsi_array(array, scale_factor=None, normalize_mode='auto'):
    array = np.nan_to_num(array.astype("float32"), copy=False)
    if normalize_mode in (None, 'none'):
        return array

    if scale_factor is not None:
        return array / float(scale_factor)

    if normalize_mode != 'auto':
        raise ValueError(f'Invalid normalize mode: {normalize_mode}')

    max_value = float(np.max(array)) if array.size else 0.0
    if max_value <= 1.5:
        return array
    if max_value <= 255.0:
        return array / 255.0
    if max_value <= 10000.0:
        return array / 10000.0
    return array / 65535.0


# ================= dataset definition ===================
def get_loader(loader_type, path):
    if loader_type == 'tiff':
        return tiff.imread(path)
    elif loader_type == 'mat':
        return io.loadmat(path)
    elif loader_type == 'npy':
        return np.load(path)
    else:
        raise ValueError('Invalid loader type')


def load_hd(loader, path, dataset_name='HD', exchange_bands=False, first_branch_channel=102,
            band_score_mode='legacy', score_lower=0.01, score_upper=2.0,
            smooth_tau_ratio=0.05, exchange_ratio=0.1):
    im_data = loader(path)
    clean_path = get_clean_of_hd(path, dataset_name)

    if dataset_name == 'HD':
        im_data = np.asanyarray(im_data, dtype="float32") / 2200
    else:
        im_data = np.asanyarray(im_data, dtype="float32")
        im_data = im_data[:305, :, :]
        im_data = im_data / 2200
    # im_data = torch.Tensor(im_data).permute(2, 0, 1)

    im_label = loader(clean_path)
    im_label = np.asanyarray(im_label, dtype="float32") / 2200
    im_label = np.transpose(im_label, (1, 2, 0))

    # im_label = torch.Tensor(im_label)

    if exchange_bands:
        indices = compute_band_exchange_indices(
            im_data,
            first_branch_channel=first_branch_channel,
            score_mode=band_score_mode,
            lower=score_lower,
            upper=score_upper,
            tau_ratio=smooth_tau_ratio,
            exchange_ratio=exchange_ratio,
        )
        im_data = apply_band_exchange_with_indices(im_data, indices, first_branch_channel=first_branch_channel)
        im_label = apply_band_exchange_with_indices(im_label, indices,
                                                    first_branch_channel=first_branch_channel)
    return im_data, im_label


def load_mat(loader, gt_path, lq_path):
    im_data = loader(lq_path)
    data_key = list(im_data.keys())[3]
    im_data = im_data[data_key]
    im_data = np.array(im_data)
    im_data = torch.Tensor(im_data)

    im_label = loader(gt_path)
    label_key = list(im_label.keys())[3]
    im_label = im_label[label_key]
    im_label = np.array(im_label)
    im_label = torch.Tensor(im_label)

    #######
    im_data = im_data.permute(2, 0, 1)
    im_label = im_label.permute(2, 0, 1)
    return im_data, im_label


def load_hyperhazeoff(loader, path, dataset_name='HyperHazeOffSyn', num_bands=182,
                      scale_factor=None, normalize_mode='auto'):
    im_data = _to_hwc_array(loader(path), num_bands)
    clean_path = get_clean_of_hyperhazeoff(path, dataset_name)
    im_label = _to_hwc_array(loader(clean_path), num_bands)

    im_data = _normalize_hsi_array(im_data, scale_factor=scale_factor, normalize_mode=normalize_mode)
    im_label = _normalize_hsi_array(im_label, scale_factor=scale_factor, normalize_mode=normalize_mode)
    return im_data, im_label


@DATASET.register_module()
class HSIDehazeDataset(Dataset):
    """Multi-level noise HSI data set based on HDF5 storage

    Args:
        gt_path: gt data directory
        lq_path: lq data directory
        loader_type: type of loader
        dataset_name: name of the dataset
        pipelines: data preprocessing pipeline
    """
    def __init__(self, gt_path=None, lq_path=None, loader_type='tiff', dataset_name='HD', pipelines=None,
                 exchange_bands=False, first_branch_channel=102, band_score_mode='legacy',
                 score_lower=0.01, score_upper=2.0, smooth_tau_ratio=0.05,
                 exchange_ratio=0.1, num_bands=305, scale_factor=None, normalize_mode='auto'):
        super(HSIDehazeDataset, self).__init__()
        self.loader = partial(get_loader, loader_type)
        self.gt_path = sorted(glob.glob(gt_path)) if gt_path else []
        self.lq_path = sorted(glob.glob(lq_path)) if lq_path else []
        self.dataset_name = dataset_name
        self.pipelines = Compose(pipelines) if pipelines else None
        self.exchange_bands = exchange_bands
        self.first_branch_channel = first_branch_channel
        self.band_score_mode = band_score_mode
        self.score_lower = score_lower
        self.score_upper = score_upper
        self.smooth_tau_ratio = smooth_tau_ratio
        self.exchange_ratio = exchange_ratio
        self.num_bands = num_bands
        self.scale_factor = scale_factor
        self.normalize_mode = normalize_mode

    def __getitem__(self, index):
        if self.dataset_name == 'HD' or self.dataset_name == 'HDD':
            lq, gt = load_hd(
                self.loader,
                self.lq_path[index],
                self.dataset_name,
                exchange_bands=self.exchange_bands,
                first_branch_channel=self.first_branch_channel,
                band_score_mode=self.band_score_mode,
                score_lower=self.score_lower,
                score_upper=self.score_upper,
                smooth_tau_ratio=self.smooth_tau_ratio,
                exchange_ratio=self.exchange_ratio,
            )
        elif self.dataset_name == 'AVIRIS':
            lq, gt = load_mat(self.loader, self.gt_path[index], self.lq_path[index])
        elif self.dataset_name == 'UAV':
            lq, gt = load_mat(self.loader, self.gt_path[index], self.lq_path[index])
        elif self.dataset_name in ('HyperHazeOffSyn', 'HyperHazeOffReal'):
            lq, gt = load_hyperhazeoff(
                self.loader,
                self.lq_path[index],
                dataset_name=self.dataset_name,
                num_bands=self.num_bands,
                scale_factor=self.scale_factor,
                normalize_mode=self.normalize_mode,
            )
        else:
            raise ValueError('Invalid dataset name')

        results = {
            'sample': lq,
            'target': gt,
            'index': index,
        }

        return self.pipelines(results) if self.pipelines else results

    def __len__(self):
        return len(self.lq_path)

    def evaluate(self, preds: List[np.ndarray], targets: List[np.ndarray], metric,
                 indexes: List[int]) -> dict:
        """
        Calculates the similarity between the predicted data and the real data. PSNR and SSIM are supported by default.

        Args:
            preds: Model output list, each element shape is (B, C, H, W)
            targets: A list of real labels, each element of shape (B, C, H, W)
            metric: metric to be calculated. Default value: ['PSNR','SSIM']
            indexes: raw data index list

        Returns:
            dict: dictionary with average psnr and ssim
        """
        assert len(preds) == len(targets) == len(indexes), "input list length must be the same"

        logger = get_root_logger()
        logger.info('start evaluating...')

        psnrs = []
        ssims = []
        uqis = []
        sams = []

        for pred_batch, target_batch, index_batch in zip(preds, targets, indexes):
            B, C, H, W = pred_batch.shape

            for i in range(B):
                pred = pred_batch[i]  # (C,H,W)
                target = target_batch[i]

                #convert to (H W C) format
                if C == 1:
                    pred_img = pred[0]
                    target_img = target[0]
                else:
                    pred_img = np.transpose(pred, (1, 2, 0))
                    target_img = np.transpose(target, (1, 2, 0))

                # calculate psnr
                data_range = target_img.max() - target_img.min()
                psnr_val = psnr(target_img, pred_img, data_range=data_range)

                # calculate ssim
                multichannel = C > 1
                ssim_val = ssim(
                    target_img, pred_img,
                    multichannel=multichannel,
                    channel_axis=2 if multichannel else None,
                    data_range=data_range
                )

                metric_target = target_img if C > 1 else target_img[:, :, None]
                metric_pred = pred_img if C > 1 else pred_img[:, :, None]

                psnrs.append(psnr_val)
                ssims.append(ssim_val)
                uqis.append(calculate_uqi(metric_target, metric_pred))
                sams.append(calculate_sam(metric_target, metric_pred))

        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        mean_uqi = np.mean(uqis)
        mean_sam = np.mean(sams)
        logger.info(f"Mean PSNR: {mean_psnr:.2f} dB")
        logger.info(f"Mean SSIM: {mean_ssim:.4f}")
        logger.info(f"Mean UQI: {mean_uqi:.4f}")
        logger.info(f"Mean SAM (°): {mean_sam:.4f}")

        return {
            'psnr': mean_psnr,
            'ssim': mean_ssim,
            'uqi': mean_uqi,
            'sam': mean_sam,
        }
