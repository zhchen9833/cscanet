import os
from collections import OrderedDict

import imageio
import numpy as np
import tifffile
import torch
from matplotlib import pyplot as plt
from scipy import io

from uchiha.models.builder import build_model
from uchiha.utils.data import normalize


def remove_module_prefix(state_dict):
    """移除权重中的module.前缀"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def load_data(data_path, num_bands=305):
    ext = os.path.splitext(data_path)[1].lower()
    if ext == '.npy':
        data = np.load(data_path)
    elif ext in ('.tif', '.tiff'):
        data = tifffile.imread(data_path)
        if data.shape[0] != 512:
            data = np.transpose(data, (1, 2, 0))
        data = data[:, :, :num_bands]
        data = np.asanyarray(data, dtype="float32")
        # data = data / 2200
        data = data / np.max(data)
        # data = normalize(data)
    elif ext == '.mat':
        full_data = io.loadmat(data_path)
        data_key = list(full_data.keys())[3]
        data = full_data[data_key]
        data = np.asanyarray(data, dtype="float32")
        # data = normalize(data)
    else:
        raise ValueError(f"不支持的格式: {ext}")

    return data


def single_visualize(lq, pred, gt, name, output_dir, result_name, channel_indices=(58, 37, 19)):
    def extract_rgb(data):
        # return np.stack([
        #     np.clip(data[:, :, channel_indices[0]], 0, 1),  # R
        #     np.clip(data[:, :, channel_indices[1]], 0, 1),  # G
        #     np.clip(data[:, :, channel_indices[2]], 0, 1)  # B
        # ], axis=-1)
        return np.stack([
            data[:, :, channel_indices[0]],  # R
            data[:, :, channel_indices[1]],  # G
            data[:, :, channel_indices[2]]  # B
        ], axis=-1)

    lq_rgb = extract_rgb(lq)
    pred_rgb = extract_rgb(pred)
    gt_rgb = extract_rgb(gt)

    # 创建可视化图像
    plt.figure(figsize=(15, 5))

    # 显示低质量数据
    plt.subplot(1, 3, 1)
    plt.imshow(lq_rgb)
    plt.title(f'Low Quality\n{name}')
    plt.axis('off')

    # 显示修复结果
    plt.subplot(1, 3, 2)
    plt.imshow(pred_rgb)
    img_uint8 = (pred_rgb * 255).astype(np.uint8)
    imageio.imwrite(f'{output_dir}/{name}-{result_name}.png',
                    img_uint8)
    plt.title(f'Restored\n{name}')
    plt.axis('off')

    # 显示GT
    plt.subplot(1, 3, 3)
    plt.imshow(gt_rgb)
    plt.title(f'Ground Truth\n{name}')
    plt.axis('off')

    # 调整布局并保存
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'compare_{name}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def single_inference(cfg):
    # model
    model = build_model(cfg.model.to_dict())
    model = model.cuda()

    num_bands = cfg.num_bands
    data = load_data(cfg.sample_path, num_bands)
    gt = load_data(cfg.gt_path, num_bands)

    for ckpt in sorted(os.listdir(cfg.checkpoint)):
        # resume
        ckpt_path = os.path.join(cfg.checkpoint, ckpt)
        param = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(remove_module_prefix(param['state_dict']))

        with torch.no_grad():
            # data
            sample = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).cuda()
            # forward
            pred = model(sample)
            pred = pred.squeeze().permute(1, 2, 0).cpu().numpy()

        pred = normalize(pred)
        # pred = np.clip(pred, 0, 1)
        name = os.path.basename(cfg.sample_path).split('.')[0]
        output_dir = cfg.work_dir
        idx = ckpt.split('.')[0]
        result_name = f'{cfg.save_name}-e{idx}'
        channel_indices = cfg.channel_indices

        single_visualize(lq=data, pred=pred, gt=gt, name=name, output_dir=output_dir, result_name=result_name,
                         channel_indices=channel_indices)
