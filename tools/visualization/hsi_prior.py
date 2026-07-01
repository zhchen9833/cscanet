import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter


def get_dcp(hyperspectral_data, window_size=3):
    """
    计算暗通道先验图（DCP）。
    每个像素取所有通道的最小值，再进行局部最小值滤波。
    """
    min_channels = np.min(hyperspectral_data, axis=2)  # H×W
    dcp_image = minimum_filter(min_channels, size=window_size, mode='reflect')
    dcp_image = (dcp_image - dcp_image.min()) / (dcp_image.max() - dcp_image.min()) * 255
    return dcp_image.astype(np.uint8)


def get_mean_channel(hyperspectral_data):
    """
    计算高光谱数据的均值通道（每个像素点所有通道的平均值）。

    参数:
        hyperspectral_data: H×W×C 的高光谱数据 (np.ndarray)

    返回:
        mean_image: 单通道均值图 (H×W)
    """
    mean_image = np.mean(hyperspectral_data, axis=2)  # H×W
    # 归一化到 [0, 255]
    mean_image = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min()) * 255
    return mean_image.astype(np.uint8)


def get_rcp(hyperspectral_data):
    """
    计算残余通道先验图（RCP）。
    按照通道均值排序（并非对每个像素点都排序）
    """
    H, W, C = hyperspectral_data.shape

    # 步骤1：计算每个通道的全局均值
    channel_means = np.mean(hyperspectral_data.reshape(-1, C), axis=0)  # 形状 [C]

    # 步骤2：按均值排序并分组（低均值组 vs 高均值组）
    sorted_indices = np.argsort(channel_means)  # 从小到大排序
    split_idx = C // 2
    low_group = sorted_indices[:split_idx]  # 低均值组通道索引
    high_group = sorted_indices[split_idx:]  # 高均值组通道索引

    # 步骤3：计算每个像素的组内均值差
    low_values = hyperspectral_data[:, :, low_group]  # 形状 [H, W, split_idx]
    high_values = hyperspectral_data[:, :, high_group]  # 形状 [H, W, C-split_idx]

    # 计算组内均值（沿通道维度）
    low_mean = np.mean(low_values, axis=2)  # 形状 [H, W]
    high_mean = np.mean(high_values, axis=2)  # 形状 [H, W]

    # 计算差值
    rcp_image = high_mean - low_mean  # 形状 [H, W]

    # 步骤4：归一化到 [0, 255]
    rcp_image = (rcp_image - rcp_image.min()) / (rcp_image.max() - rcp_image.min()) * 255
    return rcp_image.astype(np.uint8)


def get_rcp_pixel_wise(image: np.ndarray, split_points: list) -> np.ndarray:
    """
    输入: HWC ndarray
    功能: 排序 + 切分点差值
    输出: HWN ndarray
    """
    h, w, c = image.shape
    flattened = image.reshape(-1, c)
    sorted_flattened = np.sort(flattened, axis=1)

    results = []
    for split_idx in split_points:
        if not (0 < split_idx < c):
            raise ValueError(f"切分点 {split_idx} 超出通道范围 (1~{c - 1})")
        front = sorted_flattened[:, :split_idx]
        back = sorted_flattened[:, split_idx:]
        diff = np.mean(back, axis=1) - np.mean(front, axis=1)
        results.append(diff)

    output = np.stack(results, axis=1)
    return output.reshape(h, w, -1)


def get_haze_transmission(hyperspectral_data, split_band=100):
    """
    获取雾分布（理想状态）。
    雾敏感波段的均值 - 雾不敏感波段的均值

    参数:
        hyperspectral_data: H×W×C 的高光谱数据 (np.ndarray)

    返回:
        mean_image: 单通道均值图 (H×W)
    """
    # 该理论经验证并不使用去雨数据集
    heavy, light = hyperspectral_data[:, :, :split_band], hyperspectral_data[:, :, split_band:]
    heavy = np.mean(heavy, axis=2)
    light = np.mean(light, axis=2)
    t = heavy - light  # H×W
    # 归一化到 [0, 255]
    t = (t - t.min()) / (t.max() - t.min()) * 255
    return t.astype(np.uint8)


def visualize_prior_maps(*args):
    """
    可视化原始数据、DCP和RCP图，并即时显示。
    """
    nums = len(args)
    # RGB
    rgb_image = args[0][:, :, [36, 19, 8]]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # 归一化到 [0,1]

    plt.figure(figsize=(15, 5))

    # 原始RGB图像
    plt.subplot(1, nums, 1)
    plt.imshow(rgb_image)
    plt.title("LQ RGB", fontsize=14)
    plt.axis('off')

    rgb_image = args[1][:, :, [36, 19, 8]]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # 归一化到 [0,1]
    plt.subplot(1, nums, 2)
    plt.imshow(rgb_image)
    plt.title("GT RGB", fontsize=14)
    plt.axis('off')

    for i in range(2, nums):
        plt.subplot(1, nums, i + 1)
        plt.imshow(args[i], cmap='gray')
        plt.colorbar(fraction=0.03, pad=0.04)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prior.png')
    plt.show()


def visualize_channel_wise(image: np.ndarray, prefix: str = "output_channel", threshold: float = 1e-5):
    """
    将 HWN ndarray 的每个通道保存为带透明背景的 PNG
    小于阈值的像素点完全透明
    """
    h, w, n = image.shape

    for i in range(n):
        channel = image[:, :, i]

        # 归一化到 0~1（避免颜色异常）
        norm = (channel - channel.min()) / (channel.ptp() + 1e-8)

        # 创建 RGBA 图层
        cmap = plt.get_cmap('gray')
        rgba = cmap(norm)  # 返回 H x W x 4

        # 将低于阈值的地方设为全透明
        alpha_mask = (norm > threshold).astype(float)
        rgba[..., -1] = alpha_mask  # 设置 alpha 通道

        fig, ax = plt.subplots()
        ax.axis('off')  # 去掉坐标轴
        ax.imshow(rgba)

        filename = f"{prefix}_{i}.png"
        plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved transparent {filename}")


def visualize_rgb(image: np.ndarray, filename: str = "rgb_transparent.png", threshold: float = 1e-5):
    """
    将 HWC (RGB) 图像保存为带透明背景的 PNG。
    只要像素全通道都是 0，就透明；否则保留原色。
    """
    if image.shape[2] != 3:
        raise ValueError("输入图像必须是 3 通道 (RGB)")

    # 归一化到 0~1（若已是 0~255，可换成 /255.0）
    if image.dtype == np.uint8:
        norm_rgb = image.astype(np.float32) / 255.0
    else:
        norm_rgb = np.clip(image, 0, 1)

    # 构造 RGBA
    h, w, _ = norm_rgb.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = norm_rgb

    # 判断是否透明：若该像素点所有通道都小于阈值，则设为透明
    alpha_mask = (np.sum(norm_rgb, axis=2) > threshold).astype(np.float32)
    rgba[..., 3] = alpha_mask

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(rgba)

    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved transparent RGB to {filename}")


if __name__ == '__main__':
    filename = 'f130412t01p00r11rdn_e_48.npy'
    data_path = os.path.join('/home/disk2/ZR/datasets/AVIRIS/512/rain/storm', filename)
    gt_path = os.path.join('/home/disk2/ZR/datasets/AVIRIS/512/gt', filename)
    lq = np.load(data_path)
    gt = np.load(gt_path)
    H, W, C = lq.shape

    sorted_rcp = get_rcp(lq)
    transmission = get_haze_transmission(lq, split_band=190)
    sorted_rcp_pw = get_rcp_pixel_wise(lq, [C // 3, C // 2, 2 * C // 3])

    # 可视化
    # visualize_prior_maps(lq, gt, sorted_rcp, transmission, sorted_rcp_pw[:, :, 1])
    visualize_channel_wise(sorted_rcp_pw)
    visualize_rgb(lq[:, :, [36, 19, 8]], filename='lq.png')
    visualize_rgb(gt[:, :, [36, 19, 8]], filename='gt.png')
