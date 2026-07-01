import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from utils import generate_rainbow_gradient

CMAPS = ['Reds', 'Greens', 'Blues']


# -----------------------------
# 构建单色颜色映射（用于 imshow 的 cmap）
# -----------------------------
def make_single_color_cmap(rgb_color):
    R, G, B = rgb_color

    colors = [(R / 2.0, G / 2.0, B / 2.0), rgb_color]  # 白 -> 指定颜色
    return LinearSegmentedColormap.from_list('single_color', colors, N=256)


def visualize(path, rgb_bands=[36, 19, 8], band_step=1, max_bands=3, offset_x=0.1, offset_y=0.1, save_name='cube'):
    # -----------------------------
    # 加载数据
    # -----------------------------
    data = np.load(data_path)  # shape: (H, W, C)
    H, W, C = data.shape

    # 选择部分波段用于可视化
    band_indices = list(range(0, min(C, max_bands), band_step))
    num_bands = len(band_indices)
    # colors = generate_rgb_gradient(max_bands // band_step)
    colors = generate_rainbow_gradient(max_bands // band_step)

    # -----------------------------
    # 创建画布
    # -----------------------------
    fig, axes = plt.subplots(1, 1, figsize=(9, 9),
                             subplot_kw={'aspect': 'equal'})

    # 隐藏坐标轴
    axes.axis('off')

    # 第一层：RGB 图像
    if len(rgb_indices) == 3:
        R_band, G_band, B_band = rgb_indices

        # 提取 RGB 波段
        R = data[:, :, R_band]
        G = data[:, :, G_band]
        B = data[:, :, B_band]

        # 归一化到 [0, 1]
        R_norm = (R - R.min()) / (R.max() - R.min() + 1e-8)
        G_norm = (G - G.min()) / (G.max() - G.min() + 1e-8)
        B_norm = (B - B.min()) / (B.max() - B.min() + 1e-8)

        # 合成 RGB 图像
        rgb_image = np.dstack((R_norm, G_norm, B_norm))

        # 设置位置偏移 [left, right, bottom, top]
        extent = (
            0,
            W / H * 1,  # 宽高比适配
            1,
            0
        )

        # 绘制 RGB 图像
        plt.imshow(rgb_image, extent=extent, alpha=alpha_start, zorder=100001)

    # 后续波段：彩色灰度图
    for i, b in enumerate(band_indices):
        # if i == 0:  # 跳过第一个波段（已绘制 RGB 图像）
        #     continue

        img = data[:, :, b]

        # 归一化到 [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 动态选择颜色映射
        # cmap_index = i % len(cmap_list)  # 循环使用颜色映射列表
        # cmap_name = cmap_list[cmap_index]
        cmap = make_single_color_cmap(colors[i])
        # cmap = CMAPS[i]

        # 设置透明度（渐变效果）
        alpha = alpha_start - (alpha_start - alpha_end) * (i / num_bands)

        # 设置位置偏移
        extent = (
            i * offset_x,
            i * offset_x + W / H * 1,  # 宽高比适配
            i * offset_y + 1,
            i * offset_y
        )

        # 绘制图像层
        plt.imshow(img_norm, cmap=cmap, extent=extent, alpha=alpha, zorder=10000 - i)

    # 设置图形范围
    plt.xlim(0, num_bands * offset_x + W / H)
    plt.ylim(-0.5, 1.5)

    # 设置背景透明
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)

    # 保存为 SVG，仅保留 axes 区域，无背景
    extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f'{save_name}.svg', format='svg', bbox_inches=extent, pad_inches=0, transparent=True)
    # # 保存图像（可选）
    # plt.savefig("cube.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # -----------------------------
    # 参数设置
    # -----------------------------
    data_path = r"E:\datasets\f130804t01p00r06rdn_e_35 (1).npy"  # 替换为你的 .npy 文件路径
    rgb_indices = [36, 19, 8]  # 提供 RGB 波段索引（示例：[R_band, G_band, B_band]）
    _band_step = 1  # 显示每隔几个波段（减少计算量）
    _max_bands = 166  # 最多显示多少个波段（防止太宽）
    alpha_start = 1  # 起始透明度（RGB 层）
    alpha_end = 1  # 结束透明度

    # -----------------------------
    # 绘制每一层图像（带偏移）
    # -----------------------------
    _offset_x = 0.002  # 水平偏移比例（减小间隔）
    _offset_y = 0.002  # 垂直偏移比例（减小间隔）

    visualize(data_path,
              rgb_bands=rgb_indices,
              band_step=_band_step,
              max_bands=_max_bands,
              offset_x=_offset_x,
              offset_y=_offset_y,
              save_name=os.path.basename(data_path))
