import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def plot_smooth_spectral_curves(clean_data: np.ndarray,
                                noisy_data: np.ndarray,
                                pixel_x: int,
                                pixel_y: int,
                                smooth_factor: int = 300):
    """
    绘制高光谱数据中指定像素点的平滑光谱曲线对比图

    参数:
        clean_data: 干净数据 (C, H, W)
        noisy_data: 噪声数据 (C, H, W)
        pixel_x: 像素x坐标
        pixel_y: 像素y坐标
        smooth_factor: 平滑插值点数 (默认300)
    """
    # 验证输入数据
    assert clean_data.shape == noisy_data.shape, "输入数据维度必须相同"
    assert len(clean_data.shape) == 3, "输入必须是CHW格式"
    assert 0 <= pixel_x < clean_data.shape[2], "x坐标超出范围"
    assert 0 <= pixel_y < clean_data.shape[1], "y坐标超出范围"

    # 提取指定像素的光谱曲线
    clean_spectrum = clean_data[:, pixel_y, pixel_x]
    noisy_spectrum = noisy_data[:, pixel_y, pixel_x]
    channels = np.arange(clean_data.shape[0])

    # 创建平滑曲线
    def make_smooth_line(x, y):
        x_smooth = np.linspace(x.min(), x.max(), smooth_factor)
        spl = make_interp_spline(x, y, k=3)  # 三次样条插值
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth

    # 生成平滑数据
    x_smooth, clean_smooth = make_smooth_line(channels, clean_spectrum)
    _, noisy_smooth = make_smooth_line(channels, noisy_spectrum)

    # 创建绘图
    plt.figure(figsize=(12, 6))

    # 绘制平滑曲线
    plt.plot(x_smooth, clean_smooth, 'b-', linewidth=2.5,
             label='Clean Data', alpha=0.8)
    plt.plot(x_smooth, noisy_smooth, 'r-', linewidth=2.5,
             label='Noisy Data', alpha=0.8)

    # 标记原始数据点
    plt.scatter(channels, clean_spectrum, c='blue', s=60,
                edgecolors='white', zorder=3, label='Clean Samples')
    plt.scatter(channels, noisy_spectrum, c='red', s=60,
                edgecolors='white', zorder=3, label='Noisy Samples')

    # 添加图饰
    plt.title(f'Spectral Curve Comparison at Pixel ({pixel_x}, {pixel_y})',
              fontsize=14, pad=20)
    plt.xlabel('Channel Index', fontsize=12)
    plt.ylabel('Radiance/Reflectance', fontsize=12)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示差异区域
    plt.fill_between(x_smooth, clean_smooth, noisy_smooth,
                     where=(noisy_smooth > clean_smooth),
                     color='red', alpha=0.15, label='Noise Increase')
    plt.fill_between(x_smooth, clean_smooth, noisy_smooth,
                     where=(noisy_smooth < clean_smooth),
                     color='blue', alpha=0.15, label='Noise Decrease')

    plt.tight_layout()
    plt.savefig("curve.jpg")


def analyze_rain_strength(chw_image, pixel_x, pixel_y):
    channels = chw_image.shape[0]

    # 创建带两个子图的画布
    plt.figure(figsize=(14, 6))

    # ==================== 雨纹强度分析 ====================
    # 对每个通道进行分析
    rain_strengths = []
    channel_indices = np.arange(channels)
    for i in range(channels):
        channel = chw_image[i]

        # 计算图像的梯度幅值（雨纹通常导致高频成分较多）
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        strength = np.mean(gradient_magnitude)

        rain_strengths.append(strength)

    # 绘制雨纹强度曲线 (左子图)
    plt.subplot(1, 2, 1)
    plt.plot(channel_indices, rain_strengths, 'r-o', linewidth=2, markersize=8)
    plt.title('Rain Streak Strength by Channel', fontsize=12)
    plt.xlabel('Channel Index', fontsize=10)
    plt.ylabel('Rain Strength', fontsize=10)
    plt.xticks(channel_indices)
    plt.grid(True, linestyle='--', alpha=0.6)

    # ==================== 像素点通道值分析 ====================
    pixel_values = chw_image[:, pixel_y, pixel_x]

    # 绘制通道值曲线 (右子图)
    plt.subplot(1, 2, 2)
    plt.plot(channel_indices, pixel_values, 'b-s', linewidth=2, markersize=8)
    plt.title(f'Pixel Values at ({pixel_x}, {pixel_y})', fontsize=12)
    plt.xlabel('Channel Index', fontsize=10)
    plt.ylabel('Pixel Value', fontsize=10)
    plt.xticks(channel_indices)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('analysis.jpg')


# 使用示例
if __name__ == "__main__":
    pass
