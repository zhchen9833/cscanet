import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward, DWTForward
from skimage.metrics import peak_signal_noise_ratio as psnr_val

from dwt import get_rgb_data


class DTCWTAnalyzer:
    """DTCWT分析与可视化工具"""

    def __init__(self,
                 img_path: str,
                 scales: int = 1,
                 wave: str = 'qshift_a'):
        """
        Args:
            img_path: 输入图像路径（.npy或.npz格式）
            scales: 分解层数
            wave: 小波类型，'qshift_a'或'qshift_d'
        """
        # 加载数据
        self.img = torch.from_numpy(np.load(img_path)).float().permute(2, 0, 1)  # [C, H, W]
        if self.img.dim() == 3:
            self.img = self.img.unsqueeze(0)  # [B, C, H, W]

        # 初始化变换
        self.dtcwt = DTCWTForward(J=scales, biort='legall', qshift=wave)
        self.dwt = DWTForward(J=scales, wave='haar', mode='zero')

        # 存储结果
        self.dtcwt_low = None
        self.dtcwt_high = None
        self.dwt_ll = None
        self.dwt_high = None

    def transform(self):
        """执行DTCWT和DWT变换"""
        # DTCWT变换（复数结果）
        self.dtcwt_low, self.dtcwt_high = self.dtcwt(self.img)
        img1 = self.img.squeeze().permute(1, 2, 0).numpy()
        img2 = self.dtcwt_low.squeeze().permute(1, 2, 0).numpy()
        psnr_value = psnr_val(img1, img2, data_range=img1.max() - img2.min())
        print(psnr_value)

    def visualize(self):
        """可视化对比DTCWT与DWT结果"""
        if self.dtcwt_low is None:
            self.transform()

        plt.figure(figsize=(18, 9))

        # -------------------------------
        # 原始图像
        # -------------------------------
        plt.subplot(1, 8, 1)
        plt.imshow(get_rgb_data(self.img))
        plt.title("Original Image")
        plt.axis('off')

        # -------------------------------
        # DTCWT结果（6方向）
        # -------------------------------
        # 低频（复数幅值）
        plt.subplot(1, 8, 2)
        # dtcwt_low_mag = torch.sqrt(self.dtcwt_low[..., 0] ** 2 + self.dtcwt_low[..., 1] ** 2)
        plt.imshow(get_rgb_data(self.dtcwt_low))
        plt.title("DTCWT Lowpass (Mag)")
        plt.axis('off')

        # 高频（6方向幅值）
        dtcwt_high_mag = torch.sqrt(self.dtcwt_high[0][..., 0] ** 2 +
                                    self.dtcwt_high[0][..., 1] ** 2)  # [B,C,6,H,W]
        for i in range(6):
            plt.subplot(1, 8, 3 + i)
            plt.imshow(get_rgb_data(dtcwt_high_mag[:, :, i, :, :]))
            plt.title(f"DTCWT Band {i + 1}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("dtcwt", dpi=300)


# 使用示例
if __name__ == "__main__":
    analyzer = DTCWTAnalyzer(
        img_path="/home/disk2/ZR/datasets/AVIRIS/512/rain/storm/f130804t01p00r04rdn_e_23.npy",
        scales=1,
        wave='qshift_a'
    )
    analyzer.visualize()
