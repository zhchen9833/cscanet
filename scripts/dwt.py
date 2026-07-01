import numpy as np
import pywt
import torch
from matplotlib import pyplot as plt
from pytorch_wavelets import DWTForward

from analysis import analyze_rain_strength, plot_smooth_spectral_curves
from utils import resize_image


def get_rgb_data(data, bands=(30, 20, 10)):
    """

    Args:
        data (): Tensor(B, C, H, W)
        bands ():

    Returns: Ndarray(H, W, 3)

    """
    if isinstance(data, torch.Tensor):
        data = data.squeeze(0).permute(1, 2, 0).numpy()
    else:
        data = np.transpose(data, (1, 2, 0))
    data = data[:, :, list(bands)]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    return data


def load(path):
    """

    Args:
        path ():

    Returns: Tensor(B,C,H,W)

    """
    org_data = np.load(path)
    org_data = torch.from_numpy(org_data).permute(2, 0, 1)
    org_data = org_data.unsqueeze(0)

    return org_data


class DWT2d:
    """ 2D Discrete Wavelet Transform

    Args:
        scales (int): Number of scales (number of transform performed)
        wave (str): The type of wavelet
        padding (str): data padding mode before transformation
    """

    def __init__(self,
                 lq_path='/home/disk2/ZR/datasets/AVIRIS/512/storm/f130410t01p00r15rdn_e_13.npy',
                 gt_path='/home/disk2/ZR/datasets/AVIRIS/512/gt/f130410t01p00r15rdn_e_13.npy',
                 scales=1,
                 wave='haar',
                 padding='zero'):
        super().__init__()

        self.lq_data = load(lq_path)
        self.gt_data = load(gt_path)
        self.wavelet_transform = DWTForward(J=scales, wave=wave, mode=padding)

        self.LL, self.LH, self.HL, self.HH = None, None, None, None
        self.downsample_data = resize_image(self.lq_data.squeeze().permute(1, 2, 0).numpy(), 1.0 / (scales + 1))

    def forward(self):
        # x.shape:B, C, H, W
        out = self.wavelet_transform(self.lq_data)
        LL, H = out
        LH, HL, HH = H[0][:, :, 0, :, :], H[0][:, :, 1, :, :], H[0][:, :, 2, :, :]
        self.LL, self.LH, self.HL, self.HH = get_rgb_data(LL), get_rgb_data(LH), get_rgb_data(HL), get_rgb_data(HH)

    def transform(self):
        data = self.lq_data.squeeze().numpy()
        LL, (LH, HL, HH) = pywt.dwt2(data, 'haar')
        self.LL, self.LH, self.HL, self.HH = LL, LH, HL, HH

    def visualize(self):
        self.analyze_dwt()
        self.visualize_dwt()

    def analyze_dwt(self):
        analyze_rain_strength(self.LH, 0, 0)

    def visualize_dwt(self):
        top = np.concatenate([get_rgb_data(self.LL), get_rgb_data(self.HL)], axis=1)
        bottom = np.concatenate([get_rgb_data(self.LH), get_rgb_data(self.HH)], axis=1)
        combined = np.concatenate([top, bottom], axis=0)
        # 创建可视化
        plt.figure(figsize=(12, 6))

        # 原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(get_rgb_data(self.lq_data))
        plt.title('Original Image')
        plt.axis('off')

        # 拼接后的子带
        plt.subplot(1, 2, 2)
        plt.imshow(combined)
        plt.title('Wavelet Components\nLL | HL\nLH | HH')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('dwt.jpg')


if __name__ == '__main__':
    """
        子带	    名称	                        含义
        LL	    低频 - 横向低 + 纵向低	    图像的近似信息，包含图像的轮廓、背景等整体结构
        LH	    横向低 + 纵向高频	        图像中的垂直边缘信息（例如纵向变化较快的纹理）
        HL	    横向高频 + 纵向低	        图像中的水平边缘信息（例如横向变化较快的纹理）
        HH	    横向高频 + 纵向高频	    对角线方向的高频信息（例如图像中的细节、噪声等）
        
        Reference: https://blog.csdn.net/qq_30815237/article/details/89704855
    """
    _bands = (36, 19, 8)
    # bands = (136, 67, 18)

    transform = DWT2d(
        lq_path='/home/disk2/ZR/datasets/AVIRIS/512/rain/storm/f130804t01p00r04rdn_e_23.npy',
        gt_path='/home/disk2/ZR/datasets/AVIRIS/512/gt/f130804t01p00r04rdn_e_23.npy'
    )
    transform.transform()
    # transform.visualize()
    plot_smooth_spectral_curves(transform.gt_data.squeeze().numpy(), transform.lq_data.squeeze().numpy(), 56, 56)
