import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss

from pytorch_wavelets import DWTForward, DWTInverse

from .builder import CRITERION

CRITERION.register_module(module=MSELoss)
CRITERION.register_module(module=L1Loss)
CRITERION.register_module(module=CrossEntropyLoss)


@CRITERION.register_module()
class MultiL1Loss(nn.Module):
    """ multi-branch L1Loss

    When the network has multiple pipelines, the loss is calculated for
    each pipeline and then these losses are weighted and constitute the final loss.

    Args:
        weights (List[int] | int): weights of each pipeline
            If a list is provided, it will be weighted according to the value of the list,
            if not, the mean weight will be assigned initially and updated with backward propagation.
            Default: 2
    """

    def __init__(self,
                 weights=2):
        super(MultiL1Loss, self).__init__()
        self.weights = weights

    def forward(self, prediction, target):
        loss = 0
        for idx, pred in enumerate(prediction):
            loss += F.l1_loss(pred, target) * self.weights[idx]

        return loss


@CRITERION.register_module()
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): model output shape is (B, C, H, W)
            target (torch.Tensor): real label shape is (B, C, H, W)

        Returns:
            loss (torch.Tensor): scalar loss value
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps).mean()
        return loss


class SpectrumConstancyLoss(nn.Module):
    def __init__(self):
        super(SpectrumConstancyLoss, self).__init__()
        # Define differential convolution kernels (up, down, left, and right)
        self.register_buffer('diff_kernel', torch.tensor([[-1, 0, 1]]).float())

    def forward(self, pred):
        """
        Args:
            pred (torch.Tensor): 输入图像，形状为 (B, C, H, W)

        Returns:
            loss (torch.Tensor): 标量，Spectrum Constancy Loss 值
        """
        B, C, H, W = pred.shape

        # calculate horizontal difference dx
        dx = F.conv2d(pred, weight=self.diff_kernel.view(1, 1, 1, 3).repeat(C, 1, 1, 1),
                      padding=(0, 1), groups=C)  # (B, C, H, W)

        # calculate vertical difference dy
        dy = F.conv2d(pred, weight=self.diff_kernel.view(1, 1, 3, 1).repeat(C, 1, 1, 1),
                      padding=(1, 0), groups=C)  # (B, C, H, W)

        # Sum of squares in the channel dimension (Euclidean distance squared)
        dx_sq = dx.pow(2).sum(dim=1)  # (B, H, W)
        dy_sq = dy.pow(2).sum(dim=1)  # (B, H, W)

        #total loss average or sum
        loss = (dx_sq + dy_sq).mean()

        return loss


@CRITERION.register_module()
class AdaptiveSpatialSpectralLossV2(nn.Module):
    def __init__(self, eps=1e-3, min_alpha=0.1, max_alpha=0.9):
        """
        Args:
            eps: Charbonnier smoothing-coefficient
            min_alpha/max_alpha: Clipping range for dynamic alpha (to avoid extreme cases)
        """
        super().__init__()
        self.eps = eps
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        diff = torch.abs(pred - target)  # [B, C, H, W]

        spatial_diff = torch.mean(diff.view(B, C, -1), dim=1)  # B L
        spectral_diff = torch.mean(diff.view(B, C, -1), dim=2)  # B C

        spatial_loss = torch.sqrt(spatial_diff ** 2 + self.eps ** 2).mean()  # L1
        spectral_loss = torch.sqrt(spectral_diff ** 2 + self.eps ** 2).mean()  # L2

        # dynamic alpha with clipping
        alpha = spatial_loss / (spatial_loss + spectral_loss + 1e-6)
        alpha = torch.clamp(alpha, self.min_alpha, self.max_alpha)

        # joint loss
        joint_loss = alpha * spatial_loss + (1 - alpha) * spectral_loss

        return joint_loss


class WaveletProcessor(nn.Module):
    def __init__(self, wavelet='haar', level=2, mode='reflect'):
        super().__init__()
        self.dwt = DWTForward(wave=wavelet, J=level, mode=mode)  # wavelet decomposition
        self.idwt = DWTInverse(wave=wavelet)  # wavelet reconstruction
        self.level = level

    def forward(self, x):
        """x: [B, C, H, W]"""
        # wavelet decomposition
        coeffs = self.dwt(x)  # (LL, [LH, HL, HH] * level)

        # extract each frequency band component
        LL = coeffs[0]  # low frequency component [B, C, H/(2^level), W/(2^level)]
        HL = [c[0][:, :, 0] for c in coeffs[1]]  # horizontal high frequency
        LH = [c[0][:, :, 1] for c in coeffs[1]]  # vertical high frequency
        HH = [c[0][:, :, 2] for c in coeffs[1]]  # diagonal high frequency

        return {
            'LL': LL,  # low frequency
            'HL': HL,  # horizontal high frequency edge detail
            'LH': LH,  # vertical high frequency edge detail
            'HH': HH  # diagonal high frequency texture
        }


@CRITERION.register_module()
class SpatialSpectralFreqLoss(nn.Module):
    def __init__(self, wave='haar', J=1, mode='reflect'):
        """
        Args:
            wave: wavelet basis type haar db2 etc
            J: wavelet decomposition level
        """
        super().__init__()
        self.wavelet = WaveletProcessor(wave, J, mode).cuda()
        self.criterion = MSELoss()

    def forward(self, pred, target):
        spatial_spectral_loss = self.criterion(pred, target)

        # wavelet frequency domain loss
        pred_wave = self.wavelet(pred)  # predicting the wavelet coefficients of an image
        target_wave = self.wavelet(target)  # wavelet coefficients of real images
        freq_loss = 0
        weights = [0.5, 0.2, 0.2, 0.1]
        keys = ['LL', 'HL', 'LH', 'HH']
        for key, weight in zip(keys, weights):
            for p, t in zip(pred_wave[key], target_wave[key]):
                freq_loss += self.criterion(p, t) * weight

        # weighted joint loss
        alpha = spatial_spectral_loss / (spatial_spectral_loss + freq_loss)
        total_loss = alpha * spatial_spectral_loss + (1 - alpha) * freq_loss
        return total_loss


if __name__ == '__main__':
    _pred = torch.randn(1, 224, 128, 128)
    criterion = SpectrumConstancyLoss()
    criterion(_pred)
