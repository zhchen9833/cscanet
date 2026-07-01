import time

import torch
from einops import rearrange

from mamba_ssm import Mamba2
from torch import nn

from uchiha import ClassicMambaBlock
from uchiha.models.modules.cross_transformer import SpatialUniCrossAttention, ChannelUniCrossAttention
from uchiha.models.modules.mamba import ChannelMambaBlock
from uchiha.models.modules.visual_mamba import SelectiveScan2D, VisualMambaBlock


def bench():
    # x = torch.randn(2, 256, 56, 56).cuda()
    # x2 = torch.randn(2, 256, 56, 56).cuda()
    # x = torch.randn(2, 9, 256).cuda()
    # x2 = torch.randn(2, 9, 256).cuda()
    x = torch.randn(2, 256, 9).cuda()
    x2 = torch.randn(2, 256, 9).cuda()
    SS2D = SelectiveScan2D(
        input_dim=96,
        hidden_state=16,
        expand_ratio=2,
        channel_first=True
    ).cuda()

    VSSBlock = VisualMambaBlock(
        input_dim=96,
        hidden_state=16,
        expand_ratio=2,
        channel_first=True
    ).cuda()

    Mamba2Block = Mamba2(
        d_model=256,  # Model dimension d_model
        d_state=128,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
        headdim=64,
    ).cuda()

    ClassicMamba = ClassicMambaBlock(
        input_dim=256,
        hidden_state=16,
        expand_ratio=2,
    ).cuda()

    ChannelMamba = ChannelMambaBlock(
        seq_len=9,
        hidden_state=16,
        expand_ratio=2,
    ).cuda()

    SCA = SpatialUniCrossAttention(
        input_dim=256,
        num_heads=8
    ).cuda()

    CCA = ChannelUniCrossAttention(
        seq_len=9,
        factor=4.0,
    ).cuda()

    print(x.shape)

    tim = time.time()
    y = CCA(x, x2)
    print(time.time() - tim)

    print(y.shape)


if __name__ == '__main__':
    bench()
