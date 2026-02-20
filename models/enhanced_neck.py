"""
增强Neck模块
改进特征金字塔网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedPANet(nn.Module):
    """
    增强版PANet
    添加注意力机制和特征融合
    """

    def __init__(self, in_channels: list = [256, 512, 1024], out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 自顶向下路径
        self.lateral_conv0 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral_conv1 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral_conv2 = nn.Conv2d(in_channels[0], out_channels, 1)

        # 自底向上路径
        self.fpn_conv0 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 上采样和下采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2)

    def forward(self, features):
        """
        Args:
            features: [C3, C4, C5] 三个尺度的特征
        Returns:
            融合后的特征 [P3, P4, P5]
        """
        c3, c4, c5 = features

        # 自顶向下
        p5 = self.lateral_conv0(c5)
        p4 = self.lateral_conv1(c4) + self.upsample(p5)
        p3 = self.lateral_conv2(c3) + self.upsample(p4)

        # 自底向上
        p3 = self.fpn_conv0(p3)
        p4 = self.fpn_conv1(p4 + self.downsample(p3))
        p5 = self.fpn_conv2(p5 + self.downsample(p4))

        return [p3, p4, p5]


class ASFF(nn.Module):
    """
    Adaptively Spatial Feature Fusion
    自适应空间特征融合
    """

    def __init__(self, level: int, channels: int = 256):
        super().__init__()
        self.level = level
        self.channels = channels

        # 权重卷积
        self.weight_conv = nn.Conv2d(channels * 3, 3, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] 三个尺度的特征
        Returns:
            融合后的特征
        """
        p3, p4, p5 = features

        # 统一尺寸
        target_size = p3.shape[2:] if self.level == 0 else (
            p4.shape[2:] if self.level == 1 else p5.shape[2:]
        )

        p3_resized = F.interpolate(p3, size=target_size, mode='nearest')
        p4_resized = F.interpolate(p4, size=target_size, mode='nearest')
        p5_resized = F.interpolate(p5, size=target_size, mode='nearest')

        # 计算融合权重
        concat_feat = torch.cat([p3_resized, p4_resized, p5_resized], dim=1)
        weights = self.weight_conv(concat_feat)
        weights = self.softmax(weights)

        # 加权融合
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        fused = w1 * p3_resized + w2 * p4_resized + w3 * p5_resized

        return fused
