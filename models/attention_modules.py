"""
注意力机制模块
为海洋舰船检测设计的注意力模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention
    通道注意力机制
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """通道注意力模块 (CBAM组件)"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力模块 (CBAM组件)"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    通道 + 空间注意力
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MarineContextAttention(nn.Module):
    """
    海洋场景上下文注意力
    针对海洋舰船检测设计
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # 局部细节分支
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 融合权重
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局上下文
        global_feat = self.global_branch(x)
        global_feat = global_feat.expand_as(x)

        # 局部细节
        local_feat = self.local_branch(x)

        # 融合
        combined = torch.cat([global_feat, local_feat], dim=1)
        attention = self.sigmoid(self.fusion(combined))

        return x * attention + x  # 残差连接


def get_attention_module(attention_type: str, channels: int):
    """
    获取注意力模块

    Args:
        attention_type: 注意力类型 ('se', 'cbam', 'marine')
        channels: 通道数

    Returns:
        注意力模块实例
    """
    attention_map = {
        'se': SEAttention,
        'cbam': CBAM,
        'marine': MarineContextAttention,
    }

    if attention_type not in attention_map:
        raise ValueError(f"未知的注意力类型: {attention_type}")

    return attention_map[attention_type](channels)
