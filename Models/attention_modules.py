import torch
import torch.nn as nn
import torch.nn.functional as F

class MarineCBAM(nn.Module):
    """
    海洋场景优化的CBAM注意力机制
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 针对海洋水平线特性优化
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, 
                     padding=spatial_kernel//2, bias=False),
            nn.Sigmoid()
        )
        
        # 海洋场景偏置 - 关注图像下半部分（海面区域）
        self.marine_bias = nn.Parameter(torch.ones(1, 1, 1, 1) * 0.3)
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        
        # 应用海洋偏置 - 增强海面区域注意力
        height = x.size(2)
        marine_mask = torch.ones_like(sa)
        marine_mask[:, :, height//2:, :] += self.marine_bias
        sa = sa * marine_mask
        
        x = x * sa
        return x

class SmallObjectAttention(nn.Module):
    """
    小目标专用注意力机制
    """
    def __init__(self, channels):
        super().__init__()
        
        # 高频特征增强 - 小目标通常包含更多高频信息
        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 细节增强模块
        self.detail_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 增强高频特征
        high_freq = self.high_freq_enhance(x)
        
        # 细节注意力
        detail_att = self.detail_attention(x)
        
        # 融合
        x = x + high_freq * 0.5  # 部分增强高频
        x = x * detail_att       # 应用细节注意力
        
        return x