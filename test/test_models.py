"""
Models模块单元测试
"""
import unittest
import sys
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attention_modules import SEAttention, CBAM, MarineContextAttention, get_attention_module
from models.enhanced_neck import EnhancedPANet


class TestSEAttention(unittest.TestCase):
    """测试SE注意力模块"""

    def test_initialization(self):
        """测试初始化"""
        se = SEAttention(channels=128)
        self.assertIsNotNone(se)

    def test_forward_pass(self):
        """测试前向传播"""
        se = SEAttention(channels=128)
        x = torch.randn(2, 128, 32, 32)
        out = se(x)
        self.assertEqual(out.shape, x.shape)

    def test_channel_reduction(self):
        """测试通道降维"""
        se = SEAttention(channels=128, reduction=16)
        # 检查fc层的维度
        self.assertEqual(se.fc[0].out_features, 8)  # 128 / 16 = 8
        self.assertEqual(se.fc[2].out_features, 128)


class TestCBAM(unittest.TestCase):
    """测试CBAM注意力模块"""

    def test_initialization(self):
        """测试初始化"""
        cbam = CBAM(channels=128)
        self.assertIsNotNone(cbam)
        self.assertIsNotNone(cbam.channel_attention)
        self.assertIsNotNone(cbam.spatial_attention)

    def test_forward_pass(self):
        """测试前向传播"""
        cbam = CBAM(channels=128)
        x = torch.randn(2, 128, 32, 32)
        out = cbam(x)
        self.assertEqual(out.shape, x.shape)


class TestMarineContextAttention(unittest.TestCase):
    """测试MarineContext注意力模块"""

    def test_initialization(self):
        """测试初始化"""
        mca = MarineContextAttention(channels=128)
        self.assertIsNotNone(mca)
        self.assertIsNotNone(mca.global_branch)
        self.assertIsNotNone(mca.local_branch)

    def test_forward_pass(self):
        """测试前向传播"""
        mca = MarineContextAttention(channels=128)
        x = torch.randn(2, 128, 32, 32)
        out = mca(x)
        self.assertEqual(out.shape, x.shape)


class TestAttentionFactory(unittest.TestCase):
    """测试注意力模块工厂函数"""

    def test_get_se_attention(self):
        """测试获取SE注意力"""
        module = get_attention_module('se', 128)
        self.assertIsInstance(module, SEAttention)

    def test_get_cbam_attention(self):
        """测试获取CBAM注意力"""
        module = get_attention_module('cbam', 128)
        self.assertIsInstance(module, CBAM)

    def test_get_marine_attention(self):
        """测试获取MarineContext注意力"""
        module = get_attention_module('marine', 128)
        self.assertIsInstance(module, MarineContextAttention)

    def test_invalid_attention_type(self):
        """测试无效的注意力类型"""
        with self.assertRaises(ValueError):
            get_attention_module('invalid', 128)


class TestEnhancedPANet(unittest.TestCase):
    """测试增强PANet"""

    def test_initialization(self):
        """测试初始化"""
        panet = EnhancedPANet(in_channels=[256, 512, 1024], out_channels=256)
        self.assertIsNotNone(panet)

    def test_forward_pass(self):
        """测试前向传播"""
        panet = EnhancedPANet(in_channels=[256, 512, 1024], out_channels=256)

        # 创建输入特征
        c3 = torch.randn(1, 256, 64, 64)
        c4 = torch.randn(1, 512, 32, 32)
        c5 = torch.randn(1, 1024, 16, 16)

        features = [c3, c4, c5]
        outputs = panet(features)

        # 检查输出
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape[1], 256)  # P3 channels
        self.assertEqual(outputs[1].shape[1], 256)  # P4 channels
        self.assertEqual(outputs[2].shape[1], 256)  # P5 channels


if __name__ == '__main__':
    unittest.main()
