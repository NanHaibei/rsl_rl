
from __future__ import annotations

import torch
import torch.nn as nn

class R2Plus1DFeatureExtractor(nn.Module):
    """R(2+1)D卷积网络用于视频特征提取
    
    该网络将3D卷积分解为2D空间卷积和1D时间卷积，适合处理小分辨率视频流。
    专门为11×11分辨率、5帧的视频设计。
    
    Args:
        input_channels: 输入通道数，默认为1（灰度图）
        output_dim: 输出特征维度
        num_frames: 视频帧数，默认为5
        spatial_size: 空间分辨率，默认为(11, 11)
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        output_dim: int = 64,
        num_frames: int = 5,
        spatial_size: tuple[int, int] = (11, 11),
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        
        # R(2+1)D Block 1: 空间卷积 + 时间卷积
        # 输入: [batch, C, T, H, W] = [batch, 1, 5, 11, 11]
        self.spatial_conv1 = nn.Conv3d(
            input_channels, 16, 
            kernel_size=(1, 3, 3),  # 只在空间维度卷积
            padding=(0, 1, 1),
            bias=False
        )
        self.bn1_spatial = nn.BatchNorm3d(16)
        
        self.temporal_conv1 = nn.Conv3d(
            16, 16,
            kernel_size=(3, 1, 1),  # 只在时间维度卷积
            padding=(1, 0, 0),
            bias=False
        )
        self.bn1_temporal = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        # R(2+1)D Block 2
        # 输出: [batch, 16, 5, 11, 11]
        self.spatial_conv2 = nn.Conv3d(
            16, 32,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),  # 空间下采样
            padding=(0, 1, 1),
            bias=False
        )
        self.bn2_spatial = nn.BatchNorm3d(32)
        
        self.temporal_conv2 = nn.Conv3d(
            32, 32,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.bn2_temporal = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        # R(2+1)D Block 3
        # 输出: [batch, 32, 5, 5, 5]
        self.spatial_conv3 = nn.Conv3d(
            32, 64,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False
        )
        self.bn3_spatial = nn.BatchNorm3d(64)
        
        self.temporal_conv3 = nn.Conv3d(
            64, 64,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.bn3_temporal = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        # 全局平均池化
        # 输出: [batch, 64, 5, 5, 5]
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 全连接层映射到目标维度
        self.fc = nn.Linear(64, output_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入视频张量，形状为 [batch, channels, num_frames, height, width]
               或 [batch, num_frames, height, width] (会自动添加通道维度)
        
        Returns:
            特征向量，形状为 [batch, output_dim]
        """
        # 检查输入维度并调整
        if x.dim() == 4:
            # [batch, T, H, W] -> [batch, 1, T, H, W]
            x = x.unsqueeze(1)
        
        # 验证输入形状
        batch_size, channels, num_frames, height, width = x.shape
        assert channels == self.input_channels, f"Expected {self.input_channels} channels, got {channels}"
        assert num_frames == self.num_frames, f"Expected {self.num_frames} frames, got {num_frames}"
        
        # R(2+1)D Block 1
        x = self.spatial_conv1(x)
        x = self.bn1_spatial(x)
        x = self.temporal_conv1(x)
        x = self.bn1_temporal(x)
        x = self.relu1(x)
        
        # R(2+1)D Block 2
        x = self.spatial_conv2(x)
        x = self.bn2_spatial(x)
        x = self.temporal_conv2(x)
        x = self.bn2_temporal(x)
        x = self.relu2(x)
        
        # R(2+1)D Block 3
        x = self.spatial_conv3(x)
        x = self.bn3_spatial(x)
        x = self.temporal_conv3(x)
        x = self.bn3_temporal(x)
        x = self.relu3(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x


def create_r2plus1d_feature_extractor(
    input_channels: int = 1,
    output_dim: int = 64,
    num_frames: int = 5,
    spatial_size: tuple[int, int] = (11, 11),
) -> R2Plus1DFeatureExtractor:
    """创建R(2+1)D特征提取器的便捷函数
    
    Args:
        input_channels: 输入通道数（1=灰度图，3=RGB）
        output_dim: 输出特征维度
        num_frames: 视频帧数
        spatial_size: 空间分辨率 (height, width)
    
    Returns:
        R2Plus1DFeatureExtractor实例
    
    Example:
        >>> # 创建特征提取器
        >>> extractor = create_r2plus1d_feature_extractor(
        ...     input_channels=1,
        ...     output_dim=64,
        ...     num_frames=5,
        ...     spatial_size=(11, 11)
        ... )
        >>> 
        >>> # 输入视频数据 [batch, frames, height, width]
        >>> video = torch.randn(32, 5, 11, 11)
        >>> 
        >>> # 提取特征
        >>> features = extractor(video)
        >>> print(features.shape)  # [32, 64]
    """
    model = R2Plus1DFeatureExtractor(
        input_channels=input_channels,
        output_dim=output_dim,
        num_frames=num_frames,
        spatial_size=spatial_size,
    )
    
    print(f"Created R(2+1)D Feature Extractor:")
    print(f"  Input: [{input_channels}, {num_frames}, {spatial_size[0]}, {spatial_size[1]}]")
    print(f"  Output: [{output_dim}]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model