import torch
from torch import nn
import torch.nn.functional as F


# 多尺度特征提取器[对特征进入到不同尺寸的卷积核加强提取]
class FeatureExtractor(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 使用不同尺寸的卷积核进行特征提取
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=9, padding=4)
        self.conv1x1 = nn.Conv2d(out_channel * 3, out_channel, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 分别使用不同尺寸的卷积核进行卷积操作
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv1x1(x)


# 聚焦感知的相对位置编码（FAP)
class FocalAwareRelativePositionEncoding(nn.Module):
    def __init__(self, channels=32, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        # 焦点参数：均匀随机初始化
        self.focal_points = nn.Parameter(torch.rand(num_heads, 2))
        # 可学习缩放因子
        self.scale_factor = nn.Parameter(torch.ones(1) * 0.1)
        # 位置编码投影
        self.pos_proj = nn.Linear(2, channels // num_heads)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        b, c, h, w = x.size()
        device = x.device
        # 生成网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        coordinates = torch.stack([grid_x, grid_y], dim=-1)
        # 多焦点位置编码
        attention_maps = []
        for i in range(self.num_heads):
            rel_pos = coordinates - self.focal_points[i]
            pos_encoding = torch.sin(self.pos_proj(rel_pos) * self.scale)
            attention_maps.append(pos_encoding)
        pos_encoding = torch.cat(attention_maps, dim=-1).permute(2, 0, 1)
        # 自适应加权融合
        return x + self.scale_factor * pos_encoding.unsqueeze(0)


#
class GaussianEdgeAwareSmooth(nn.Module):
    def __init__(self, channels=32, kernel_size=5, sigma=1.0, trainable=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.trainable = trainable
        # 初始化高斯核
        self.gaussian_kernel = self._create_gaussian_kernel()
        # 如果允许训练，则注册为可学习参数
        if trainable:
            self.gaussian_kernel = nn.Parameter(self.gaussian_kernel, requires_grad=True)

    # 生成高斯核（可学习或固定）
    def _create_gaussian_kernel(self):
        # 创建坐标网格
        x = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        y = x.clone()
        y, x = torch.meshgrid(y, x, indexing='ij')
        # 计算高斯分布
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        # 转换为卷积核格式 [out_ch, in_ch, H, W]
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # 匹配通道数
        return kernel

    def forward(self, x):
        return F.conv2d(
            x,
            self.gaussian_kernel,
            padding=self.kernel_size // 2,  # 保持尺寸不变
            groups=x.shape[1]  # 分组卷积（每个通道独立处理）
        )


# 编码器
class Encoder(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim_in * 2, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in * 2, dim_in, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim_in, eps=1e-5, momentum=0.1)
        )
        self.norm = nn.InstanceNorm2d(dim_in)
        self.focal_aware_relative_position_encoding = FocalAwareRelativePositionEncoding(dim_in, num_heads=8)  # FAP 模块
        self.gaussian_smoothing = GaussianEdgeAwareSmooth(channels=dim_in, kernel_size=5)  # GES 模块
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # 经过layers层
        x = self.layers(x)
        # 进行聚焦感知的相对位置编码
        fap_x = self.norm(self.focal_aware_relative_position_encoding(x))
        fap_x = fap_x * x
        # 进行高斯平滑进行边缘感知平滑
        fap_x = self.gaussian_smoothing(fap_x)
        # 残差连接
        out = fap_x + residual

        return self.relu(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_attention


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        x = x * attention
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# 多组空间通道注意力机制（GCE）
class GroupCBAMEnhancer(nn.Module):
    def __init__(self, channel, group=8, cov1=1, cov2=1):
        super().__init__()
        self.cov1 = None
        self.cov2 = None
        if cov1 != 0:
            self.cov1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.group = group
        cbam = []
        for i in range(self.group):
            cbam_ = CBAM(channel // group)
            cbam.append(cbam_)
        self.cbam = nn.ModuleList(cbam)
        self.sigomid = nn.Sigmoid()
        if cov2 != 0:
            self.cov2 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        x0 = x
        if self.cov1 != None:
            x = self.cov1(x)
        y = torch.split(x, x.size(1) // self.group, dim=1)
        mask = []
        for y_, cbam in zip(y, self.cbam):
            y_ = cbam(y_)
            y_ = self.sigomid(y_)
            mean = torch.mean(y_, [1, 2, 3])
            mean = mean.view(-1, 1, 1, 1)
            gate = torch.ones_like(y_) * mean
            mk = torch.where(y_ > gate, 1, y_)
            mask.append(mk)
        mask = torch.cat(mask, dim=1)
        x = x * mask
        if self.cov2 != None:
            x = self.cov2(x)
        x = x + x0

        return x


# 解码器
class Decoder(nn.Module):
    def __init__(self, dim_in, depth):
        super().__init__()
        self.layers1 = nn.ModuleList([])
        for _ in range(depth):
            self.layers1.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_in * 2, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(dim_in * 2, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True)
            ]))
            dim_in = dim_in * 2
        self.layers2 = nn.ModuleList([])
        self.group_CBAM_enhancer = GroupCBAMEnhancer(dim_in, group=4)
        for _ in range(depth):
            self.layers2.append(nn.ModuleList([
                nn.Conv2d(dim_in, dim_in // 2, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(dim_in // 2, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True)
            ]))
            dim_in = dim_in // 2
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        for conv, bn, relu in self.layers1:
            x = relu(bn(conv(x)))
        x = self.group_CBAM_enhancer(x)
        for conv, bn, relu in self.layers2:
            x = relu(bn(conv(x)))
        x = x + residual
        return x


class Net(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.multi_scale_extractor = FeatureExtractor(img_channels, 32)
        self.encoder = Encoder(32)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU()
        )
        self.decoder = Decoder(32, 2)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, A, B):
        # 进行多尺度特征提取
        A = self.multi_scale_extractor(A)
        B = self.multi_scale_extractor(B)
        # 编码
        A_encoder = self.encoder(A)
        B_encoder = self.encoder(B)
        # 特征融合
        Fused = self.conv3x3(torch.cat([A_encoder, B_encoder], dim=1))
        # 解码
        Fused_decoder = self.decoder(Fused)
        # 重建
        Fused_fin = self.reconstruct(Fused_decoder)

        # 显示过程特征图
        # import os
        # from utils.showFeatureMap import draw_features_1
        # os.makedirs('1_feature/', exist_ok=True)
        # draw_features_1(A, '1_feature/1A_multi_scale_extractor')
        # draw_features_1(B, '1_feature/1B_multi_scale_extractor')
        # draw_features_1(A_encoder, '1_feature/2A_encoder')
        # draw_features_1(B_encoder, '1_feature/2B_encoder')
        # draw_features_1(Fused, '1_feature/3_fused')
        # draw_features_1(Fused_decoder, '1_feature/4_fused_decoder')

        return Fused_fin


if __name__ == '__main__':
    A = torch.rand(1, 3, 256, 256)
    B = torch.rand(1, 3, 256, 256)
    model = Net()
    NetOut = model(A, B)
    print(model)
    print(NetOut.shape)




