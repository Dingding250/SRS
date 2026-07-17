import torch
import torch.nn as nn
import torch.nn.functional as F

class MS_LSSDBlock(nn.Module):
    """
    多尺度局部自相似性判别模块 (Multi-Scale LSSD)
    通过并行不同大小的感受野窗口，捕捉从微观纹理到中观结构的生成伪影。
    """
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super(MS_LSSDBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        
        # 为每个尺度定义 Unfold 操作
        self.unfolds = nn.ModuleList([
            nn.Unfold(kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        
        # 计算所有尺度的总维度: 3x3=9, 5x5=25, 7x7=49 -> 总和 83
        total_diff_dims = sum([k * k for k in kernel_sizes])
        
        # 多尺度融合判别器
        self.discriminator = nn.Sequential(
            nn.Conv2d(total_diff_dims, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 可学习的残差控制
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape
        
        # 1. 提取空间分布（通道均值）
        x_spatial = torch.mean(x, dim=1, keepdim=True) # [B, 1, H, W]
        
        # 2. 并行提取多尺度局部差异
        diff_maps = []
        for i, k in enumerate(self.kernel_sizes):
            # 提取邻域块 [B, k*k, H*W]
            unfolded = self.unfolds[i](x_spatial)
            unfolded = unfolded.view(b, k * k, h, w)
            
            # 计算局部偏差（中心像素与领域每个点的差异）
            diff = torch.abs(unfolded - x_spatial)
            diff_maps.append(diff)
        
        # 3. 拼接所有尺度的差异图
        # shape: [B, (9+25+49), H, W]
        all_scales_diff = torch.cat(diff_maps, dim=1)
        
        # 4. 生成多尺度伪影引导权重
        artifact_weight = self.discriminator(all_scales_diff)
        
        # 5. 增强特征表达
        return x * (1 + self.gamma * artifact_weight)
    

class CAFR(nn.Module):
    def __init__(self, in_channels):
        super(CAFR, self).__init__()
        # 3x3卷积用于特征对齐（调整通道数与高层特征一致）
        self.align_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 全连接层生成全局光谱向量
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, high_level_feat, low_level_feat):

        # 1. 特征对齐：低层特征通过3x3卷积调整通道，并上采样至高层特征尺寸
        low_level_feat = self.align_conv(low_level_feat)  # (B, C, 2H, 2W)
        high_level_feat = self.align_conv(high_level_feat)

        # 2. 计算全局光谱向量（GAP + 全连接）
        # 高层特征全局平均池化 + FC
        gap_high = F.adaptive_avg_pool2d(high_level_feat, (1, 1)).flatten(1)  # (B, C)
        Vh = self.fc(gap_high).unsqueeze(-1)  # (B, C, 1) → 高层向量 (C×1)
        
        # 低层特征全局平均池化 + FC
        gap_low = F.adaptive_avg_pool2d(low_level_feat, (1, 1)).flatten(1)  # (B, C)
        Vl = self.fc(gap_low).unsqueeze(-1)  # (B, C, 1) → 低层向量 (C×1)

        # 3. 计算交叉光谱注意力权重
        A = torch.matmul(Vh, Vl.transpose(1, 2))  # (B,C,1) @ (B,1,C) → (B,C,C) 生成C×C注意力矩阵
        A = F.softmax(A, dim=-1)  # 沿最后一维归一化

        # 4. 生成融合权重
        W_h = torch.matmul(A.transpose(-1, -2), Vh)  # (B,C,C) @ (B,C,1) → (B,C,1)
        W_l = torch.matmul(A, Vl)  # (B,C,C) @ (B,C,1) → (B,C,1)
        
        # 扩展为4D张量以支持广播乘法
        W_h = W_h.unsqueeze(-1)  # (B,C,1,1)
        W_l = W_l.unsqueeze(-1)  # (B,C,1,1)
        
        # 5. 特征加权融合
        refined_feat = W_h * high_level_feat + W_l * low_level_feat
        
        return refined_feat


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # 深度可分离卷积（groups=out_channels确保通道分离）
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, 
                      groups=out_channels, bias=False),  # 深度卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),  # 点卷积
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 跳跃连接
        if out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            self.shortcut = nn.Identity()
            
        # --- 使用多尺度 LSSD ---
        self.ms_lssd = MS_LSSDBlock(out_channels, kernel_sizes=[3, 5, 7])
        # ---------------------
        
        self.act = nn.GELU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        residual = self.shortcut(identity)
        
        out = self.main_branch(x)
        out = self.skip_add.add(out, residual)
        
        # 应用多尺度判别
        out = self.ms_lssd(out)
        
        return self.act(out)

    
class FeatureFusionBlock(nn.Module):
    
    def __init__(
        self, 
        in_features, 
        out_features, 
    ):
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = Res_block(in_features, in_features)
        self.resConfUnit2 = Res_block(in_features, out_features)
        
        self.skip_add = nn.quantized.FloatFunctional()
        self.cafr = CAFR(in_features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.cafr(output, res)

            # output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output