"""
Curvature-Guided Geometric Attention Module (CGAM)
====================================================
基于微分几何曲率的双路注意力模块，用于语义分割网络
(DeepLabV3+、UNet、SegFormer 等) 的即插即用注意力机制。

核心思想：将 CNN 特征图视为二维连续曲面 z = f(x, y)，
通过计算局部微分几何曲率（平均曲率 H、高斯曲率 K）构建
曲率能量图，指导通道注意力与空间注意力的生成。

参考文献风格：IEEE / TGRS / TIP / CVPR
Python 3.10+ | PyTorch 2.x | CUDA | AMP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class CGAM(nn.Module):
    """
    Curvature-Guided Geometric Attention Module

    将特征图视为二维曲面，计算微分几何曲率信息，
    引导双路注意力机制增强边缘、角点、小目标感知。

    Args:
        in_channels:     输入通道数 C
        reduction:       通道降维比率（默认 4，即 C → C/4）
        mlp_ratio:       通道注意力 MLP 缩减比（默认 4，即 C/4 → C/16）
        use_norm_energy: 是否对曲率能量图做 sigmoid 归一化
        eps:             数值稳定性常数
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        mlp_ratio: int = 4,
        use_norm_energy: bool = True,
        residual_init: float = 0.1,
        eps: float = 1e-6,
    ):
        super(CGAM, self).__init__()

        self.in_channels = in_channels
        self.reduction = reduction
        self.mlp_ratio = mlp_ratio
        self.use_norm_energy = use_norm_energy
        self.residual_init = residual_init
        self.eps = eps

        # ---- 缩减后通道数 ----
        self.channels_r = max(1, in_channels // reduction)
        self.mlp_hidden = max(1, self.channels_r // mlp_ratio)

        # ================================================================
        # Step 2: 通道降维 Conv1x1: C → C/4  (BN + ReLU)
        # ================================================================
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, self.channels_r, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.channels_r),
            nn.ReLU(inplace=True),
        )

        # ================================================================
        # Step 3: 固定 Sobel / 差分卷积核（不可训练，register_buffer）
        # ================================================================
        self._build_derivative_kernels()

        # ================================================================
        # Step 6: 通道注意力 MLP: C_r → C_r/mlp_ratio → C_r
        # ================================================================
        self.channel_attn = nn.Sequential(
            nn.Linear(self.channels_r, self.mlp_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden, self.channels_r, bias=True),
            nn.Sigmoid(),
        )

        # ================================================================
        # Step 7: 空间注意力 1×1 Conv: C_r → 1  + Sigmoid
        # ================================================================
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(self.channels_r, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ================================================================
        # Step 9: 升维恢复 Conv1x1: C_r → C  (BN, 无 ReLU)
        # ================================================================
        self.conv_expand = nn.Sequential(
            nn.Conv2d(self.channels_r, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

        self._init_weights()
        self._init_residual_as_weak_branch()

    # ------------------------------------------------------------------
    # 构建固定导数卷积核（register_buffer，不参与训练）
    # ------------------------------------------------------------------
    def _build_derivative_kernels(self):
        """
        构建五组固定 3×3 导数卷积核：

        - sobel_x:  一阶 x 方向  ∂f/∂x
        - sobel_y:  一阶 y 方向  ∂f/∂y
        - laplace_x: 二阶 x 方向  ∂²f/∂x²
        - laplace_y: 二阶 y 方向  ∂²f/∂y²
        - cross_xy:  混合偏导     ∂²f/∂x∂y

        形状均为 (C_r, 1, 3, 3)，适配 depthwise convolution (groups=C_r)。
        """

        # ---- Sobel X: ∂f/∂x ----
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        # ---- Sobel Y: ∂f/∂y ----
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
            dtype=torch.float32,
        )

        # ---- 二阶 x 方向 ∂²f/∂x² (finite difference [1, -2, 1]) ----
        laplace_x = torch.tensor(
            [[ 0.0,  0.0,  0.0],
             [ 1.0, -2.0,  1.0],
             [ 0.0,  0.0,  0.0]],
            dtype=torch.float32,
        )

        # ---- 二阶 y 方向 ∂²f/∂y² ----
        laplace_y = torch.tensor(
            [[0.0,  1.0, 0.0],
             [0.0, -2.0, 0.0],
             [0.0,  1.0, 0.0]],
            dtype=torch.float32,
        )

        # ---- 混合偏导 ∂²f/∂x∂y ----
        # 中心差分: [f(x+h,y+h)-f(x+h,y-h)-f(x-h,y+h)+f(x-h,y-h)] / (4h²)
        cross_xy = torch.tensor(
            [[-1.0,  0.0,  1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  0.0, -1.0]],
            dtype=torch.float32,
        ) / 4.0

        # 扩展为 depthwise 权重形状: (C_r, 1, 3, 3)
        Cr = self.channels_r
        self.register_buffer("kernel_fx",  sobel_x.view(1, 1, 3, 3).expand(Cr, -1, -1, -1).contiguous().clone())
        self.register_buffer("kernel_fy",  sobel_y.view(1, 1, 3, 3).expand(Cr, -1, -1, -1).contiguous().clone())
        self.register_buffer("kernel_fxx", laplace_x.view(1, 1, 3, 3).expand(Cr, -1, -1, -1).contiguous().clone())
        self.register_buffer("kernel_fyy", laplace_y.view(1, 1, 3, 3).expand(Cr, -1, -1, -1).contiguous().clone())
        self.register_buffer("kernel_fxy", cross_xy.view(1, 1, 3, 3).expand(Cr, -1, -1, -1).contiguous().clone())

    # ------------------------------------------------------------------
    # 初始化可训练权重
    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_residual_as_weak_branch(self):
        """Start CGAM as a weak residual branch so it can learn within short runs."""
        expand_bn = self.conv_expand[1]
        nn.init.constant_(expand_bn.weight, 1)
        nn.init.constant_(expand_bn.bias, 0)

    # ------------------------------------------------------------------
    # Step 4: 微分几何曲率计算
    # ------------------------------------------------------------------
    def _compute_curvature(
        self,
        fx: torch.Tensor,
        fy: torch.Tensor,
        fxx: torch.Tensor,
        fyy: torch.Tensor,
        fxy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算平均曲率 H 与高斯曲率 K。

        输入 shape 均为 (B, C_r, H, W)

        Mean Curvature:
            H = ((1+fx²)·fyy - 2·fx·fy·fxy + (1+fy²)·fxx)
                /
                (2·(1+fx²+fy²)^(3/2) + ε)

        Gaussian Curvature:
            K = (fxx·fyy - fxy²)
                /
                ((1+fx²+fy²)² + ε)
        """
        eps = self.eps

        fx2 = fx.pow(2)
        fy2 = fy.pow(2)
        fxy2 = fxy.pow(2)

        # 一阶基本形式系数
        E_coeff = 1.0 + fx2                     # E = 1 + fx²
        G_coeff = 1.0 + fy2                     # G = 1 + fy²
        F_coeff = fx * fy                        # F = fx·fy（未用于 H/K 公式主体）

        # 分母公共项
        denom_common = 1.0 + fx2 + fy2           # 1 + fx² + fy²
        denom_common = denom_common.clamp(min=eps)

        # ---- 平均曲率 H ----
        numerator_H = (
            E_coeff * fyy
            - 2.0 * fx * fy * fxy
            + G_coeff * fxx
        )
        denominator_H = 2.0 * denom_common.pow(1.5) + eps
        H = numerator_H / denominator_H

        # ---- 高斯曲率 K ----
        numerator_K = fxx * fyy - fxy2
        denominator_K = denom_common.pow(2.0) + eps
        K = numerator_K / denominator_K

        return H, K

    # ------------------------------------------------------------------
    # Step 5: 曲率能量图
    # ------------------------------------------------------------------
    def _curvature_energy(self, H: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        E = (|H| + |K|) / 2

        使用 tanh 归一化代替 sigmoid:
        - tanh(0) = 0: 平坦区域能量为零，不激活注意力
        - tanh(x) → 1: 高曲率区域(边缘/角点)趋向最大激活
        - 值域 (0,1), 对比度远优于 sigmoid 的 (0.5, 1.0)
        """
        E = (H.abs() + K.abs()) / 2.0

        if self.use_norm_energy:
            E = torch.tanh(E)

        E = E.clamp(min=0.0, max=1.0)
        return E

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            x:               输入特征图 (B, C, H, W)
            return_attention: 若为 True，返回 (output, attention_dict)
                              attention_dict 包含:
                                - "energy":      曲率能量图 E   (B, C_r, H, W)
                                - "channel_attn": 通道注意力 W_c (B, C_r, 1, 1)
                                - "spatial_attn": 空间注意力 A_s (B, 1, H, W)

        Returns:
            Y: 增强后特征图 (B, C, H, W)
            attention_dict (可选): 用于论文可视化的中间张量
        """
        B, C, H, W = x.shape                     # Step 1

        # ---- Step 2: 通道降维 C → C_r ----
        F_r = self.conv_reduce(x)                # (B, C_r, H, W)
        Cr = F_r.size(1)

        # ---- Step 3: 固定导数场（depthwise conv, groups=Cr）----
        # 确保 kernel dtype 与输入一致（AMP 兼容）
        dtype = F_r.dtype
        pad = 1                                   # 3×3 conv padding=1 保持尺寸

        fx = F.conv2d(
            F_r, self.kernel_fx.to(dtype), bias=None, stride=1, padding=pad, groups=Cr
        )                                        # (B, C_r, H, W)
        fy = F.conv2d(
            F_r, self.kernel_fy.to(dtype), bias=None, stride=1, padding=pad, groups=Cr
        )                                        # (B, C_r, H, W)
        fxx = F.conv2d(
            F_r, self.kernel_fxx.to(dtype), bias=None, stride=1, padding=pad, groups=Cr
        )                                        # (B, C_r, H, W)
        fyy = F.conv2d(
            F_r, self.kernel_fyy.to(dtype), bias=None, stride=1, padding=pad, groups=Cr
        )                                        # (B, C_r, H, W)
        fxy = F.conv2d(
            F_r, self.kernel_fxy.to(dtype), bias=None, stride=1, padding=pad, groups=Cr
        )                                        # (B, C_r, H, W)

        # ---- 梯度裁剪防爆 ----
        fx = fx.clamp(-10.0, 10.0)
        fy = fy.clamp(-10.0, 10.0)
        fxx = fxx.clamp(-10.0, 10.0)
        fyy = fyy.clamp(-10.0, 10.0)
        fxy = fxy.clamp(-10.0, 10.0)

        # ---- Step 4: 曲率计算 ----
        H, K = self._compute_curvature(fx, fy, fxx, fyy, fxy)
        # H, K: (B, C_r, H, W)

        # ---- Step 5: 曲率能量图 ----
        E = self._curvature_energy(H, K)         # (B, C_r, H, W)

        # ---- Step 6: 通道注意力 ----
        # 曲率能量加权特征池化: 高曲率区域的特征响应贡献更多通道统计
        E_weight = E / (E.sum(dim=[2, 3], keepdim=True) + self.eps)
        gap = (F_r * E_weight).sum(dim=[2, 3])  # 加权平均 → (B, C_r)
        W_c = self.channel_attn(gap)             # (B, C_r)
        W_c = W_c.view(B, Cr, 1, 1)              # (B, C_r, 1, 1)

        # ---- Step 7: 空间注意力 ----
        A_s = self.spatial_attn(E)               # (B, 1, H, W)

        # ---- Step 8: 双路融合 ----
        F_att = F_r * (1.0 + W_c * A_s)          # (B, C_r, H, W)
        # 广播: W_c (B,C_r,1,1) × A_s (B,1,H,W) → (B,C_r,H,W)

        # ---- Step 9: 升维恢复 C_r → C ----
        F_out = self.conv_expand(F_att)          # (B, C, H, W)

        # ---- Step 10: 残差连接 ----
        Y = x + self.residual_scale * F_out      # (B, C, H, W)

        if return_attention:
            attn_dict = {
                "energy":       E.detach(),
                "channel_attn": W_c.detach(),
                "spatial_attn": A_s.detach(),
                "residual_scale": self.residual_scale.detach(),
            }
            return Y, attn_dict

        return Y

    # ------------------------------------------------------------------
    # 参数量统计
    # ------------------------------------------------------------------
    def count_parameters(self) -> Dict[str, int]:
        """返回可训练参数与总参数量"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fixed = sum(p.numel() for p in self.buffers())
        total = trainable + fixed
        return {
            "trainable": trainable,
            "fixed_buffers": fixed,
            "total": total,
        }

    # ------------------------------------------------------------------
    # FLOPs 粗略估算
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_flops(in_channels: int, spatial_size: int = 64) -> Dict[str, float]:
        """
        基于输入 (1, C, S, S) 做粗略 FLOPs 估算。

        Returns:
            { "conv_reduce_M": ..., "derivatives_M": ..., "attn_M": ...,
              "conv_expand_M": ..., "total_M": ... }
            单位：M FLOPs
        """
        C = in_channels
        Cr = max(1, C // 4)
        S = spatial_size
        M = 1e-6

        # Conv1x1 reduce: C * Cr * S * S * 2 (multiply-add = 2 ops)
        flops_reduce = C * Cr * S * S * 2 * M

        # 5 depthwise 3×3 convs: 5 * Cr * 3 * 3 * S * S * 2
        flops_deriv = 5 * Cr * 9 * S * S * 2 * M

        # GAP: Cr * S * S
        # MLP: Cr * (Cr//4) * 2 + (Cr//4) * Cr * 2
        flops_mlp = Cr * (Cr // 4) * 4 * M

        # Spatial attn: Cr * 1 * S * S * 2
        flops_spatial = Cr * S * S * 2 * M

        # Conv1x1 expand: Cr * C * S * S * 2
        flops_expand = Cr * C * S * S * 2 * M

        flops_total = (
            flops_reduce + flops_deriv + flops_mlp
            + flops_spatial + flops_expand
        )

        return {
            "conv_reduce_M":    round(flops_reduce, 4),
            "derivatives_M":    round(flops_deriv, 4),
            "channel_mlp_M":    round(flops_mlp, 4),
            "spatial_attn_M":   round(flops_spatial, 4),
            "conv_expand_M":    round(flops_expand, 4),
            "total_M":          round(flops_total, 4),
        }


# ======================================================================
# 便捷工厂函数
# ======================================================================

def cgam_c32(in_channels: int, **kwargs) -> CGAM:
    """C=32 场景（UNet 浅层），reduction=2 保证足够通道数"""
    kwargs.setdefault("reduction", 2)
    return CGAM(in_channels, **kwargs)


def cgam_c64(in_channels: int, **kwargs) -> CGAM:
    """C=64 场景（UNet 中层），reduction=4"""
    kwargs.setdefault("reduction", 4)
    return CGAM(in_channels, **kwargs)


def cgam_c256(in_channels: int, **kwargs) -> CGAM:
    """C=256 场景（DeepLabV3+ decoder / SegFormer），reduction=4"""
    kwargs.setdefault("reduction", 4)
    return CGAM(in_channels, **kwargs)


def cgam_c512(in_channels: int, **kwargs) -> CGAM:
    """C=512 场景（backbone 深层输出），reduction=8"""
    kwargs.setdefault("reduction", 8)
    return CGAM(in_channels, **kwargs)


# ======================================================================
# 单元测试入口
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CGAM 模块单元测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 测试不同通道配置 ----
    for C, name in [(32, "UNet浅层"), (64, "UNet中层"), (256, "DeepLabV3+"), (512, "Backbone深层")]:
        reduction = 4 if C >= 64 else 2
        model = CGAM(in_channels=C, reduction=reduction).to(device)
        x = torch.randn(2, C, 64, 64).to(device)

        # AMP 兼容性测试
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            y = model(x)
            y_attn, attn_dict = model(x, return_attention=True)

        print(f"\n--- {name} (C={C}, reduction={reduction}) ---")
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(y.shape)}")
        assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
        print(f"  Energy shape:     {attn_dict['energy'].shape}")
        print(f"  Channel attn:     {attn_dict['channel_attn'].shape}")
        print(f"  Spatial attn:     {attn_dict['spatial_attn'].shape}")

        params = model.count_parameters()
        print(f"  Trainable params: {params['trainable']:,}")
        print(f"  Fixed buffers:    {params['fixed_buffers']:,}")

        flops = CGAM.estimate_flops(C, spatial_size=64)
        print(f"  Total FLOPs:      {flops['total_M']:.4f} M")

    # ---- 梯度流测试 ----
    print("\n--- 梯度流测试 ---")
    model = CGAM(in_channels=64).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    x.requires_grad_(True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"  Input grad norm:  {x.grad.norm().item():.4f}")
    has_nan = torch.isnan(x.grad).any().item()
    print(f"  grad has NaN:     {has_nan}")
    assert not has_nan, "Gradient contains NaN!"

    print("\n✓ 所有测试通过")
