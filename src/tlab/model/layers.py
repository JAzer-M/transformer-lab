# USER-WRITTEN: agent 只提建议， 不改实现
# 例外：标注“agent 编写”的 norm 开关代码，经用户 2026-11-03 同意由 agent 加入
"""现代 Transformer 组件：RMSNorm（M3 第 1 步）；之后的 SwiGLU、RoPE 也写在这里"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        # 防止除以 0 的小常数， 加在根号里面； 默认值与 Qwen3 相同
        self.eps = eps
        # 逐分量的可学习缩放， 形状(C,), 初始全1； 没有bias (LayerNorm 有)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 token 自己的均方： 只在最后一维 C 上平均， keepdim 留下这一维以便广播
        ms = x.pow(2).mean(dim=-1, keepdim=True)  # (B, T, 1)
        # 除以均方根: rsqrt(a) 就是 1/√a, eps 加在根号里面
        x_normed = x * torch.rsqrt(ms + self.eps)  # (B, T, C)
        # 逐分量乘以缩放： weight(C,) 广播到 (B, T, C)
        return x_normed * self.weight  # (B, T, C)


# agent 编写（norm 开关）：配置里两个开关的合法取值、取值检查，以及按名字建归一化层
NORMS = ("layernorm", "rmsnorm")
NORM_POSITIONS = ("pre", "post")


def check_norm_settings(norm: str, norm_position: str) -> None:
    """取值拼错时报错；训练配置、GPTConfig、Block 创建时都调用，不会被悄悄当成别的结构。"""
    if norm not in NORMS:
        raise ValueError(f"norm 应为 {NORMS} 之一，收到 {norm!r}")
    if norm_position not in NORM_POSITIONS:
        raise ValueError(
            f"norm_position 应为 {NORM_POSITIONS} 之一，收到 {norm_position!r}"
        )


def make_norm(kind: str, d_model: int) -> nn.Module:
    """按名字建归一化层：layernorm 是 M1/M2 用的 nn.LayerNorm，rmsnorm 是上面的 RMSNorm。"""
    if kind == "layernorm":
        return nn.LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"norm 应为 {NORMS} 之一，收到 {kind!r}")
