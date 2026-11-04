# USER-WRITTEN: agent 只提建议， 不改实现
# 例外：标注“agent 编写”的 norm、mlp、pos_encoding 开关代码，经用户 2026-11-03、11-04 同意由 agent 加入
"""现代 Transformer 组件：RMSNorm（M3 第 1 步）；之后的 SwiGLU、RoPE 也写在这里"""

import torch
import torch.nn.functional as F
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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        # 门： C 维提升到 H 维， 激活后决定每个分量能通过多少
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # 内容： C 维提升到 H 维， 这一路不过激活函数
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # 缩回层： H 维降回 C 维， 残差相加要求形状和输入一致
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门： SiLU(z) = z * sigmoid(z)
        gate = F.silu(self.gate_proj(x))  # (B, T, H)
        # 内容
        content = self.up_proj(x)  # (B, T, H)
        # 门控: 逐元素相乘， 每个分量各乘各的， 再缩回C维
        return self.down_proj(gate * content)  # (B, T, C)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        # 分量要两两配对旋转， 维度必须是偶数
        assert head_dim % 2 == 0, "head_dim 必须是偶数"
        # 对每一对分量的转速： 第 0 对最快， 越往后越慢
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )  # (d/2,)
        # 登记成buffer： 跟着模型换设备， 但不是参数， 也不存进存档
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 位置编号 0, 1, ..., T - 1; x 的倒数第二维是 T
        T = x.size(-2)
        pos = torch.arange(T, device=x.device).float()  # (T,)
        # 每个位置，每一对分量要转的角度 = 位置 * 转速
        angles = pos[:, None] * self.inv_freq  # (T, d/2)
        # 前半段和后半对配对, 所以两段用同样的角度
        angles = torch.cat((angles, angles), dim=-1)  # (T, d)
        # 把向量对半分， 交换两段， 并给后半段取负
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]  # (B, h, T, d/2), 两段形状相同
        swapped = torch.cat((-x2, x1), dim=-1)  # (B, h, T, d)
        # 旋转: (T, d) 的 cos， sin 广播到 (B, h, T, d)
        return x * angles.cos() + swapped * angles.sin()  # (B, h, T, d)


# agent 编写（norm、mlp 开关）：配置里各开关的合法取值、取值检查，以及按名字建归一化层
NORMS = ("layernorm", "rmsnorm")
NORM_POSITIONS = ("pre", "post")
MLPS = ("gelu", "swiglu")
# agent 编写（pos_encoding 开关）：下面的取值表、检查函数和正弦位置编码
POS_ENCODINGS = ("learned", "sinusoidal", "rope")


def check_block_settings(norm: str, norm_position: str, mlp: str) -> None:
    """取值拼错时报错；训练配置、GPTConfig、Block 创建时都调用，不会被悄悄当成别的结构。"""
    if mlp not in MLPS:
        raise ValueError(f"mlp 应为 {MLPS} 之一，收到 {mlp!r}")
    if norm not in NORMS:
        raise ValueError(f"norm 应为 {NORMS} 之一，收到 {norm!r}")
    if norm_position not in NORM_POSITIONS:
        raise ValueError(
            f"norm_position 应为 {NORM_POSITIONS} 之一，收到 {norm_position!r}"
        )


def check_pos_encoding(pos_encoding: str, rope_theta: float) -> None:
    """取值不对时报错；训练配置和 GPTConfig 创建时调用。"""
    # theta 是转速公式的底数，不大于 0 时角度不是有限数
    if rope_theta <= 0:
        raise ValueError(f"rope_theta 必须大于 0，收到 {rope_theta!r}")
    if pos_encoding not in POS_ENCODINGS:
        raise ValueError(
            f"pos_encoding 应为 {POS_ENCODINGS} 之一，收到 {pos_encoding!r}"
        )


def sinusoidal_positions(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    """原论文的正弦位置编码：不用训练，序列多长都能算。"""
    assert d_model % 2 == 0, "正弦位置编码的 sin、cos 成对出现，d_model 必须是偶数"
    pos = torch.arange(T, device=device).float()[:, None]  # (T, 1)
    # 分母 10000^(2i/C)，i 是第几对
    div = 10000.0 ** (
        torch.arange(0, d_model, 2, device=device).float() / d_model
    )  # (C/2,)
    pe = torch.zeros(T, d_model, device=device)  # (T, C)
    # 偶数维用 sin，奇数维用 cos
    pe[:, 0::2] = torch.sin(pos / div)  # (T, C/2)
    pe[:, 1::2] = torch.cos(pos / div)  # (T, C/2)
    return pe


def make_norm(kind: str, d_model: int) -> nn.Module:
    """按名字建归一化层：layernorm 是 M1/M2 用的 nn.LayerNorm，rmsnorm 是上面的 RMSNorm。"""
    if kind == "layernorm":
        return nn.LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"norm 应为 {NORMS} 之一，收到 {kind!r}")
