# USER-WRITTEN: agent 只提建议， 不改实现
# 例外：标注“agent 编写”的 norm 开关代码，经用户 2026-11-03 同意由 agent 加入

import torch
import torch.nn.functional as F
from torch import nn

from tlab.model.attention import CausalSelfAttention

# agent 编写（norm 开关）：检查开关取值、按配置名建归一化层
from tlab.model.layers import check_norm_settings, make_norm


class MLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # 放大层: C 维提升到 4C 维
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        # 缩回层: 4C 维降回 C 维, 残差相加要求形状和输入一致
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 放大
        h = self.fc(x)  # (B, T, 4C)
        # 非线性激活: 没有它， 两个 Linear 会合并成一个
        h = F.gelu(h)  # (B, T, 4C)
        # 缩回
        y = self.proj(h)  # (B, T, C)
        return y


class Block(nn.Module):
    # agent 编写（norm 开关）：多了 norm、norm_position 参数，默认值与 M1/M2 相同
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm: str = "layernorm",
        norm_position: str = "pre",
    ) -> None:
        super().__init__()
        # agent 编写（norm 开关）：记下归一化放在子层前（pre）还是加法后（post）
        check_norm_settings(norm, norm_position)
        self.norm_position = norm_position
        # agent 编写（norm 开关）：ln1、ln2 沿用 M1 的名字（改名会让旧存档载不进来），
        # 里面装的可能是 LayerNorm 或 RMSNorm
        # 注意力子层： 先归一化， 再让 token 之间交换信息
        self.ln1 = make_norm(norm, d_model)  # agent 编写（norm 开关）
        self.attn = CausalSelfAttention(d_model, n_heads)
        # MLP子层： 先归一化， 再让每个 token 各自加工
        self.ln2 = make_norm(norm, d_model)  # agent 编写（norm 开关）
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # agent 编写（norm 开关）：Post-LN 先把子层输出加回 x，再归一化（原论文的 Add & Norm）
        if self.norm_position == "post":
            x = self.ln1(x + self.attn(x))  # (B, T, C)
            x = self.ln2(x + self.mlp(x))  # (B, T, C)
            return x
        # 注意力子层 + 残差： 只把“改动量” 加回 x
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        # MLP子层 + 残差
        x = x + self.mlp(self.ln2(x))  # (B, T, C)
        return x
