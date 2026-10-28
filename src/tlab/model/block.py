# USER-WRITTEN: agent 只提建议， 不改实现

import torch
import torch.nn.functional as F
from torch import nn

from tlab.model.attention import CausalSelfAttention


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
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # 注意力子层： 先归一化， 再让 token 之间交换信息
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        # MLP子层： 先归一化， 再让每个 token 各自加工
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层 + 残差： 只把“改动量” 加回 x
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        # MLP子层 + 残差
        x = x + self.mlp(self.ln2(x))  # (B, T, C)
        return x
