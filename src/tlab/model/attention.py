# USER-WRITTEN: agent 只提建议，不改实现。
# 例外：标注“agent 编写”的 pos_encoding 开关代码，经用户 2026-11-04 同意由 agent 加入

import torch
from torch import nn

# agent 编写（pos_encoding 开关）：RoPE 的旋转
from tlab.model.layers import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    # agent 编写（pos_encoding 开关）：多了 rope_theta 参数，None 表示不用 RoPE
    def __init__(
        self, d_model: int, n_heads: int, rope_theta: float | None = None
    ) -> None:
        super().__init__()
        # 检查 d_model 能否被头数整除
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self._head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # agent 编写（pos_encoding 开关）：没有参数，也不进存档
        self.rope = (
            None if rope_theta is None else RotaryEmbedding(self._head_dim, rope_theta)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # 拆头: 把 C 维切成 n_heads 段， 再把头挪到 T 前面
        B, T, C = x.shape
        q = q.view(B, T, self.n_heads, self._head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_heads, self._head_dim).transpose(1, 2)  # (B, h, T, d)
        v = v.view(B, T, self.n_heads, self._head_dim).transpose(1, 2)  # (B, h, T, d)

        # agent 编写（pos_encoding 开关）：RoPE 只转 q、k；v 是被读走的内容，不带位置
        if self.rope is not None:
            q = self.rope(q)  # (B, h, T, d)
            k = self.rope(k)  # (B, h, T, d)

        k_t = k.transpose(-2, -1)  # 交换最后两个维度 因为原来是(B, h, T, d)
        # 计算q乘以k的转置
        scores = q @ k_t  # (B, h, T, T)
        scores = scores / (self._head_dim**0.5)  # (B, h, T, T)

        # 掩码操作
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )  # (T, T)
        scores = scores.masked_fill(mask, float("-inf"))  # (B, h, T, T)

        # 计算attention
        attn = torch.softmax(
            scores, dim=-1
        )  # (B, h, T, T) 固定其他下标, 只允许列进行变化
        z = attn @ v  # (B, h, T, d)
        # 合头: 把头维换回去， 再把各头的 d 维拼回 C 维
        z = z.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.out_proj(z)  # (B, T, C)
        return y
