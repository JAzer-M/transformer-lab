# USER-WRITTEN: agent 只提建议，不改实现。
# 例外：标注“agent 编写”的 pos_encoding 开关与 KV cache 代码，经用户 2026-11-04、11-05 同意由 agent 加入

import torch
from torch import nn

# agent 编写（pos_encoding 开关）：RoPE 的旋转
from tlab.model.layers import RotaryEmbedding


# agent 编写（KV cache）：生成时每层一个；g 是 k、v 的组数，T_past 是已经存下的 token 数
class KVCache:
    def __init__(self) -> None:
        # 存旋转后、复制前的 k、v：旧 token 的旋转角度不会变，复制前只有 g 组，GQA 才省缓存
        self.k: torch.Tensor | None = None  # (B, g, T_past, d)
        self.v: torch.Tensor | None = None  # (B, g, T_past, d)

    @property
    def length(self) -> int:
        """已经存了多少个 token。"""
        return 0 if self.k is None else self.k.size(-2)

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把新 token 的 k、v 接到末尾，返回过去加现在的全部 k、v。"""
        self.k = k if self.k is None else torch.cat((self.k, k), dim=-2)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=-2)
        return self.k, self.v  # (B, g, T_past + T, d)


class CausalSelfAttention(nn.Module):
    # agent 编写（pos_encoding 开关）：多了 rope_theta 参数，None 表示不用 RoPE
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_theta: float | None = None,
        n_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        # 检查 d_model 能否被头数整除
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self._head_dim = d_model // n_heads

        # k, v 的组数 g： 不指定时每个头一组， 即MHA 多头注意力
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads

        # 每组要分到同样多的头； 先判断大于0， 否则 n_heads % 0 会直接报错
        assert self.n_kv_heads > 0 and n_heads % self.n_kv_heads == 0, (
            "n_heads 必须能被 n_kv_heads 整除"
        )

        # 每组要复制几份， 才够每个头用一份
        self._n_repeat = n_heads // self.n_kv_heads

        # k, v 的总维度： g组， 每组 d 维
        kv_dim = self.n_kv_heads * self._head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # agent 编写（pos_encoding 开关）：没有参数，也不进存档
        self.rope = (
            None if rope_theta is None else RotaryEmbedding(self._head_dim, rope_theta)
        )

    # agent 编写（KV cache）：cache 只在生成时传，不传时和以前完全一样
    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, g*d)
        v = self.v_proj(x)  # (B, T, g*d)

        B, T, C = x.shape
        g = self.n_kv_heads
        q = q.view(B, T, self.n_heads, self._head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, g, self._head_dim).transpose(1, 2)  # (B, g, T, d)
        v = v.view(B, T, g, self._head_dim).transpose(1, 2)  # (B, g, T, d)

        # agent 编写（KV cache）：缓存里已有的 token 数，新 token 的位置从这里数起
        T_past = 0 if cache is None else cache.length
        T_all = T_past + T

        # agent 编写（pos_encoding 开关）：RoPE 只转 q、k；v 是被读走的内容，不带位置
        if self.rope is not None:
            q = self.rope(q, offset=T_past)  # (B, h, T, d)
            k = self.rope(k, offset=T_past)  # (B, g, T, d)

        # agent 编写（KV cache）：接上旧 token 的 k、v；q 不用存，新 token 只需要自己的 q
        if cache is not None:
            k, v = cache.update(k, v)  # (B, g, T_all, d)

        # 把每组复制给它下面的头: k0 k0 k0 k1 k1 k1, 同组头排在一起
        k = k.repeat_interleave(self._n_repeat, dim=1)  # (B, h, T_all, d)
        v = v.repeat_interleave(self._n_repeat, dim=1)  # (B, h, T_all, d)
        # agent 编写（KV cache）：下面的形状注释里 T_all 是列数；不用缓存时 T_all = T
        k_t = k.transpose(-2, -1)  # 交换最后两个维度 因为原来是(B, h, T_all, d)
        # 计算q乘以k的转置
        scores = q @ k_t  # (B, h, T, T_all)
        scores = scores / (self._head_dim**0.5)  # (B, h, T, T_all)

        # 掩码操作
        # agent 编写（KV cache）：第 i 个新 token 的绝对位置是 T_past + i，能看到第 0 到
        # T_past + i 列，所以从第 T_past + 1 条对角线起遮住；T_past = 0 时和以前相同
        mask = torch.triu(
            torch.ones(T, T_all, dtype=torch.bool, device=x.device),
            diagonal=T_past + 1,
        )  # (T, T_all)
        scores = scores.masked_fill(mask, float("-inf"))  # (B, h, T, T_all)

        # 计算attention
        attn = torch.softmax(
            scores, dim=-1
        )  # (B, h, T, T_all) 固定其他下标, 只允许列进行变化
        z = attn @ v  # (B, h, T, d)
        # 合头: 把头维换回去， 再把各头的 d 维拼回 C 维
        z = z.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.out_proj(z)  # (B, T, C)
        return y
