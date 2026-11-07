# USER-WRITTEN: agent 只提建议， 不改实现
# 例外：标注“agent 编写”的 norm、mlp、pos_encoding、n_kv_heads、head_dim、qk_norm、mlp_hidden_dim 开关与 KV cache 代码，经用户 2026-11-03、11-04、11-05、11-07 同意由 agent 加入

import torch
import torch.nn.functional as F
from torch import nn

# agent 编写（KV cache）：KVCache 只用于标注类型
from tlab.model.attention import CausalSelfAttention, KVCache

# agent 编写（各开关）：检查开关取值、按配置名建归一化层、SwiGLU
from tlab.model.layers import (
    MLPS,
    SwiGLU,
    check_block_settings,
    check_qwen3_switches,
    make_norm,
)


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


# agent 编写（mlp 开关）：按配置名建前馈子层。SwiGLU 的隐藏维度默认取 8C/3：
# C 能被 3 整除时，3·C·(8C/3) = 8C²，参数量与 GELU MLP 相同，比较才公平。
# 载入别人的权重时要跟着对方，比如 Qwen3-0.6B 是 3072，所以也可以指定
def make_mlp(kind: str, d_model: int, hidden_dim: int | None = None) -> nn.Module:
    if kind == "gelu":
        return MLP(d_model)
    if kind == "swiglu":
        return SwiGLU(d_model, 8 * d_model // 3 if hidden_dim is None else hidden_dim)
    raise ValueError(f"mlp 应为 {MLPS} 之一，收到 {kind!r}")


class Block(nn.Module):
    # agent 编写（各开关）：多了 norm、norm_position、mlp、rope_theta、n_kv_heads、
    # head_dim、qk_norm、mlp_hidden_dim 参数，默认值与 M1/M2 相同
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm: str = "layernorm",
        norm_position: str = "pre",
        mlp: str = "gelu",
        rope_theta: float | None = None,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        qk_norm: bool = False,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        # agent 编写（各开关）：检查取值；记下归一化放在子层前（pre）还是加法后（post）
        check_block_settings(norm, norm_position, mlp)
        check_qwen3_switches(mlp, head_dim, qk_norm, mlp_hidden_dim)
        self.norm_position = norm_position
        # agent 编写（norm 开关）：ln1、ln2 沿用 M1 的名字（改名会让旧存档载不进来），
        # 里面装的可能是 LayerNorm 或 RMSNorm
        # 注意力子层： 先归一化， 再让 token 之间交换信息
        self.ln1 = make_norm(norm, d_model)  # agent 编写（norm 开关）
        # agent 编写（各开关）：rope_theta 为 None 时不用 RoPE，n_kv_heads 为 None 时每个头
        # 一组（MHA），head_dim 为 None 时是 C/h
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            rope_theta=rope_theta,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
        )
        # MLP子层： 先归一化， 再让每个 token 各自加工
        self.ln2 = make_norm(norm, d_model)  # agent 编写（norm 开关）
        # agent 编写（mlp、mlp_hidden_dim 开关）
        self.mlp = make_mlp(mlp, d_model, mlp_hidden_dim)

    # agent 编写（KV cache）：cache 只在生成时传，原样交给注意力
    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        # agent 编写（norm 开关）：Post-LN 先把子层输出加回 x，再归一化（原论文的 Add & Norm）
        if self.norm_position == "post":
            x = self.ln1(x + self.attn(x, cache))  # (B, T, C)
            x = self.ln2(x + self.mlp(x))  # (B, T, C)
            return x
        # 注意力子层 + 残差： 只把“改动量” 加回 x
        x = x + self.attn(self.ln1(x), cache)  # (B, T, C)
        # MLP子层 + 残差
        x = x + self.mlp(self.ln2(x))  # (B, T, C)
        return x
