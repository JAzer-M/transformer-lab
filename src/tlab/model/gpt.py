# USER-WRITTEN: agent 只提建议， 不改实现
# 例外：标注“agent 编写”的 norm、mlp、pos_encoding、n_kv_heads、head_dim、qk_norm、mlp_hidden_dim 开关与 KV cache 代码，经用户 2026-11-03、11-04、11-05、11-07 同意由 agent 加入
from dataclasses import dataclass

import torch
from torch import nn

# agent 编写（KV cache）
from tlab.model.attention import KVCache
from tlab.model.block import Block

# agent 编写（各开关）：检查开关取值、按配置名建最终归一化层、正弦位置编码
from tlab.model.layers import (
    check_block_settings,
    check_kv_heads,
    check_pos_encoding,
    check_qwen3_switches,
    make_norm,
    sinusoidal_positions,
)


@dataclass
class GPTConfig:
    #  模型的5个超参数
    vocab_size: int  # 词表大小V
    block_size: int  # 最长序列长度
    d_model: int  # 每个 token 的向量长度 C
    n_layers: int  # block 层数 L
    n_heads: int  # 注意力头数 h
    # agent 编写（各开关）：M3 消融用，默认值与 M1/M2 相同
    norm: str = "layernorm"  # 归一化种类："layernorm" 或 "rmsnorm"
    norm_position: str = "pre"  # 归一化位置："pre"（子层前）或 "post"（加法后）
    mlp: str = "gelu"  # 前馈子层："gelu"（M1/M2 的 MLP）或 "swiglu"
    pos_encoding: str = "learned"  # 位置编码："learned"、"sinusoidal" 或 "rope"
    rope_theta: float = 10000.0  # RoPE 转速公式里的底数，只在 "rope" 时有用
    n_kv_heads: int | None = None  # k、v 的组数；None 表示等于 n_heads（MHA）
    head_dim: int | None = None  # 每个头的维度；None 表示 d_model // n_heads
    qk_norm: bool = False  # q、k 拆头后各做一次 RMSNorm（Qwen3 的做法）
    mlp_hidden_dim: int | None = None  # SwiGLU 的隐藏维度；None 表示 8C//3

    def __post_init__(self) -> None:
        # agent 编写（各开关）：建配置时就检查取值，0 层时也不会被悄悄放过
        check_block_settings(self.norm, self.norm_position, self.mlp)
        check_pos_encoding(self.pos_encoding, self.rope_theta, self.head_dim)
        check_kv_heads(self.n_heads, self.n_kv_heads)
        check_qwen3_switches(self.mlp, self.head_dim, self.qk_norm, self.mlp_hidden_dim)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 记下最长序列长度, forward 里要检查输入是否超长
        self.block_size = config.block_size
        # token 查表： 每个编号对应一个 C 维可学习向量， 形状 (V, C)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # agent 编写（pos_encoding 开关）：只有 learned 才有位置表；forward 按种类走分支
        self.pos_encoding = config.pos_encoding
        self.d_model = config.d_model
        if config.pos_encoding == "learned":
            # 位置查表：第 i 行是第 i 个位置的可学习向量， 形状 (block_size, C)
            self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        # 叠 n_layers 个block；必须用 ModuleList, 普通列表里的参数不会被登记
        # agent 编写（各开关）：把归一化的种类、位置和前馈子层的种类传给每个 block
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.d_model,
                    config.n_heads,
                    # agent 编写（各开关）：开关多了容易传错位置，都写参数名
                    norm=config.norm,
                    norm_position=config.norm_position,
                    mlp=config.mlp,
                    # 只有 rope 才在注意力里旋转
                    rope_theta=(
                        config.rope_theta if config.pos_encoding == "rope" else None
                    ),
                    n_kv_heads=config.n_kv_heads,
                    head_dim=config.head_dim,
                    qk_norm=config.qk_norm,
                    mlp_hidden_dim=config.mlp_hidden_dim,
                )
                for _ in range(config.n_layers)
            ]
        )
        # 最终 LayerNorm: Pre-LN 的残差主干道上没有归一化， 输出前补一次
        # agent 编写（norm 开关）：名字沿用 ln_f（旧存档靠它载入），里面可能是 LayerNorm、
        # RMSNorm，或者 Post-LN 时什么都不做的 Identity（每层已以归一化结尾）
        if config.norm_position == "pre":
            self.ln_f = make_norm(config.norm, config.d_model)
        else:
            self.ln_f = nn.Identity()
        # 输出层： 把 C 维向量变成词表里每个字的分数， weight 形状 (V, C)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying: 输出层直接指向 token 查表的同一个 (V, C) 参数， 不是复制
        self.lm_head.weight = self.tok_emb.weight
        # 遍历所有子模块， 把 Linear 和 Embedding 的权重按 N(0, 0.02) 重新初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # 只处理 Linear 和 embedding: LayerNorm 保持 weight = 1, bias = 0
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    # agent 编写（KV cache）
    def new_cache(self) -> list[KVCache]:
        """生成前建一份空缓存，每层一个。"""
        return [KVCache() for _ in self.blocks]

    # agent 编写（KV cache）：cache 只在生成时传，这时 idx 只有新 token；不传时和以前一样
    def forward(
        self, idx: torch.Tensor, cache: list[KVCache] | None = None
    ) -> torch.Tensor:
        B, T = idx.shape
        # agent 编写（KV cache）：缓存里已有的 token 数，新 token 的位置从这里数起
        T_past = cache[0].length if cache else 0
        # agent 编写（pos_encoding 开关）：正弦和 RoPE 没有位置表，也就没有长度上限
        if self.pos_encoding == "sinusoidal":
            # 原论文做法：token 向量乘 √C，否则初始标准差 0.02 的 token 向量
            # 会被均方根约 0.7 的位置向量盖住
            pe = sinusoidal_positions(T, self.d_model, idx.device, T_past)  # (T, C)
            x = self.tok_emb(idx) * self.d_model**0.5 + pe  # (B, T, C)
        elif self.pos_encoding == "rope":
            # 位置信息在每层注意力里给，入口只有 token 向量
            x = self.tok_emb(idx)  # (B, T, C)
        else:
            # agent 编写（KV cache）：下面的长度检查和位置编号都算上缓存里的 token；
            # 不用缓存时 T_past = 0，和以前相同
            T_all = T_past + T
            # 检查序列长度：位置表只有 block_size 行
            assert T_all <= self.block_size, (
                f"序列长度{T_all}超过 block_size {self.block_size}"
            )
            # 位置编号 T_past, ..., T_all - 1（不用缓存时是 0, ..., T - 1），和 idx 放在同一个设备上
            pos = torch.arange(T_past, T_all, device=idx.device)  # (T,)
            # token 向量加上位置向量，(T, C)会广播到每条序列
            x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        # 逐层通过 block， 每层的输出是下一层的输入
        # agent 编写（KV cache）：每层用自己的缓存；不用缓存时每层都传 None
        caches = cache if cache is not None else [None] * len(self.blocks)
        for block, layer_cache in zip(self.blocks, caches, strict=True):
            x = block(x, layer_cache)  # (B, T, C)
        # 最终归一化， 再映射成词表里每个字的分数
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)
        return logits


def count_parameters(model: nn.Module) -> int:
    # parameters() 会对共用的参数去重， weight tying 的那张表只算一次
    return sum(p.numel() for p in model.parameters())
