# USER-WRITTEN: agent 只提建议， 不改实现
# 例外：标注“agent 编写”的 norm、mlp 开关代码，经用户 2026-11-03、11-04 同意由 agent 加入
from dataclasses import dataclass

import torch
from torch import nn

from tlab.model.block import Block

# agent 编写（norm、mlp 开关）：检查开关取值、按配置名建最终归一化层
from tlab.model.layers import check_block_settings, make_norm


@dataclass
class GPTConfig:
    #  模型的5个超参数
    vocab_size: int  # 词表大小V
    block_size: int  # 最长序列长度
    d_model: int  # 每个 token 的向量长度 C
    n_layers: int  # block 层数 L
    n_heads: int  # 注意力头数 h
    # agent 编写（norm、mlp 开关）：M3 消融用，默认值与 M1/M2 相同
    norm: str = "layernorm"  # 归一化种类："layernorm" 或 "rmsnorm"
    norm_position: str = "pre"  # 归一化位置："pre"（子层前）或 "post"（加法后）
    mlp: str = "gelu"  # 前馈子层："gelu"（M1/M2 的 MLP）或 "swiglu"

    def __post_init__(self) -> None:
        # agent 编写（norm、mlp 开关）：建配置时就检查取值，0 层时也不会被悄悄放过
        check_block_settings(self.norm, self.norm_position, self.mlp)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 记下最长序列长度, forward 里要检查输入是否超长
        self.block_size = config.block_size
        # token 查表： 每个编号对应一个 C 维可学习向量， 形状 (V, C)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # 位置查表：第 i 行是第 i 个位置的可学习向量， 形状 (block_size, C)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        # 叠 n_layers 个block；必须用 ModuleList, 普通列表里的参数不会被登记
        # agent 编写（norm、mlp 开关）：把归一化的种类、位置和前馈子层的种类传给每个 block
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.d_model,
                    config.n_heads,
                    config.norm,
                    config.norm_position,
                    config.mlp,
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # 检查序列长度：位置表只有 block_size 行
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度{T}超过 block_size {self.block_size}"
        # 位置编号 0, 1, ..., T - 1, 和 idx 放在同一个设备上
        pos = torch.arange(T, device=idx.device)  # (T,)
        # token 向量加上位置向量，(T, C)会广播到每条序列
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        # 逐层通过 block， 每层的输出是下一层的输入
        for block in self.blocks:
            x = block(x)  # (B, T, C)
        # 最终归一化， 再映射成词表里每个字的分数
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)
        return logits


def count_parameters(model: nn.Module) -> int:
    # parameters() 会对共用的参数去重， weight tying 的那张表只算一次
    return sum(p.numel() for p in model.parameters())
