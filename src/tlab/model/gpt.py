# USER-WRITTEN: agent 只提建议， 不改实现
from dataclasses import dataclass

import torch
from torch import nn

from tlab.model.block import Block


@dataclass
class GPTConfig:
    #  模型的5个超参数
    vocab_size: int  # 词表大小V
    block_size: int  # 最长序列长度
    d_model: int  # 每个 token 的向量长度 C
    n_layers: int  # block 层数 L
    n_heads: int  # 注意力头数 h


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
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.n_heads) for _ in range(config.n_layers)]
        )
        # 最终 LayerNorm: Pre-LN 的残差主干道上没有归一化， 输出前补一次
        self.ln_f = nn.LayerNorm(config.d_model)
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
