"""自回归生成：每次根据已有文本预测下一个 token，抽样后接到末尾，重复。"""

import torch
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """idx 是开头 (B, T)；返回 (B, T + max_new_tokens)。

    temperature < 1 让分布更尖（更保守），> 1 更平（更随机）；
    top_k 只在分数最高的 k 个 token 里抽样。
    """
    assert temperature > 0, "temperature 必须大于 0"
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        # 只把最后 block_size 个 token 送进模型：可学习位置表只有这么多行；
        # 正弦和 RoPE 没有这个限制，但也按训练时的长度截断，与训练时的上下文一致
        idx_cond = idx[:, -model.block_size :]  # (B, T')，T' ≤ block_size
        # 只需要最后一个位置的分数，它预测的是“下一个 token”
        logits = model(idx_cond)[:, -1, :]  # (B, V)
        # 温度：先缩放分数，再做 softmax
        logits = logits / temperature  # (B, V)

        if top_k is not None:
            # 第 k 大的分数作为门槛，低于它的全部设为 -inf，softmax 后概率为 0
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[:, -1:]  # (B, 1)
            logits = logits.masked_fill(logits < kth, float("-inf"))  # (B, V)

        # 分数 → 概率 → 按概率抽一个 token
        probs = torch.softmax(logits, dim=-1)  # (B, V)
        next_id = torch.multinomial(probs, 1, generator=generator)  # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)  # (B, T + 1)

    model.train(was_training)
    return idx
