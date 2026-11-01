"""学习率调度：先线性 warmup，再保持不变（constant）或按余弦曲线下降（cosine）。"""

import math

from tlab.train.config import TrainConfig


def get_lr(step: int, config: TrainConfig) -> float:
    """第 step 步（从 0 数）该用的学习率；lr_schedule 已在建配置时检查过。"""
    lr, warmup = config.learning_rate, config.warmup_iters

    # warmup：第 0 步是 lr / warmup，第 warmup - 1 步升到 lr；两种调度都适用
    if step < warmup:
        return lr * (step + 1) / warmup
    if config.lr_schedule == "constant":
        return lr

    # cosine：progress 在第 warmup 步为 0、在循环最后一步（max_iters - 1）为 1，
    # 系数 0.5 · (1 + cos(π · progress)) 从 1 平滑降到 0，学习率从 lr 降到 min_lr
    progress = (step - warmup) / max(1, config.max_iters - 1 - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_lr + coeff * (lr - config.min_lr)
