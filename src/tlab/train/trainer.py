"""训练主循环"""

# 断点续训：checkpoint 参数的类型标注要用
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from tlab.data.char import get_batch
from tlab.train.config import TrainConfig
from tlab.train.schedule import get_lr
from tlab.train.utils import (
    RunLogger,
    autocast_context,
    cross_entropy_loss,
    estimate_loss,
)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor | np.ndarray,
    val_data: torch.Tensor | np.ndarray,
    config: TrainConfig,
    device: torch.device,
    logger: RunLogger,
    # 断点续训：从第几步开始、外部传入的生成器、存档函数
    start_step: int = 0,
    generator: torch.Generator | None = None,
    checkpoint: Callable[[int], None] | None = None,
) -> list[float]:
    # 断点续训：续训时 run.py 传入已恢复状态的生成器，否则在这里新建
    if generator is None:
        # 训练用的随机数生成器： 固定种子， 每次运行取到的 batch 顺序都一样
        generator = torch.Generator().manual_seed(config.seed)
    losses = []
    model.train()

    # 断点续训：续训时从 start_step 开始，而不是从 0 开始
    for step in range(start_step, config.max_iters):
        # 设定这一步的学习率： 按调度算出来， 写进 optimizer 的每一组参数
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 定期评估： 第零步， 每 eval_interval 步， 最后一步
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            val = estimate_loss(model, train_data, val_data, config, device)
            logger.log_eval(step, val, lr)

        # 清空上一步的梯度： 每一步只清一次， 下面几个 micro batch 的梯度要攒在一起
        optimizer.zero_grad(set_to_none=True)

        # 梯度累积： 连续做 grad_accum_steps 次前向和反向， 梯度自动加在一起
        step_loss = 0.0
        for _ in range(config.grad_accum_steps):
            # 取一个 micro batch
            x, y = get_batch(
                train_data, config.batch_size, config.block_size, generator, device
            )  # (B, T) (B, T)

            # 前向： 在 GPU 上用bf 16 计算 logits 和 loss. loss 除以累计次数， K份加起来才是平均
            with autocast_context(device):
                logits = model(x)  # (B, T, V)
                loss = cross_entropy_loss(logits, y) / config.grad_accum_steps  # ()

            # 反向： 这次算出的梯度加到每个参数已有的.grad上
            loss.backward()
            step_loss += loss.item()

        # 梯度裁剪： 所有梯度合起来的长度超过 grad_clip时， 整体按比例缩小
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # 更新参数： 攒够之后只更新一次
        optimizer.step()

        # 记录这一步的 loss (转成 float， 不保留计算图)
        losses.append(step_loss)

        # 断点续训：每 ckpt_interval 步存一次档；存的是“下一步”的编号
        if checkpoint is not None and config.ckpt_interval > 0:
            if (step + 1) % config.ckpt_interval == 0:
                checkpoint(step + 1)

    return losses
