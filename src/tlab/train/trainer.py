# USER-WRITTEN: agent 只提建议， 不改实现
"""训练主循环"""

import torch
from torch import nn

from tlab.data.char import get_batch
from tlab.train.config import TrainConfig
from tlab.train.utils import (
    RunLogger,
    autocast_context,
    cross_entropy_loss,
    estimate_loss,
)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    logger: RunLogger,
) -> list[float]:
    # 训练用的随机数生成器： 固定种子， 每次运行取到的 batch 顺序都一样
    generator = torch.Generator().manual_seed(config.seed)
    losses = []
    model.train()

    for step in range(config.max_iters):
        # 定期评估： 第零步， 每 eval_interval 步， 最后一步
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            val = estimate_loss(model, train_data, val_data, config, device)
            logger.log_eval(step, val)

        # 取一批数据
        x, y = get_batch(
            train_data, config.batch_size, config.block_size, generator, device
        )  # (B, T), (B, T)

        # 前向： 在 GPU 上用bf 16 计算 logits 和 loss
        with autocast_context(device):
            logits = model(x)  # (B, T, V)
            loss = cross_entropy_loss(logits, y)  # ()

        # 反向与更新： 清空旧梯度 -> 算新梯度 -> 更新参数
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 记录这一步的 loss (转成 float， 不保留计算图)
        losses.append(loss.item())

    return losses
