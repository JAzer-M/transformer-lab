"""完整验证集评估：把 token 序列切成首尾相接、互不重叠的窗口，每个 token 恰好预测一次。

M3 消融的最终比较用它。训练中的 estimate_loss 只按种子随机抽 eval_iters 个 batch
（约占验证集 6%），不同种子抽到的子集不同，带来约 0.02 的抽样误差
（见 PROGRESS 2026-11-03 14:03 记录）。
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tlab.data.batch import read_window
from tlab.data.loader import load_splits
from tlab.train.utils import autocast_context, git_commit, load_model


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    data: torch.Tensor | np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int]:
    """返回 (按 token 平均的交叉熵, 被预测的 token 数)；被预测的是 data[1:] 的每一个。"""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in _window_batches(data, block_size, batch_size):
        x, y = x.to(device), y.to(device)  # (B, T), (B, T)
        # 前向与训练时一样：GPU 上 bf16，CPU/MPS 上 fp32
        with autocast_context(device):
            logits = model(x)  # (B, T, V)
        # 先按 token 加总：最后一个窗口更短，加总后再除以 token 数才是按 token 的平均
        # (B*T, V) 与 (B*T,) → () 的总和；bf16 的 logits 先转回 fp32 再算
        total += F.cross_entropy(
            logits.flatten(0, 1).float(), y.flatten(), reduction="sum"
        ).item()
        count += y.numel()
    model.train(was_training)
    return total / count, count


def _window_batches(
    data: torch.Tensor | np.ndarray, block_size: int, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """依次给出一批批 (x, y)：第 k 个窗口输入 data[kT : kT+T]、预测 data[kT+1 : kT+T+1]。

    满窗口每 batch_size 个叠成一批；最后不满一窗的部分单独成一批。
    """
    n_targets = len(data) - 1
    n_full = n_targets // block_size
    for first in range(0, n_full, batch_size):
        last = min(first + batch_size, n_full)
        starts = [k * block_size for k in range(first, last)]
        x = torch.stack([read_window(data, s, block_size) for s in starts])  # (B, T)
        y = torch.stack(
            [read_window(data, s + 1, block_size) for s in starts]
        )  # (B, T)
        yield x, y
    tail = n_targets - n_full * block_size
    if tail:
        s = n_full * block_size
        x = read_window(data, s, tail)[None]  # (1, tail)
        y = read_window(data, s + 1, tail)[None]  # (1, tail)
        yield x, y


def evaluate_checkpoint(
    ckpt_path: Path, batch_size: int, device: torch.device
) -> dict[str, Any]:
    """按存档重建模型，在它训练时用的验证集上做完整评估；返回写进 full_val.json 的内容。"""
    # 结构按存档里的配置重建（与 sample.py 相同），参数严格载入
    model, ckpt = load_model(ckpt_path, device)
    config = ckpt["config"]
    _, val_data = load_splits(Path(config["data_dir"]))  # (N_val,)
    loss, n_tokens = evaluate_full(
        model, val_data, config["block_size"], batch_size, device
    )
    return {
        "val_loss": loss,
        "n_tokens": n_tokens,
        "block_size": config["block_size"],
        "ckpt": str(ckpt_path),
        "ckpt_step": ckpt["step"],
        "device": str(device),
        "eval_commit": git_commit(),
    }
