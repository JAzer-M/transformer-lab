"""完整验证集评估与外推评测：把 token 序列切成首尾相接、互不重叠的窗口。

完整评估里每个 token 恰好预测一次，M3 消融的最终比较用它。训练中的 estimate_loss 只按种子随机抽 eval_iters 个 batch
（约占验证集 6%），不同种子抽到的子集不同，带来约 0.02 的抽样误差。
外推评测用 2 倍训练长度的窗口，
前后半段分开报，见 evaluate_extrapolation。
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


@torch.no_grad()
def evaluate_extrapolation(
    model: nn.Module,
    data: torch.Tensor | np.ndarray,
    length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """用长度为 length 的窗口打分，前半段和后半段位置的 loss 分开算。

    A1 的外推评测：length 是训练长度的 2 倍时，后半段的位置都超出了训练时见过的范围。
    只算满窗口，最后不满一窗的部分没有完整的后半段。
    """
    assert length % 2 == 0, "外推窗口长度必须是偶数，前后半段才一样长"
    was_training = model.training
    model.eval()
    half = length // 2
    first_sum, second_sum, windows = 0.0, 0.0, 0
    for x, y in _window_batches(data, length, batch_size):
        if y.size(1) != length:
            continue
        x, y = x.to(device), y.to(device)  # (B, T), (B, T)
        with autocast_context(device):
            logits = model(x)  # (B, T, V)
        # 每个位置各算一个 loss，再按位置分成两段加总
        per_token = F.cross_entropy(
            logits.flatten(0, 1).float(), y.flatten(), reduction="none"
        ).view(y.shape)  # (B, T)
        first_sum += per_token[:, :half].sum().item()  # (B, T/2) 加总成一个数
        second_sum += per_token[:, half:].sum().item()  # (B, T/2) 加总成一个数
        windows += y.size(0)
    model.train(was_training)
    if windows == 0:
        raise ValueError(
            f"数据只有 {len(data)} 个 token，不够一个长度 {length} 的满窗口"
        )
    n_tokens = windows * half
    return {
        "first_half_loss": first_sum / n_tokens,
        "second_half_loss": second_sum / n_tokens,
        "n_tokens_per_half": n_tokens,
    }


def extrapolate_checkpoint(
    ckpt_path: Path, batch_size: int, device: torch.device
) -> dict[str, Any]:
    """在 2 倍训练长度上评测存档；返回写进 extrapolation.json 的内容。"""
    model, ckpt = load_model(ckpt_path, device)
    config = ckpt["config"]
    block_size = config["block_size"]
    if model.pos_encoding == "learned":
        raise ValueError(
            f"可学习位置表只有 {block_size} 行，超出训练长度的位置没有向量，不能做外推评测"
        )
    _, val_data = load_splits(Path(config["data_dir"]))  # (N_val,)
    length = 2 * block_size
    result = evaluate_extrapolation(model, val_data, length, batch_size, device)
    return {
        **result,
        "pos_encoding": model.pos_encoding,
        "length": length,
        "block_size": block_size,
        "ckpt": str(ckpt_path),
        "ckpt_step": ckpt["step"],
        "device": str(device),
        "eval_commit": git_commit(),
    }
