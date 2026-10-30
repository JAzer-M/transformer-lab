"""训练主循环用到的外围工具：混合精度、评估、日志、checkpoint、运行信息。"""

import contextlib
import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from tlab.data.char import get_batch
from tlab.train.config import TrainConfig


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    # GPU 上统一用 bf16 autocast（项目规范）；CPU/MPS 保持 fp32
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # (B, T, V) 与 (B, T) 摊平成 (B*T, V) 与 (B*T,)，每个位置算一次交叉熵再取平均
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))  # ()


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """在训练集和验证集上各取 eval_iters 个 batch，返回平均 loss。"""
    was_training = model.training
    # eval 模式关闭 dropout 等训练专用行为（目前模型没有，但保持习惯）
    model.eval()
    # 每次评估都用同一个种子取 batch，不同步数的评估结果才可比，也不扰动训练的随机流
    generator = torch.Generator().manual_seed(config.seed + 1)
    losses = {}
    for split, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(config.eval_iters):
            x, y = get_batch(
                data, config.batch_size, config.block_size, generator, device
            )  # (B, T), (B, T)
            with autocast_context(device):
                total += cross_entropy_loss(model(x), y).item()
        losses[split] = total / config.eval_iters
    model.train(was_training)
    return losses


class RunLogger:
    """把每次评估的 train/val loss 追加到 out_dir/metrics.csv，并打印到终端。"""

    def __init__(self, out_dir: str | Path) -> None:
        self.path = Path(out_dir) / "metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss"])

    def log_eval(self, step: int, losses: dict[str, float]) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([step, losses["train"], losses["val"]])
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")


def save_checkpoint(
    path: Path, model: nn.Module, config: TrainConfig, vocab_size: int
) -> None:
    # 保存参数和重建模型所需的配置；M1 不做断点续训，暂不存优化器状态
    payload = {
        "model": model.state_dict(),
        "config": asdict(config),
        "vocab_size": vocab_size,
    }
    torch.save(payload, path)


def git_commit() -> str:
    # 记录代码版本；工作区有未提交改动时加上 -dirty，方便追溯
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def peak_memory_mb(device: torch.device) -> float | None:
    # 只有 CUDA 能准确读到显存峰值；其他设备记为 None
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return None


def write_run_info(out_dir: Path, config: TrainConfig, extra: dict) -> None:
    info = {"config": asdict(config), "git_commit": git_commit(), **extra}
    text = json.dumps(info, ensure_ascii=False, indent=2)
    (out_dir / "run.json").write_text(text, encoding="utf-8")
