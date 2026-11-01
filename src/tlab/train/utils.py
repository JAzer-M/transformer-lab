"""训练主循环用到的外围工具：混合精度、评估、日志、checkpoint、运行信息。"""

import contextlib
import csv
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tlab.data.batch import get_batch
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
    train_data: torch.Tensor | np.ndarray,
    val_data: torch.Tensor | np.ndarray,
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
    """把每次评估的 train/val loss 与学习率追加到 out_dir/metrics.csv，并打印到终端。"""

    def __init__(
        self, out_dir: str | Path, resume: bool = False, start_step: int | None = None
    ) -> None:
        self.path = Path(out_dir) / "metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = ["step", "train_loss", "val_loss", "lr"]
        # 续训时保留旧记录；给了 start_step 时，该步及以后的会重新跑，先删掉以免重复
        kept: list[list[str]] = []
        if resume and self.path.exists():
            with open(self.path, newline="") as f:
                rows = list(csv.reader(f))[1:]  # 去掉表头
            kept = [r for r in rows if start_step is None or int(r[0]) < start_step]
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerows([header, *kept])

    def log_eval(
        self, step: int, losses: dict[str, float], lr: float | None = None
    ) -> None:
        row = [step, losses["train"], losses["val"], "" if lr is None else lr]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        line = f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
        print(line if lr is None else f"{line} | lr {lr:.2e}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    vocab_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    generator: torch.Generator | None = None,
) -> None:
    """保存参数与重建模型所需的配置；给了 optimizer 等时，还能据此断点续训。"""
    payload = {
        "model": model.state_dict(),
        "config": asdict(config),
        "vocab_size": vocab_size,
        # 以下用于续训：step 是“下一步从哪开始”
        "step": step,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "generator": None if generator is None else generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    # 先写临时文件再改名：存到一半被打断，也不会留下损坏的 ckpt.pt
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> int:
    """把模型、优化器、训练用生成器与全局随机数恢复到存档时的状态，返回下一步的 step。"""
    # 先读到 CPU：随机数状态必须是 CPU 张量；参数与优化器状态在 load 时会搬到各自设备
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("optimizer") is None or ckpt.get("step") is None:
        raise ValueError(f"{path} 只存了模型参数，不能用来续训")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    generator.set_state(ckpt["generator"])
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return ckpt["step"]


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


def write_run_info(
    out_dir: Path, config: TrainConfig, extra: dict, keep_previous: bool = False
) -> None:
    """写 run.json；keep_previous=True（续训）时，把旧的记录放进 previous_runs 保留。"""
    info = {"config": asdict(config), "git_commit": git_commit(), **extra}
    path = out_dir / "run.json"
    if keep_previous and path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        # 更早的几段已经在旧记录的 previous_runs 里，按时间顺序接在前面
        info["previous_runs"] = [*old.pop("previous_runs", []), old]
    text = json.dumps(info, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
