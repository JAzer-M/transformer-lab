"""LoRA SFT 的训练循环和配置（M5）。

和 train/trainer.py 的预训练不同：数据是一条一条的问答，不是随机窗口；
只有答案算 loss；梯度累积按整个 batch 的答案 token 数平均。bf16 autocast、梯度累积、
裁剪、warmup + cosine、日志和存档的写法照 trainer.py 来。
"""

import csv
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tlab.convert.qwen3 import load_qwen3, resolve_model_dir
from tlab.device import get_device
from tlab.eval.gsm8k import Tokenizer, load_examples
from tlab.finetune.lora_model import (
    DEFAULT_TARGETS,
    SUPPORTED_TARGETS,
    apply_lora,
    count_trainable,
    load_lora,
    lora_settings,
    save_lora,
)
from tlab.finetune.sft_data import (
    Item,
    collate,
    epoch_order,
    optimizer_steps,
    tokenize_examples,
)
from tlab.finetune.sft_labels import IGNORE_INDEX
from tlab.tokenizer.hf_bpe import HFTokenizer
from tlab.train.config import load_dataclass
from tlab.train.schedule import get_lr
from tlab.train.utils import (
    autocast_context,
    file_sha256,
    git_commit,
    peak_memory_mb,
)

BASE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
METRICS = [
    "step",
    "train_loss",
    "val_loss",
    "lr",
    "grad_norm",
    "tokens",  # 这一步的真实 token 数（不含 pad）
    "tokens_per_second",
]


@dataclass
class SFTConfig:
    """一次 LoRA SFT 的全部设置；默认值就是 configs/finetune/sft_r8.yaml。"""

    repo: str = "Qwen/Qwen3-0.6B-Base"
    data_dir: str = "data/gsm8k"
    split_file: str = "configs/finetune/gsm8k_sft_split.json"
    out_dir: str = "runs/m5/sft_r8"
    # LoRA
    r: int = 8
    alpha: float = 16.0
    targets: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    # 优化器和学习率：warmup 之后 cosine 降到 min_lr
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    warmup_iters: int = 20
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0  # 梯度总范数上限；0 表示不裁剪
    # 数据：每步 batch_size × grad_accum_steps 条
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_len: int = 512  # 输入比这长的样本丢掉
    epochs: int = 1
    max_steps: int | None = None  # 短测时只跑前几步；学习率仍按完整的步数调度
    eval_interval: int = 50  # 每多少步看一次 val loss；最后一步也看
    eos_token_id: int = 151643  # 基座的 <|endoftext|>，也拿来当 pad
    base_dtype: str = "float32"  # 显存不够时改成 bfloat16，只转参数
    # 显存短测：只跑一步，用最长的 batch_size × grad_accum_steps 条，显存峰值就是上限
    memory_probe: bool = False
    # 从这个 LoRA 存档开始训练，不随机初始化 A；F4 让 PEFT 和自研从同一个起点出发
    init_lora: str | None = None
    seed: int = 1337
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("r", "batch_size", "grad_accum_steps", "max_len", "epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 至少是 1，收到 {getattr(self, name)}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps 至少是 1，收到 {self.max_steps}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval 至少是 1，收到 {self.eval_interval}")
        if self.base_dtype not in BASE_DTYPES:
            raise ValueError(
                f"base_dtype 应为 {list(BASE_DTYPES)}，收到 {self.base_dtype!r}"
            )
        if not self.targets or not set(self.targets) <= set(SUPPORTED_TARGETS):
            raise ValueError(
                f"targets 只能是 {SUPPORTED_TARGETS} 的子集，收到 {self.targets}"
            )
        for name in ("alpha", "learning_rate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 要大于 0，收到 {getattr(self, name)}")
        # 负数不报错的话，grad_clip 会悄悄关掉裁剪
        for name in ("grad_clip", "weight_decay", "warmup_iters"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能是负数，收到 {getattr(self, name)}")
        if not 0 <= self.min_lr <= self.learning_rate:
            raise ValueError(f"min_lr 应在 0 和 learning_rate 之间，收到 {self.min_lr}")


def load_sft_config(path: str | Path, overrides: list[str] | None = None) -> SFTConfig:
    """读取 SFT 配置，命令行可覆盖单个字段，如 r=16 seed=2024。"""
    return load_dataclass(SFTConfig, path, overrides)


def cast_base_params(model: nn.Module, dtype: torch.dtype) -> None:
    """只把参数转成 dtype；buffer 不动：RoPE 的 inv_freq 转成 bf16 会让角度变粗。"""
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype)


def sft_step(
    model: nn.Module,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, int]:
    """一步的前向和反向：返回这一步答案 token 的平均 loss 和 token 数。

    每个微批次先求 loss 的总和，再除以整步的答案 token 数：分成几个微批次和一次算完
    结果相同。如果每个微批次各自平均，答案短的那几条权重就变大了。
    """
    n_tokens = sum(count_answer_tokens(targets) for _, targets in micro_batches)
    total = 0.0
    for inputs, targets in micro_batches:
        loss_sum = answer_loss_sum(model, inputs, targets, device)  # ()
        (loss_sum / n_tokens).backward()
        total += loss_sum.item()
    return total / n_tokens, n_tokens


def count_answer_tokens(targets: torch.Tensor) -> int:
    """要算 loss 的 token 数：提示和 pad 的目标是 -100，不算。targets 是 (B, T)。"""
    return int((targets != IGNORE_INDEX).sum())


def answer_loss_sum(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """一个微批次里所有答案 token 的 loss 之和。inputs、targets 都是 (B, T)。"""
    inputs, targets = inputs.to(device), targets.to(device)  # (B, T)
    with autocast_context(device):
        logits = model(inputs)  # (B, T, V)
        return F.cross_entropy(
            logits.flatten(0, 1),  # (B·T, V)
            targets.flatten(),  # (B·T,)
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )  # ()


@torch.no_grad()
def answer_loss(
    model: nn.Module,
    items: Sequence[Item],
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> float:
    """在 items 上算答案 token 的平均 loss（验证集用），和训练时一样按 token 加权。"""
    total, n_tokens = 0.0, 0
    for start in range(0, len(items), batch_size):
        inputs, targets = collate(items[start : start + batch_size], pad_id)  # (B, T)
        total += answer_loss_sum(model, inputs, targets, device).item()
        n_tokens += count_answer_tokens(targets)
    return total / n_tokens


def training_plan(items: Sequence[Item], config: SFTConfig) -> list[list[list[int]]]:
    """每一步用哪些样本：plan[步][微批次] 是样本下标的列表。PEFT 对照用同一个函数。

    正常训练时每个 epoch 按种子重新打乱，凑不满一步的丢掉；
    显存短测时只有一步，用最长的那几条。
    """
    if config.memory_probe:
        longest = sorted(
            range(len(items)), key=lambda i: len(items[i][0]), reverse=True
        )
        return optimizer_steps(longest, config.batch_size, config.grad_accum_steps)[:1]
    return [
        step
        for epoch in range(config.epochs)
        for step in optimizer_steps(
            epoch_order(len(items), config.seed, epoch),
            config.batch_size,
            config.grad_accum_steps,
        )
    ]


def train_lora(
    model: nn.Module,
    train_items: Sequence[Item],
    val_items: Sequence[Item],
    config: SFTConfig,
    device: torch.device,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """训练已经装好自研 LoRA 的模型；写出 metrics.csv、run.json、lora_init.pt 和 lora.pt。"""
    # 没装 LoRA、或各层的 r 不同时报错
    settings = lora_settings(model)
    return fit(
        model,
        train_items,
        val_items,
        config,
        device,
        save_weights=lambda path: save_lora(model, path),
        extra_info={"lora": settings, **(extra_info or {})},
    )


def fit(
    model: nn.Module,
    train_items: Sequence[Item],
    val_items: Sequence[Item],
    config: SFTConfig,
    device: torch.device,
    save_weights: Callable[[Path], None],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """通用的训练循环：只训练 requires_grad 的参数。自研 LoRA 和 PEFT 对照（F4）共用它，
    两边的差别才只在 LoRA 本身。model(inputs) 要返回 (B, T, V) 的分数；
    save_weights(path) 负责把要训练的参数存到 path，训练前后各调用一次。
    """
    out_dir = Path(config.out_dir)
    if (out_dir / "run.json").exists() or (out_dir / "lora.pt").exists():
        # 换种子再跑时忘了改 out_dir，会把上一次的结果覆盖掉
        raise ValueError(f"{out_dir} 里已经有训练结果，换一个 out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    plan = training_plan(train_items, config)
    if not plan:
        raise ValueError(f"训练样本只有 {len(train_items)} 条，凑不满一步")
    total_steps = len(plan)
    steps_run = min(total_steps, config.max_steps or total_steps)
    # get_lr 只用这几个字段；max_iters 是完整的步数，短测时学习率也和正式训练一样
    schedule = SimpleNamespace(
        learning_rate=config.learning_rate,
        min_lr=config.min_lr,
        warmup_iters=config.warmup_iters,
        lr_schedule="cosine",
        max_iters=total_steps,
    )

    # 训练前的 A、B：PEFT 对照（F4）把它们拷过去，两边才从同一个起点出发
    save_weights(out_dir / "lora_init.pt")
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(METRICS)
    model.train()
    start = time.perf_counter()
    seen_tokens = 0
    train_seconds = 0.0  # 只算训练，不含看 val loss 和存档
    train_loss = val_loss = float("nan")
    for step in range(steps_run):
        step_start = time.perf_counter()
        lr = get_lr(step, schedule)  # type: ignore[arg-type]
        for group in optimizer.param_groups:
            group["lr"] = lr
        micro_batches = [
            collate([train_items[i] for i in indices], config.eos_token_id)
            for indices in plan[step]
        ]
        optimizer.zero_grad(set_to_none=True)
        train_loss, _ = sft_step(model, micro_batches, device)
        # 裁剪前的总范数；不裁剪时上限设成无穷大，只拿范数
        limit = config.grad_clip if config.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, limit).item()
        optimizer.step()

        # 真实 token 数（不含 pad），用来算速度
        step_tokens = sum(len(train_items[i][0]) for micro in plan[step] for i in micro)
        seen_tokens += step_tokens
        step_seconds = time.perf_counter() - step_start
        train_seconds += step_seconds
        speed = step_tokens / step_seconds
        last = step == steps_run - 1
        logged_val = ""
        if (step + 1) % config.eval_interval == 0 or last:
            model.eval()
            val_loss = answer_loss(
                model, val_items, config.batch_size, config.eos_token_id, device
            )
            model.train()
            logged_val = val_loss
            print(
                f"step {step + 1}/{total_steps} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | lr {lr:.2e} | {speed:.0f} tok/s",
                flush=True,
            )
        with open(out_dir / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow(
                [step + 1, train_loss, logged_val, lr, grad_norm, step_tokens, speed]
            )

    save_weights(out_dir / "lora.pt")
    info = {
        "config": asdict(config),
        "git_commit": git_commit(),
        "seed": config.seed,
        "n_trainable": count_trainable(model),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "total_steps": total_steps,
        "steps_run": steps_run,
        "device": str(device),
        "seconds": time.perf_counter() - start,
        "train_seconds": train_seconds,
        "tokens_per_second": seen_tokens / train_seconds,
        "peak_memory_mb": peak_memory_mb(device),
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        **(extra_info or {}),
    }
    (out_dir / "run.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info


def prepare_items(
    config: SFTConfig, tokenizer: Tokenizer
) -> tuple[list[Item], list[Item], dict[str, Any]]:
    """按划分文件读训练集，做成训练和验证样本。PEFT 对照用同一个函数。"""
    split = json.loads(Path(config.split_file).read_text(encoding="utf-8"))
    examples = load_examples(Path(config.data_dir) / "train.jsonl")
    if len(examples) != split["train_total"]:
        raise ValueError(
            f"划分文件是按 {split['train_total']} 条做的，数据有 {len(examples)} 条"
        )
    train_items, dropped_train = tokenize_examples(
        [examples[i] for i in split["train_indices"]],
        tokenizer,
        config.eos_token_id,
        config.max_len,
    )
    val_items, dropped_val = tokenize_examples(
        [examples[i] for i in split["val_indices"]],
        tokenizer,
        config.eos_token_id,
        config.max_len,
    )
    info = {
        "split_seed": split["seed"],
        "dropped_train": dropped_train,
        "dropped_val": dropped_val,
    }
    return train_items, val_items, info


def resolve_device(config: SFTConfig) -> torch.device:
    """按配置选设备；基座转 bf16 时必须在 GPU 上。"""
    device = get_device() if config.device == "auto" else torch.device(config.device)
    if config.base_dtype != "float32" and device.type != "cuda":
        # 只有 GPU 上有 bf16 autocast；CPU 上 bf16 的基座和 fp32 的 A、B 相乘会报错
        raise ValueError(f"base_dtype={config.base_dtype} 需要 GPU，当前是 {device}")
    return device


def check_start(settings: dict[str, Any], config: SFTConfig) -> dict[str, str]:
    """init_lora 存档的 r、alpha、targets 必须和配置相同；返回要记在结果里的来源。"""
    expected = {"r": config.r, "alpha": config.alpha, "targets": config.targets}
    if settings != expected:
        raise ValueError(f"init_lora 的设置是 {settings}，配置里是 {expected}")
    assert config.init_lora is not None
    return {"path": config.init_lora, "sha256": file_sha256(Path(config.init_lora))}


def run_sft(config: SFTConfig, model_dir: Path | None = None) -> dict[str, Any]:
    """读数据和划分、载入基座、装 LoRA、训练。model_dir 不给时按 config.repo 下载。"""
    device = resolve_device(config)
    model_dir, model_name = resolve_model_dir(config.repo, model_dir)
    tokenizer = HFTokenizer.load(model_dir / "tokenizer.json")
    train_items, val_items, data_info = prepare_items(config, tokenizer)

    model = load_qwen3(model_dir, device)
    if config.base_dtype != "float32":
        cast_base_params(model, BASE_DTYPES[config.base_dtype])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # 种子在装 LoRA 之前设：A 的随机初始值由它决定
    torch.manual_seed(config.seed)
    init = None
    if config.init_lora is None:
        apply_lora(model, config.r, config.alpha, config.targets)
    else:
        init = check_start(load_lora(model, Path(config.init_lora)), config)
    return train_lora(
        model,
        train_items,
        val_items,
        config,
        device,
        extra_info={"model": model_name, "init_lora": init, **data_info},
    )
