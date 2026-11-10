"""把数据、模型、优化器、主循环和记录串起来，完成一次训练。"""

import time
from dataclasses import asdict
from pathlib import Path

import torch

from tlab.data.loader import load_splits, load_tokenizer
from tlab.device import get_device
from tlab.model.gpt import GPT, count_parameters
from tlab.train.config import TrainConfig
from tlab.train.trainer import train
from tlab.train.utils import (
    RunLogger,
    check_same_structure,
    estimate_loss,
    gpt_config_from,
    load_checkpoint,
    peak_memory_mb,
    save_checkpoint,
    write_run_info,
)


def run(config: TrainConfig) -> Path:
    """训练一次，输出目录里写 metrics.csv、run.json、ckpt.pt，返回输出目录。"""
    torch.manual_seed(config.seed)
    device = get_device() if config.device == "auto" else torch.device(config.device)

    # 数据：词表决定 vocab_size；字符级整个读进内存，BPE 用 memmap 按需读
    data_dir = Path(config.data_dir)
    tokenizer = load_tokenizer(data_dir)
    train_data, val_data = load_splits(data_dir)  # (N_train,), (N_val,)

    # 模型与优化器
    model_config = gpt_config_from(asdict(config), tokenizer.vocab_size)
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    n_params = count_parameters(model)
    print(f"设备 {device}，参数量 {n_params:,}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"

    # 训练用的随机数生成器放在这里创建，才能随存档一起保存、续训时一起恢复
    generator = torch.Generator().manual_seed(config.seed)
    start_step = 0
    if config.resume:
        # 说了要续训却没有存档时直接报错，免得以为在续训、其实从头开始
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume=True，但找不到 {ckpt_path}")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check_same_structure(saved["config"], config)
        start_step = load_checkpoint(ckpt_path, model, optimizer, generator)
        # 总步数比存档已训的还少：继续下去会写出 step 倒退的存档和负的 token 数
        if config.max_iters < start_step:
            raise ValueError(
                f"存档已训到第 {start_step} 步，max_iters={config.max_iters} 不能更小"
            )
        print(f"从 {ckpt_path} 的第 {start_step} 步继续训练")
    # 续训时保留存档之前的日志，删掉存档之后、会被重新跑的那几行
    logger = RunLogger(out_dir, resume=config.resume, start_step=start_step)

    def checkpoint(step: int) -> None:
        # step 是“下一步从哪开始”
        save_checkpoint(
            ckpt_path, model, config, tokenizer.vocab_size, optimizer, step, generator
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 主循环；计时包含其中的评估
    start = time.perf_counter()
    train_losses = train(
        model,
        optimizer,
        train_data,
        val_data,
        config,
        device,
        logger,
        start_step=start_step,
        generator=generator,
        checkpoint=checkpoint,
    )
    elapsed = time.perf_counter() - start

    final = estimate_loss(model, train_data, val_data, config, device)
    # 这次实际训练的 token 数：每步 batch_size × block_size × 累积次数，续训只算新跑的步
    steps_run = config.max_iters - start_step
    tokens = steps_run * config.batch_size * config.block_size * config.grad_accum_steps
    checkpoint(config.max_iters)
    # 续训时把上一段的 run.json 放进 previous_runs，不覆盖它的 commit、吞吐和显存记录
    write_run_info(
        out_dir,
        config,
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "n_params": n_params,
            "start_step": start_step,
            "train_tokens": tokens,
            "train_seconds": elapsed,
            "tokens_per_sec": tokens / elapsed,
            "peak_memory_mb": peak_memory_mb(device),
            "final_train_loss": final["train"],
            "final_val_loss": final["val"],
            "last_step_loss": train_losses[-1] if train_losses else None,
        },
        keep_previous=config.resume,
    )
    print(f"完成：val loss {final['val']:.4f}，{tokens / elapsed:,.0f} tokens/s")
    return out_dir
