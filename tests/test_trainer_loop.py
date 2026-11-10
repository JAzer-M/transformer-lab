import csv
import json
import math
from pathlib import Path

import pytest
import torch

# M2 主循环的行为测试：学习率调度、梯度累积、梯度裁剪由用户在 trainer.py 中实现，
# 这里只规定断点续训的行为，不规定写法。

VOCAB = 11


def tiny_config(tmp_path: Path, **overrides):
    from tlab.train.config import TrainConfig

    fields = {
        "data_dir": str(tmp_path / "data"),
        "out_dir": str(tmp_path / "out"),
        "n_layers": 1,
        "n_heads": 2,
        "d_model": 16,
        "block_size": 8,
        "batch_size": 4,
        "learning_rate": 1e-2,
        "max_iters": 4,
        "eval_interval": 1000,
        "eval_iters": 1,
        "seed": 0,
        "device": "cpu",
    }
    fields.update(overrides)
    return TrainConfig(**fields)


def tiny_model(seed: int = 0):
    from tlab.model.gpt import GPT, GPTConfig

    torch.manual_seed(seed)
    return GPT(GPTConfig(VOCAB, 8, 16, 1, 2))


def toy_data() -> torch.Tensor:
    return torch.randint(0, VOCAB, (300,), generator=torch.Generator().manual_seed(1))


def run_train(model, optimizer, config, tmp_path: Path, **kwargs) -> list[float]:
    from tlab.train.trainer import train
    from tlab.train.utils import RunLogger

    data = toy_data()  # (N,)
    logger = RunLogger(tmp_path / "logs")
    return train(
        model, optimizer, data, data, config, torch.device("cpu"), logger, **kwargs
    )


class RecordingSGD(torch.optim.SGD):
    """普通 SGD，额外记下每次 step() 时用的学习率。"""

    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.lrs: list[float] = []

    def step(self, closure=None):
        self.lrs.append(self.param_groups[0]["lr"])
        return super().step(closure)


# ---------- 默认配置：与 M1 完全一致 ----------


def test_default_config_matches_m1_loop_step_by_step(tmp_path: Path) -> None:
    from tlab.data.batch import get_batch
    from tlab.train.utils import cross_entropy_loss

    config = tiny_config(tmp_path, max_iters=6)
    data = toy_data()  # (N,)

    # 参照：M1 的主循环（不含评估；评估不改参数，也不动训练用的随机数）
    ref_model = tiny_model()
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    ref_losses = []
    for _ in range(config.max_iters):
        x, y = get_batch(
            data, config.batch_size, config.block_size, generator
        )  # (B, T), (B, T)
        loss = cross_entropy_loss(ref_model(x), y)  # (B, T, V) 与 (B, T) → ()
        ref_opt.zero_grad(set_to_none=True)
        loss.backward()
        ref_opt.step()
        ref_losses.append(loss.item())

    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses = run_train(model, optimizer, config, tmp_path)

    # 不开新功能时逐步完全相同：M1 的实验仍可复现
    assert losses == ref_losses
    for a, b in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(a, b)


# ---------- 梯度累积 ----------


def test_grad_accumulation_equals_averaging_the_micro_batches(tmp_path: Path) -> None:
    from tlab.data.batch import get_batch
    from tlab.train.utils import cross_entropy_loss

    lr = 0.1
    config = tiny_config(tmp_path, max_iters=2, grad_accum_steps=2, learning_rate=lr)
    data = toy_data()  # (N,)

    # 参照：每步依次取 2 个 micro batch，把两份 loss 平均后只反向、更新一次
    ref_model = tiny_model()
    generator = torch.Generator().manual_seed(config.seed)
    ref_losses = []
    for _ in range(config.max_iters):
        # 两个 micro batch，各自 x, y 为 (B, T)，loss 为 ()
        micro = [
            cross_entropy_loss(ref_model(x), y)
            for x, y in (
                get_batch(data, config.batch_size, config.block_size, generator)
                for _ in range(2)
            )
        ]
        loss = (micro[0] + micro[1]) / 2  # ()
        ref_model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in ref_model.parameters():
                p -= lr * p.grad  # 普通 SGD
        ref_losses.append(loss.item())

    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = run_train(model, optimizer, config, tmp_path)

    # 记录的 loss 是 micro batch 的平均值；参数与“大 batch 平均 loss”的更新一致
    assert losses == pytest.approx(ref_losses, rel=1e-5)
    for a, b in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_optimizer_steps_once_per_iteration(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, max_iters=4, grad_accum_steps=3)
    model = tiny_model()
    optimizer = RecordingSGD(model.parameters(), lr=config.learning_rate)

    losses = run_train(model, optimizer, config, tmp_path)

    # 累积 3 次只更新 1 次：4 步就是 4 次 step()，每步记一个 loss
    assert len(optimizer.lrs) == 4
    assert len(losses) == 4


# ---------- 梯度裁剪 ----------


def update_norm(config, tmp_path: Path) -> float:
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    run_train(model, optimizer, config, tmp_path)

    # 所有参数的变化量拼成一个向量后的长度
    deltas = [
        (p.detach() - b).flatten() for p, b in zip(model.parameters(), before)
    ]  # 每个参数一段 (n_i,)
    return torch.cat(deltas).norm().item()  # (Σ n_i,) → 标量


def test_grad_clip_caps_the_update_size(tmp_path: Path) -> None:
    clip = 1e-3
    clipped = tiny_config(tmp_path, max_iters=1, learning_rate=1.0, grad_clip=clip)
    unclipped = tiny_config(tmp_path, max_iters=1, learning_rate=1.0, grad_clip=0.0)

    # SGD 一步的更新量 = lr × 梯度；裁剪后梯度总范数等于 clip
    assert update_norm(clipped, tmp_path) == pytest.approx(clip, rel=1e-3)
    # grad_clip = 0 表示不裁剪，更新量远大于 clip
    assert update_norm(unclipped, tmp_path) > 10 * clip


# ---------- 学习率调度 ----------


def test_learning_rate_follows_schedule_and_is_logged(tmp_path: Path) -> None:
    from tlab.train.schedule import get_lr

    config = tiny_config(
        tmp_path,
        max_iters=6,
        lr_schedule="cosine",
        warmup_iters=2,
        min_lr=1e-4,
        eval_interval=3,
    )
    model = tiny_model()
    optimizer = RecordingSGD(model.parameters(), lr=config.learning_rate)

    run_train(model, optimizer, config, tmp_path)

    # 每一步 step() 时用的学习率都等于 get_lr(step)
    expected = [get_lr(s, config) for s in range(config.max_iters)]
    assert optimizer.lrs == pytest.approx(expected)
    # 评估时把当时的学习率记进 metrics.csv（第 0、3 步与最后一步）
    with open(tmp_path / "logs" / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [0, 3, 5]
    assert [float(r["lr"]) for r in rows] == pytest.approx(
        [expected[0], expected[3], expected[5]]
    )


# ---------- 断点续训 ----------


class StopTraining(Exception):
    pass


def test_checkpoint_callback_runs_every_interval(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, max_iters=7, ckpt_interval=3)
    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    saved_at: list[int] = []

    run_train(model, optimizer, config, tmp_path, checkpoint=saved_at.append)

    # 存的是“下一步从哪开始”：跑完第 0–2 步存 3，跑完第 3–5 步存 6；结束时由 run.py 存
    assert saved_at == [3, 6]


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    from tlab.train.utils import load_checkpoint, save_checkpoint

    # 把新功能全打开，确保存档里每一样状态都必不可少
    config = tiny_config(
        tmp_path,
        max_iters=6,
        ckpt_interval=3,
        grad_accum_steps=2,
        grad_clip=1.0,
        lr_schedule="cosine",
        warmup_iters=2,
        min_lr=1e-4,
    )

    # 一口气跑完 6 步
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    full_losses = run_train(model, optimizer, config, tmp_path)

    # 跑到第 3 步存档后“断电”
    path = tmp_path / "ckpt.pt"
    model_a = tiny_model()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=config.learning_rate)
    gen_a = torch.Generator().manual_seed(config.seed)

    def save_then_stop(step: int) -> None:
        save_checkpoint(path, model_a, config, VOCAB, opt_a, step, gen_a)
        raise StopTraining

    with pytest.raises(StopTraining):
        run_train(
            model_a, opt_a, config, tmp_path, generator=gen_a, checkpoint=save_then_stop
        )

    # 换一个初始化完全不同的模型，从存档接着跑剩下的 3 步
    model_b = tiny_model(seed=99)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=config.learning_rate)
    gen_b = torch.Generator().manual_seed(12345)
    start = load_checkpoint(path, model_b, opt_b, gen_b)
    resumed_losses = run_train(
        model_b, opt_b, config, tmp_path, start_step=start, generator=gen_b
    )

    assert start == 3
    assert resumed_losses == full_losses[3:]
    for a, b in zip(model.parameters(), model_b.parameters()):
        assert torch.equal(a, b)


# ---------- run()：读档、续训与 tokens 统计 ----------


def write_char_dir(data_dir: Path) -> None:
    from tlab.data.char import CharTokenizer, save_ids, split_ids

    text = "to be or not to be, that is the question.\n" * 20
    tokenizer = CharTokenizer.from_text(text)
    train, val = split_ids(tokenizer.encode(text))
    data_dir.mkdir(parents=True)
    save_ids(train, data_dir / "train.bin")
    save_ids(val, data_dir / "val.bin")
    tokenizer.save(data_dir / "meta.json")


def test_run_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_char_dir(tmp_path / "data")
    first = tiny_config(tmp_path, max_iters=4, eval_interval=2, ckpt_interval=2)
    run(first)
    # 同一个输出目录，把总步数加到 6，从存档继续
    out_dir = run(tiny_config(tmp_path, max_iters=6, eval_interval=2, resume=True))

    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    assert ckpt["step"] == 6
    # 日志接着写：第一次的 0、2、3 加上续训的 4、5
    with open(out_dir / "metrics.csv", newline="") as f:
        steps = [int(r["step"]) for r in csv.DictReader(f)]
    assert steps == [0, 2, 3, 4, 5]
    # tokens 只算续训这 2 步
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["start_step"] == 4
    assert meta["train_tokens"] == 2 * 4 * 8


def test_run_counts_tokens_with_grad_accum(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_char_dir(tmp_path / "data")
    out_dir = run(tiny_config(tmp_path, max_iters=3, grad_accum_steps=2))

    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    # 每步处理 batch_size × block_size × grad_accum_steps 个 token
    assert meta["train_tokens"] == 3 * 4 * 8 * 2
    assert math.isfinite(meta["tokens_per_sec"])


def test_run_resume_without_checkpoint_is_an_error(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_char_dir(tmp_path / "data")

    # 说了要续训却找不到存档：直接报错，避免以为在续训、其实从头开始；
    # 报错信息要点明是 resume 找不到存档，而不只是一个文件名
    with pytest.raises(FileNotFoundError, match="resume=True"):
        run(tiny_config(tmp_path, resume=True))


def test_run_resume_keeps_the_previous_run_info(tmp_path: Path) -> None:
    from tlab.train.run import run

    write_char_dir(tmp_path / "data")
    run(tiny_config(tmp_path, max_iters=4, eval_interval=2))
    first = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    out_dir = run(tiny_config(tmp_path, max_iters=6, eval_interval=2, resume=True))

    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    # 外层是这次续训的信息；第一段的 commit、tokens/s、显存峰值等原样留在 previous_runs
    assert meta["start_step"] == 4
    assert meta["previous_runs"] == [first]


def test_run_resume_with_fewer_steps_than_checkpoint_is_an_error(
    tmp_path: Path,
) -> None:
    from tlab.train.run import run

    write_char_dir(tmp_path / "data")
    run(tiny_config(tmp_path, max_iters=4))

    # 存档已训到第 4 步，再要求总共只训 2 步：直接报错，不写出 step 倒退的存档
    with pytest.raises(ValueError, match="max_iters"):
        run(tiny_config(tmp_path, max_iters=2, resume=True))


def test_run_resume_from_an_earlier_checkpoint_has_no_duplicate_steps(
    tmp_path: Path,
) -> None:
    from tlab.train.run import run

    write_char_dir(tmp_path / "data")
    run(tiny_config(tmp_path, max_iters=6, eval_interval=2))
    # 模拟：第 3 步存过档，之后跑到第 4、5 步评估后崩溃——把存档的 step 改回 3
    path = tmp_path / "out" / "ckpt.pt"
    ckpt = torch.load(path, weights_only=False)
    ckpt["step"] = 3
    torch.save(ckpt, path)

    out_dir = run(tiny_config(tmp_path, max_iters=6, eval_interval=2, resume=True))

    with open(out_dir / "metrics.csv", newline="") as f:
        steps = [int(r["step"]) for r in csv.DictReader(f)]
    # 第 4、5 步被重新跑了一遍：每一步只保留一行
    assert steps == [0, 2, 4, 5]
