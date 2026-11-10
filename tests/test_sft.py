"""SFT 的训练循环和配置：只有答案算 loss、按 token 加权的累积、学习率、输出。"""

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from tlab.finetune.lora import LoRALinear
from tlab.finetune.lora_model import apply_lora, load_lora
from tlab.finetune.sft import (
    SFTConfig,
    answer_loss,
    cast_base_params,
    load_sft_config,
    sft_step,
    train_lora,
    training_plan,
)
from tlab.finetune.sft_data import collate
from tlab.finetune.sft_labels import IGNORE_INDEX
from tlab.model.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
VOCAB, EOS = 20, 19


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(
            vocab_size=VOCAB,
            block_size=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            qk_norm=True,
            norm="rmsnorm",
            mlp="swiglu",
            pos_encoding="rope",
            rmsnorm_fp32=True,
        )
    )


def tiny_lora_model(seed: int = 0) -> GPT:
    model = tiny_model(seed)
    torch.manual_seed(seed + 100)
    apply_lora(model, r=4, alpha=8)
    return model


def copy_task(n: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """能学会的小任务：提示 [a, b]，答案是两个 b，再加 eos。"""
    from tlab.finetune.sft_labels import build_example

    g = torch.Generator().manual_seed(seed)
    items = []
    for _ in range(n):
        a, b = torch.randint(0, EOS, (2,), generator=g).tolist()
        items.append(build_example([a, b], [b, b], EOS))
    return items


def tiny_config(tmp_path: Path, **fields: object) -> SFTConfig:
    values: dict = {
        "out_dir": str(tmp_path / "run"),
        "batch_size": 4,
        "grad_accum_steps": 2,
        "learning_rate": 1e-2,
        "min_lr": 1e-3,
        "warmup_iters": 2,
        "eval_interval": 3,
        "device": "cpu",
        "eos_token_id": EOS,
    }
    values.update(fields)
    return SFTConfig(**values)


def read_metrics(out_dir: Path) -> list[dict[str, str]]:
    with open(out_dir / "metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


# ---- 损失 ----


class FixedScores(nn.Module):
    """每个位置的分数都一样：1 号是 ln 3，其余是 0。所以 1 号的概率是 3/6，其余各 1/6。"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))  # 让 backward 有东西可算

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        scores = torch.zeros(B, T, 4)  # (B, T, V)
        scores[..., 1] = math.log(3)
        return scores * self.scale


def test_only_answer_positions_count_in_the_loss() -> None:
    inputs = torch.tensor([[0, 0, 0]])  # (1, 3)
    targets = torch.tensor([[IGNORE_INDEX, 1, 2]])  # 第 0 个位置是提示

    loss, n_tokens = sft_step(FixedScores(), [(inputs, targets)], torch.device("cpu"))

    # 只平均答案的两个位置：-ln(3/6) 和 -ln(1/6)；提示位置如果也算，会再多一个 -ln(1/6)
    assert n_tokens == 2
    assert loss == pytest.approx((-math.log(0.5) - math.log(1 / 6)) / 2)


def with_random_b(model: GPT) -> GPT:
    torch.manual_seed(5)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.1)
    return model


def lora_grads(model: GPT) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.parameters() if p.requires_grad]


def test_accumulated_loss_is_weighted_by_answer_tokens() -> None:
    # 两个微批次的答案长度不同：1 个和 3 个 token
    short = ([3, 4, 5], [IGNORE_INDEX, IGNORE_INDEX, 6])
    long = ([7, 8, 9, 10], [IGNORE_INDEX, 9, 10, 11])
    device = torch.device("cpu")

    split = with_random_b(tiny_lora_model())
    loss_split, n_split = sft_step(
        split, [collate([short], 0), collate([long], 0)], device
    )
    whole = with_random_b(tiny_lora_model())
    loss_whole, n_whole = sft_step(whole, [collate([short, long], 0)], device)

    # 按 token 加权时，分成两个微批次和一次算完完全一样；各自平均就会让短的那条权重变大
    assert n_split == n_whole == 4
    assert loss_split == pytest.approx(loss_whole, rel=1e-5)
    for a, b in zip(lora_grads(split), lora_grads(whole), strict=True):
        torch.testing.assert_close(a, b)


# ---- 训练循环 ----


def test_training_lowers_the_loss_and_changes_only_lora(tmp_path: Path) -> None:
    model = tiny_model()
    # 随机小模型的词表权重很小（std 0.02），lm_head 和它共用又被冻结，分数最大不到 1，
    # loss 怎么学也降不了多少；放大 10 倍，才看得出 LoRA 确实在学（实测 2.01 → 0.90）
    with torch.no_grad():
        model.tok_emb.weight.mul_(10)
    torch.manual_seed(100)
    apply_lora(model, r=4, alpha=8)
    base_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    lora_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    config = tiny_config(tmp_path, epochs=4)

    train_lora(model, copy_task(64), copy_task(8, seed=1), config, torch.device("cpu"))

    rows = read_metrics(Path(config.out_dir))
    assert len(rows) == 4 * 64 // 8  # 每个 epoch 8 步
    first = sum(float(r["train_loss"]) for r in rows[:4]) / 4
    last = sum(float(r["train_loss"]) for r in rows[-4:]) / 4
    assert last < first * 0.7
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, base_before[name]), name  # 基座一点没变
    after = [p for p in model.parameters() if p.requires_grad]
    assert all(not torch.equal(a, b) for a, b in zip(after, lora_before, strict=True))


def test_learning_rate_follows_warmup_then_cosine(tmp_path: Path) -> None:
    # 48 条，每步 4 × 2 = 8 条，共 6 步；warmup 2 步，学习率 1e-3 降到 1e-4
    config = tiny_config(tmp_path, learning_rate=1e-3, min_lr=1e-4)

    train_lora(
        tiny_lora_model(), copy_task(48), copy_task(8), config, torch.device("cpu")
    )

    lrs = [float(r["lr"]) for r in read_metrics(Path(config.out_dir))]
    # warmup：1e-3 × 1/2、1e-3 × 2/2；之后 cosine 的进度是 (step − 2) / 3
    # 第 3 步 0.75、第 4 步 0.25 的系数，第 5 步降到 min_lr
    expected = [5e-4, 1e-3, 1e-3, 7.75e-4, 3.25e-4, 1e-4]
    assert lrs == pytest.approx(expected)


def test_outputs_metrics_run_info_and_lora_weights(tmp_path: Path) -> None:
    model = tiny_lora_model()
    config = tiny_config(tmp_path)

    result = train_lora(model, copy_task(48), copy_task(8), config, torch.device("cpu"))

    out_dir = Path(config.out_dir)
    rows = read_metrics(out_dir)
    assert list(rows[0]) == [
        "step",
        "train_loss",
        "val_loss",
        "lr",
        "grad_norm",
        "tokens",
        "tokens_per_second",
    ]
    # 每 3 步和最后一步看一次 val loss，其余留空
    assert [r["val_loss"] != "" for r in rows] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert all(float(r["grad_norm"]) > 0 for r in rows)
    info = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert info["config"]["learning_rate"] == config.learning_rate
    assert info["git_commit"]
    assert info["seed"] == config.seed
    assert info["n_trainable"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert info["total_steps"] == info["steps_run"] == 6
    assert info["n_train"] == 48 and info["n_val"] == 8
    assert "peak_memory_mb" in info
    # 速度只算训练的时间，不含看 val loss 和存档
    tokens = sum(int(r["tokens"]) for r in rows)
    assert info["tokens_per_second"] == pytest.approx(tokens / info["train_seconds"])
    assert info["train_seconds"] <= info["seconds"]
    # 训练前的 A、B 也存下来，PEFT 对照（F4）要拷过去
    init = torch.load(out_dir / "lora_init.pt", weights_only=True)
    assert all(
        torch.equal(w, torch.zeros_like(w))
        for name, w in init["weights"].items()
        if name.endswith("lora_B")
    )
    assert info["final_val_loss"] == pytest.approx(float(rows[-1]["val_loss"]))
    assert result["final_val_loss"] == info["final_val_loss"]

    # 存下的 LoRA 载进新模型，输出和训练完的模型相同
    fresh = tiny_model()
    load_lora(fresh, out_dir / "lora.pt")
    idx = torch.randint(0, VOCAB, (2, 6), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        torch.testing.assert_close(fresh(idx), model(idx))


def test_max_steps_stops_early_but_keeps_the_full_schedule(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, learning_rate=1e-3, min_lr=1e-4, max_steps=4)

    train_lora(
        tiny_lora_model(), copy_task(48), copy_task(8), config, torch.device("cpu")
    )

    rows = read_metrics(Path(config.out_dir))
    info = json.loads((Path(config.out_dir) / "run.json").read_text(encoding="utf-8"))
    # 只跑 4 步做短测，学习率还是按 6 步的调度走，和正式训练前 4 步完全相同；
    # 如果按 4 步调度，第 4 步就已经降到 1e-4 了
    assert info["steps_run"] == 4 and info["total_steps"] == 6
    assert [float(r["lr"]) for r in rows] == pytest.approx([5e-4, 1e-3, 1e-3, 7.75e-4])
    # 第 4 步不是 eval_interval=3 的倍数，但它是最后一步，也要看 val loss
    assert rows[-1]["val_loss"] != ""


def test_training_matches_a_hand_written_reference_loop(tmp_path: Path) -> None:
    """照最常见的写法手写 4 步：清梯度、前向反向、裁剪、按调度设学习率、更新。"""
    train, val = copy_task(48), copy_task(8)
    # 裁剪上限设得很小，每一步都会被裁：去掉裁剪的话 Adam 的动量就不一样了
    config = tiny_config(
        tmp_path, learning_rate=1e-3, min_lr=1e-4, max_steps=4, grad_clip=1e-3
    )
    model = tiny_lora_model()
    train_lora(model, train, val, config, torch.device("cpu"))

    from tlab.finetune.sft_data import epoch_order, optimizer_steps

    reference = tiny_lora_model()
    params = [p for p in reference.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    steps = optimizer_steps(epoch_order(48, config.seed, 0), 4, 2)
    for step, indices in enumerate(steps[:4]):
        # 6 步的调度：warmup 2 步，之后 cosine 从 1e-3 降到 1e-4
        if step < 2:
            lr = 1e-3 * (step + 1) / 2
        else:
            lr = 1e-4 + 0.5 * (1 + math.cos(math.pi * (step - 2) / 3)) * 9e-4
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        sft_step(
            reference,
            [collate([train[i] for i in micro], EOS) for micro in indices],
            torch.device("cpu"),
        )
        torch.nn.utils.clip_grad_norm_(params, 1e-3)
        optimizer.step()

    # 运算和顺序都相同，应该逐位相同；用容差的话，漏了清梯度这种错也可能混过去
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a, b)


def test_same_seed_gives_bit_for_bit_the_same_run(tmp_path: Path) -> None:
    results = []
    for name in ("a", "b"):
        config = tiny_config(tmp_path / name)
        model = tiny_lora_model()
        train_lora(model, copy_task(48), copy_task(8), config, torch.device("cpu"))
        results.append((read_metrics(Path(config.out_dir)), model))

    (rows_a, model_a), (rows_b, model_b) = results
    strip = [{k: v for k, v in r.items() if k != "tokens_per_second"} for r in rows_a]
    assert strip == [
        {k: v for k, v in r.items() if k != "tokens_per_second"} for r in rows_b
    ]
    for a, b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(a, b)


def test_a_different_seed_trains_in_a_different_order(tmp_path: Path) -> None:
    losses = []
    for seed in (1337, 2024):
        config = tiny_config(tmp_path / str(seed), seed=seed)
        train_lora(
            tiny_lora_model(), copy_task(48), copy_task(8), config, torch.device("cpu")
        )
        losses.append([r["train_loss"] for r in read_metrics(Path(config.out_dir))])

    assert losses[0] != losses[1]


def test_training_requires_a_model_with_lora(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LoRA"):
        train_lora(
            tiny_model(),
            copy_task(48),
            copy_task(8),
            tiny_config(tmp_path),
            torch.device("cpu"),
        )


# ---- 基座转 bf16（显存不够时的退路）----


def test_cast_base_params_keeps_buffers_like_rope_in_fp32() -> None:
    model = tiny_model()

    cast_base_params(model, torch.bfloat16)

    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    # RoPE 的 inv_freq 是 buffer：转成 bf16 会让旋转角度变粗，计划要求保持 fp32
    buffers = dict(model.named_buffers())
    assert buffers and all(b.dtype == torch.float32 for b in buffers.values())
    # 之后再装 LoRA：A、B 还是 fp32
    apply_lora(model, r=4, alpha=8)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)


# ---- 配置 ----


def test_sft_config_file_has_the_starting_hyperparameters() -> None:
    config = load_sft_config(ROOT / "configs" / "finetune" / "sft_r8.yaml")

    assert (config.r, config.alpha) == (8, 16.0)
    assert config.targets == ["q_proj", "k_proj", "v_proj", "out_proj"]
    assert config.learning_rate == 2e-4 and config.min_lr == 2e-5
    assert config.warmup_iters == 20
    assert (config.beta1, config.beta2) == (0.9, 0.999)
    assert config.weight_decay == 0.0
    # 2026-11-08 用户确认：4 × 4 时最长一批要 11 GB，改成 2 × 8，等效 batch 还是 16
    assert (config.batch_size, config.grad_accum_steps) == (2, 8)
    assert config.grad_clip == 1.0
    assert config.max_len == 512 and config.epochs == 1
    assert config.eval_interval == 50
    assert config.base_dtype == "float32"
    assert config.repo == "Qwen/Qwen3-0.6B-Base"
    assert config.eos_token_id == 151643
    assert config.max_steps is None


def test_sft_config_can_be_overridden_from_the_command_line() -> None:
    config = load_sft_config(
        ROOT / "configs" / "finetune" / "sft_r8.yaml", ["r=16", "seed=2024"]
    )

    assert config.r == 16 and config.seed == 2024


@pytest.mark.parametrize(
    "fields",
    [
        {"batch_size": 0},
        {"grad_accum_steps": 0},
        {"r": 0},
        {"epochs": 0},
        {"max_steps": 0},
        {"base_dtype": "float16"},
        {"targets": ["qkv"]},
        {"targets": []},
        {"min_lr": 1.0},  # 比 learning_rate 还大
        {"alpha": 0.0},
        {"learning_rate": 0.0, "min_lr": 0.0},
        {"grad_clip": -1.0},  # 负数会悄悄关掉裁剪
        {"weight_decay": -0.1},
        {"warmup_iters": -1},
    ],
)
def test_sft_config_rejects_bad_values(fields: dict) -> None:
    with pytest.raises(ValueError):
        SFTConfig(**fields)


def test_sft_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("learning_rat: 1e-3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="未知配置项"):
        load_sft_config(path)


def test_config_defaults_match_the_file() -> None:
    # 文件里只写和默认值不同的地方也可以；这里确认两者一致，免得改了一边忘了另一边
    from_file = load_sft_config(ROOT / "configs" / "finetune" / "sft_r8.yaml")

    assert replace(from_file, out_dir=SFTConfig().out_dir) == SFTConfig()


# ---- 训练计划 ----


def test_every_epoch_is_shuffled_again(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=2)
    items = copy_task(48)

    plan = training_plan(items, config)

    # 每个 epoch 6 步，每步 8 条；两个 epoch 都用到全部 48 条，但顺序不同
    first = [i for step in plan[:6] for micro in step for i in micro]
    second = [i for step in plan[6:] for micro in step for i in micro]
    assert sorted(first) == sorted(second) == list(range(48))
    assert first != second


def varied_lengths(n: int) -> list[tuple[list[int], list[int]]]:
    """答案长度从 1 到 n 不等的样本：第 i 条的输入有 i + 2 个 token。"""
    from tlab.finetune.sft_labels import build_example

    return [build_example([1, 2], [3] * i, EOS) for i in range(1, n + 1)]


def test_memory_probe_runs_one_step_with_the_longest_examples(tmp_path: Path) -> None:
    items = varied_lengths(12)  # 输入长度 3 到 14
    config = tiny_config(tmp_path, memory_probe=True)

    plan = training_plan(items, config)
    info = train_lora(tiny_lora_model(), items, items[:4], config, torch.device("cpu"))

    # 显存由最长的那一批决定：短测只跑这一步，用最长的 4 × 2 = 8 条
    assert len(plan) == 1
    used = sorted(i for micro in plan[0] for i in micro)
    assert used == list(range(4, 12))  # 第 4 到第 11 条最长
    rows = read_metrics(Path(config.out_dir))
    assert int(rows[0]["tokens"]) == sum(i + 2 for i in range(5, 13))
    assert info["steps_run"] == info["total_steps"] == 1


def test_an_existing_run_is_not_overwritten(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    train_lora(
        tiny_lora_model(), copy_task(48), copy_task(8), config, torch.device("cpu")
    )

    # 换种子再跑时忘了改 out_dir，会把上一次的结果覆盖掉
    with pytest.raises(ValueError, match="out_dir"):
        train_lora(
            tiny_lora_model(), copy_task(48), copy_task(8), config, torch.device("cpu")
        )


@torch.no_grad()
def test_val_loss_is_weighted_by_answer_tokens_across_batches() -> None:
    model = with_random_b(tiny_lora_model())
    items = varied_lengths(5)  # 答案长度 1 到 5，加 eos 后要算的 token 是 2 到 6 个

    # batch_size=2 时分成 3 批，答案长度不同；按 token 加权的话和一次算完一样
    split = answer_loss(model, items, 2, EOS, torch.device("cpu"))
    whole = answer_loss(model, items, 5, EOS, torch.device("cpu"))

    assert split == pytest.approx(whole, rel=1e-5)


# ---- F2 的配置 ----


@pytest.mark.parametrize("r", [4, 8, 16, 32])
def test_f2_configs_differ_from_the_base_config_only_in_r_and_out_dir(r: int) -> None:
    base = load_sft_config(ROOT / "configs" / "finetune" / "sft_r8.yaml")
    f2 = load_sft_config(ROOT / "configs" / "finetune" / f"f2_r{r}.yaml")

    # 只改 r 一个变量；alpha 固定 16，所以缩放 alpha/r 会跟着 r 变
    assert f2.r == r
    assert f2.out_dir == f"runs/m5/f2_r{r}"
    assert replace(f2, r=8, out_dir=base.out_dir) == base
