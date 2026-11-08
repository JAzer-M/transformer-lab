"""M5 的汇总（agent 编写）：从 experiments/m5_gsm8k/ 下的原始结果生成 SUMMARY.md 的表格。

表里的数字都从 json 和 csv 算出来，不手抄；测试核对 SUMMARY.md 和 README 与它一致。
判定规则同 M3：两个均值之差大于双方种子间差距中的较大者，才算有结论。
"""

import csv
import json
from pathlib import Path
from typing import Any

RANKS = (4, 8, 16, 32)
SEEDS = (1337, 2024)
BASE_PARAMS = 596_049_920  # Qwen3-0.6B-Base 的参数量（M4 手算并核对过）


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _eval_cells(report: dict[str, Any]) -> list[str]:
    items = report["items"]
    stopped = sum(item["stopped_by_eos"] for item in items)
    new_tokens = sum(item["new_tokens"] for item in items) / len(items)
    return [
        f"{report['correct']}/{report['total']}",
        _pct(report["accuracy"]),
        f"{stopped}/{report['total']}",
        f"{new_tokens:.1f}",
    ]


def verdict(diff: float, gap: float) -> str:
    """M3 的判定规则：差值比种子间差距大才算有结论。"""
    if abs(diff) <= gap:
        return "差异不显著"
    return "更好" if diff > 0 else "更差"


def load_m5(exp_dir: Path) -> dict[str, Any]:
    f2 = exp_dir / "f2"
    runs = {
        (r, seed): {
            "run": _json(f2 / f"r{r}_seed{seed}_run.json"),
            "eval": _json(f2 / f"r{r}_seed{seed}_eval.json"),
        }
        for r in RANKS
        for seed in SEEDS
    }
    return {
        "base_few": _json(exp_dir / "f1_base_fewshot_rmsnorm32.json"),
        "base_zero": _json(exp_dir / "f1_base_zeroshot_rmsnorm32.json"),
        "runs": runs,
        "probe": _json(f2 / "probe_r32_b2_run.json"),
        "f4": _json(exp_dir / "f4" / "compare.json"),
        "f4_run": _json(exp_dir / "f4" / "f4_peft_r8_seed1337_run.json"),
        "f4_metrics": _rows(exp_dir / "f4" / "f4_peft_r8_seed1337_metrics.csv"),
        "ours_metrics": _rows(f2 / "r8_seed1337_metrics.csv"),
        "first_steps": [
            _rows(f2 / f"r{r}_seed{s}_metrics.csv")[0] for r in RANKS for s in SEEDS
        ],
        # 第 1 批：rmsnorm_fp32 打开之前的基座，用来看 bf16 的数值本身有多大波动
        "base_zero_before": _json(exp_dir / "f1_base_zeroshot.json"),
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rank_stats(data: dict[str, Any], r: int) -> dict[str, float]:
    acc = [data["runs"][(r, s)]["eval"]["accuracy"] for s in SEEDS]
    val = [data["runs"][(r, s)]["run"]["final_val_loss"] for s in SEEDS]
    return {
        "acc": acc,
        "mean": sum(acc) / len(acc),
        "gap": abs(acc[0] - acc[1]),
        "val_mean": sum(val) / len(val),
    }


def paired(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, int, int]:
    """同一批题逐题比较：(两边都对, 只有 a 对, 只有 b 对)。"""
    pairs = list(zip(a["items"], b["items"], strict=True))
    both = sum(x["correct"] and y["correct"] for x, y in pairs)
    only_a = sum(x["correct"] and not y["correct"] for x, y in pairs)
    only_b = sum(y["correct"] and not x["correct"] for x, y in pairs)
    return both, only_a, only_b


def f1_table(data: dict[str, Any]) -> str:
    header = "| 模型 | 提示 | 答对 | 准确率 | 在 eos 停下 | 平均新 token |\n| --- | --- | --- | --- | --- | --- |"
    rows = [
        ["基座", "few-shot（4 个例子）", *_eval_cells(data["base_few"])],
        ["基座", "不给例子", *_eval_cells(data["base_zero"])],
    ]
    for seed in SEEDS:
        rows.append(
            [
                f"LoRA SFT r=8，种子 {seed}",
                "不给例子",
                *_eval_cells(data["runs"][(8, seed)]["eval"]),
            ]
        )
    correct = sum(v["eval"]["correct"] for v in data["runs"].values())
    total = sum(v["eval"]["total"] for v in data["runs"].values())
    rows.append(
        [
            "LoRA SFT，8 次合计",
            "不给例子",
            f"{correct}/{total}",
            _pct(correct / total),
            "—",
            "—",
        ]
    )
    return "\n".join([header, *("| " + " | ".join(row) + " |" for row in rows)])


def f2_table(data: dict[str, Any]) -> str:
    header = (
        "| r | 可训练参数 | 占全部参数 | 种子 1337 | 种子 2024 | 均值 | 种子间差距 | 与 r=8 差 | 结论 "
        "| 最后的 val loss（均值） | 显存峰值 | 训练用时 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    control = rank_stats(data, 8)
    rows = []
    for r in RANKS:
        stats = rank_stats(data, r)
        run = data["runs"][(r, 1337)]["run"]
        peak = max(data["runs"][(r, s)]["run"]["peak_memory_mb"] for s in SEEDS)
        minutes = (
            sum(data["runs"][(r, s)]["run"]["train_seconds"] for s in SEEDS)
            / len(SEEDS)
            / 60
        )
        diff = stats["mean"] - control["mean"]
        rows.append(
            [
                str(r),
                f"{run['n_trainable']:,}",
                f"{100 * run['n_trainable'] / BASE_PARAMS:.2f}%",
                *(_pct(a) for a in stats["acc"]),
                _pct(stats["mean"]),
                f"{100 * stats['gap']:.1f}",
                "—" if r == 8 else f"{100 * diff:+.1f}",
                "对照" if r == 8 else verdict(diff, max(stats["gap"], control["gap"])),
                f"{stats['val_mean']:.4f}",
                f"{peak:,.0f} MiB",
                f"{minutes:.1f} 分钟",
            ]
        )
    return "\n".join([header, *("| " + " | ".join(row) + " |" for row in rows)])


def f4_table(data: dict[str, Any]) -> str:
    f4, run = data["f4"], data["f4_run"]
    ours = data["runs"][(8, 1337)]["run"]
    lines = [
        "| 项目 | 自研 LoRA | PEFT |",
        "| --- | --- | --- |",
        f"| 最后的 val loss | {ours['final_val_loss']} | {run['final_val_loss']} |",
        f"| 显存峰值 | {ours['peak_memory_mb']:,.0f} MiB | {run['peak_memory_mb']:,.0f} MiB |",
        f"| 训练用时 | {ours['train_seconds'] / 60:.1f} 分钟 | {run['train_seconds'] / 60:.1f} 分钟 |",
        "",
        "| 比较 | 数值 |",
        "| --- | --- |",
        f"| 步数 | {f4['steps']} |",
        f"| 第 1 步的 train loss 差 | {f4['first_step_train_diff']} |",
        f"| 每步 train loss 差的平均 / 最大 | {f4['mean_abs_train_diff']} / {f4['max_abs_train_diff']} |",
        f"| 最后的 val loss 差 | {f4['final_val_diff']} |",
        f"| 参照：自研换种子，最后的 val loss 差 | {f4['seed_gap_final_val']:.5f} |",
        f"| 门槛（跑之前定好） | 每步平均 ≤ {f4['thresholds']['mean_abs_train_diff']}，最后 val ≤ {f4['thresholds']['final_val_diff']} |",
        f"| 结论 | {'达标' if f4['passed'] else '未达标'} |",
    ]
    return "\n".join(lines)


def render_m5_summary(exp_dir: Path) -> str:
    data = load_m5(exp_dir)
    control = rank_stats(data, 8)
    base_few, base_zero = data["base_few"]["accuracy"], data["base_zero"]["accuracy"]
    both, only_base, only_sft = paired(
        data["base_zero"], data["runs"][(8, 1337)]["eval"]
    )
    capped = sum(
        not item["stopped_by_eos"]
        and item["new_tokens"] == data["base_zero"]["config"]["max_new_tokens"]
        for item in data["base_zero"]["items"]
    )
    probe = data["probe"]["peak_memory_mb"]
    columns = ("train_loss", "val_loss", "lr", "grad_norm")
    identical = sum(
        all(a[c] == b[c] for c in columns)
        for a, b in zip(data["ours_metrics"], data["f4_metrics"], strict=True)
    )
    first_loss = sum(float(row["train_loss"]) for row in data["first_steps"]) / len(
        data["first_steps"]
    )
    before = data["base_zero_before"]
    same_text = sum(
        a["output"] == b["output"]
        for a, b in zip(before["items"], data["base_zero"]["items"], strict=True)
    )
    peaks = {
        r: max(data["runs"][(r, s)]["run"]["peak_memory_mb"] for s in SEEDS)
        for r in RANKS
    }
    f4 = data["f4"]
    return f"""# M5：LoRA SFT 的结果

本文件由 `scripts/summarize_m5.py` 从 `experiments/m5_gsm8k/` 下的原始结果生成，不要手改。数字都是 RTX 5060 Laptop（8 GB）上实际跑出来的。

## 设置

- **基座：** Qwen/Qwen3-0.6B-Base（{BASE_PARAMS:,} 个参数），用自研架构载入，`rmsnorm_fp32` 打开。
- **数据：** GSM8K 训练集 7,473 条，留 200 条验证，7,273 条训练；评测固定用测试集里的 500 题。
- **训练：** LoRA 装在每层的 q、k、v、out 投影上，alpha 固定 16，学习率 2e-4（warmup 20 步，cosine 降到 2e-5），微批 2 × 累积 8，1 个 epoch（454 步），只对答案算 loss，bf16 autocast。
- **评测：** 贪心解码，最多 256 个新 token；500 题的标准误在准确率 50% 附近约 2.2 个百分点。
- **判定规则：** 同 M3，两个均值之差大于双方种子间差距中的较大者，才算有结论。

## F1：微调前后

{f1_table(data)}

- **准确率：** SFT r=8 两个种子的均值 {_pct(control["mean"])}，比基座 few-shot 高 {100 * (control["mean"] - base_few):.1f} 个百分点，比不给例子的基座高 {100 * (control["mean"] - base_zero):.1f} 个，都不超过两个种子之间的 {100 * control["gap"]:.1f}，差异不显著。
- **逐题配对：** 基座不给例子和 SFT r=8 种子 1337 比，两边都对 {both} 题，只有基座对 {only_base} 题，只有 SFT 对 {only_sft} 题。
- **学到的是格式和停下：** 基座不给例子时有 {capped} 题写满了 {data["base_zero"]["config"]["max_new_tokens"]} 个 token；SFT 之后绝大多数题都在 eos 停下，按 `#### 数字` 作答。

## F2：LoRA 的 rank

{f2_table(data)}

各 r 之间的差别都不超过种子间的差距；最后的 val loss 几乎相同。r=32 的可训练参数是 r=4 的 8 倍，显存峰值只多了 {peaks[32] - peaks[4]:.0f} MiB。r=32 最长一批的显存短测峰值 {probe:,.1f} MiB。

## F4：自研 LoRA 和 PEFT

PEFT（{data["f4_run"]["peft_version"]}，transformers {data["f4_run"]["transformers_version"]}）从自研那次训练（r=8、种子 1337）存下的初始 A、B 出发，数据、顺序、训练循环和超参数都相同，只有 LoRA 的实现不同。

{f4_table(data)}

两次训练的 metrics.csv 里，train loss、val loss、学习率和梯度范数全部逐字相同的有 {identical}/{len(data["ours_metrics"])} 步。

## 读这些结果时要注意

- **准确率没有明显提高。** 8 次训练第 1 步的 train loss（这时 B 还是 0，等于基座在这批题上的 loss）平均 {first_loss:.3f}，训练完的 val loss 约 {control["val_mean"]:.3f}，loss 降了不少，但准确率的提升在噪声范围内。原因没有验证；推断之一是基座在预训练时已经见过大量类似的数学数据，GSM8K 训练集教会它的主要是格式和停下。
- **bf16 的数值本身就有波动。** 打开 `rmsnorm_fp32` 前后，基座不给例子的准确率是 {_pct(before["accuracy"])} 和 {_pct(base_zero)}，500 题里只有 {same_text} 题的生成文本完全相同。
- **F4 只比较了一次**（r=8、种子 1337）；PEFT 训练出的 LoRA 没有评测准确率。
- **没有试过** 更多 epoch、更大的学习率、全参微调（F5）和完整的 1,319 题测试集。
- F4 的门槛是 {f4["thresholds"]["mean_abs_train_diff"]}，实测差值是 {f4["mean_abs_train_diff"]}。
"""
