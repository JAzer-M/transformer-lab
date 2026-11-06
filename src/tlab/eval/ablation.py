"""消融结果汇总：读取每次训练回传的 run.json 与 metrics.csv，按变体合并种子，与对照组比较。

目录结构（见 docs/m3-windows-run.md）：<root>/<变体>/seed<种子>/{run.json, metrics.csv}，
第 2 批起每个种子目录还有完整验证集评估的 full_val.json（scripts/eval_full_val.py），
A1 的变体另有外推评测的 extrapolation.json（scripts/eval_extrapolation.py）。
"""

import csv
import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from tlab.train.config import TrainConfig, load_config

# 同一变体的几个种子，只允许这些配置字段不同
PER_SEED_KEYS = ("seed", "out_dir")
# 判定规则（PLAN「M3 实施计划」）：至少 2 个种子；补到第 3 个种子后不再看配对方向
MIN_SEEDS = 2
SEEDS_AFTER_EXTRA = 3
# 发散判定：与第 250 步比较；比此前最低点高 0.5 以上（含 0.5）
EARLY_STEP = 250
REBOUND = 0.5
# 收敛速度看第一次 val ≤ 2.0 的步数
CONVERGE_AT = 2.0

# 变体名就是 configs/ablations/<变体>.yaml 的文件名，也是结果目录名
CONTROL = "a0_control"
GROUPS = {
    "A1 位置编码": ["a1_sinusoidal", "a1_rope"],
    "A2 归一化": ["a2_postln", "a2_rmsnorm"],
    "A3 MLP": ["a3_swiglu"],
    "A4 注意力": ["a4_gqa2", "a4_mqa"],
}
STRESS = ["a2_postln_lr3e-3", "a2_preln_lr3e-3", "a2_rmsnorm_lr3e-3"]  # 高学习率加测
REFERENCE = ["a9_modern"]  # RoPE + RMSNorm + SwiGLU + GQA 组合，仅作参考


@dataclass
class SeedRun:
    """一个变体在一个种子下的一次训练。"""

    seed: int
    final_val_loss: float
    n_params: int
    evals: list[tuple[int, float]]  # 每次评估的 (step, val loss)
    train_losses: list[float]  # 每次评估的 train loss，以及最终的 train loss
    config: dict
    full_val_loss: float | None = None  # 完整验证集上的 val loss；还没评估时为 None
    extrapolation: dict | None = None  # 2 倍长度上的外推评测；没做时为 None


@dataclass
class Variant:
    """同一个配置文件、不同种子的几次训练。"""

    name: str
    runs: dict[int, SeedRun] = field(default_factory=dict)  # 种子 → 训练

    @property
    def finals(self) -> list[float]:
        return [run.final_val_loss for run in self.runs.values()]

    @property
    def has_nan(self) -> bool:
        return any(not math.isfinite(v) for v in self.finals)

    @property
    def mean_val(self) -> float:
        # 有 NaN 时直接给 NaN，不让结果随种子顺序变化
        return math.nan if self.has_nan else sum(self.finals) / len(self.finals)

    @property
    def spread(self) -> float:
        # 种子间差距：最大减最小；只有一个种子时为 0
        return math.nan if self.has_nan else max(self.finals) - min(self.finals)

    @property
    def has_full_val(self) -> bool:
        return all(run.full_val_loss is not None for run in self.runs.values())

    @property
    def has_nan_in_either_score(self) -> bool:
        # 训练末尾的抽样估计与完整验证集分数，任何一个不是有限数都不能下结论
        scores = [run.final_val_loss for run in self.runs.values()]
        scores += [
            r.full_val_loss for r in self.runs.values() if r.full_val_loss is not None
        ]
        return any(not math.isfinite(s) for s in scores)

    @property
    def n_params(self) -> int:
        return next(iter(self.runs.values())).n_params


def compare(variant: Variant, control: Variant) -> str:
    """按 PLAN「M3 实施计划」的判定规则，给出变体相对对照组的结论（val loss 越低越好）。"""
    if variant.has_nan or control.has_nan:
        return "无法比较"
    # 规范要求主结论至少 2 个种子
    if min(len(variant.runs), len(control.runs)) < MIN_SEEDS:
        return "种子不足"
    # 同一个种子配对比较：方向相反说明差异还淹没在随机性里，先补第 3 个种子；
    # 补到 3 个种子以后不再看配对方向，只按下面的均值规则下结论
    if min(len(variant.runs), len(control.runs)) < SEEDS_AFTER_EXTRA:
        paired = [
            variant.runs[s].final_val_loss - control.runs[s].final_val_loss
            for s in sorted(variant.runs.keys() & control.runs.keys())
        ]
        if any(d < 0 for d in paired) and any(d > 0 for d in paired):
            return "需补种子"
    # 均值差要超过两者种子间差距中较大的那个，才算有结论
    diff = variant.mean_val - control.mean_val
    if abs(diff) <= max(variant.spread, control.spread):
        return "差异不显著"
    return "更好" if diff < 0 else "更差"


def divergence(
    evals: list[tuple[int, float]],
    final_val: float,
    train_losses: list[float] | tuple[float, ...] = (),
) -> str | None:
    """判断一次训练是否发散；发散或无法判定时返回原因，没有发散返回 None。"""
    vals = [val for _, val in evals] + [final_val] + list(train_losses)
    if any(not math.isfinite(v) for v in vals):
        return "出现 NaN 或 inf"
    # 一路上（含最终那次评估）val 比此前最低点高出 REBOUND 及以上：训练中途跑飞过
    best = math.inf
    for step, val in [*evals, ("最终", final_val)]:
        if val >= best + REBOUND:
            return (
                f"第 {step} 步 val {val:.4f} 比此前最低 {best:.4f} 反弹 {REBOUND} 以上"
            )
        best = min(best, val)
    # 训练完还不如第 EARLY_STEP 步：越训越差
    early = dict(evals).get(EARLY_STEP)
    if early is None:
        return f"无法判定：没有第 {EARLY_STEP} 步的评估"
    if final_val > early:
        return f"最终 val {final_val:.4f} 高于第 {EARLY_STEP} 步的 {early:.4f}"
    return None


def steps_to_threshold(evals: list[tuple[int, float]], threshold: float) -> int | None:
    """收敛速度：第一次评估 val loss ≤ threshold 的步数；从未达到时返回 None。"""
    for step, val in evals:
        if val <= threshold:
            return step
    return None


def _read_run(run_dir: Path) -> SeedRun:
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with open(run_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    train_losses = [float(r["train_loss"]) for r in rows]
    if "final_train_loss" in info:
        train_losses.append(float(info["final_train_loss"]))
    max_iters = int(info["config"]["max_iters"])
    full_val = _read_final_eval(run_dir / "full_val.json", max_iters)
    return SeedRun(
        seed=int(info["config"]["seed"]),
        final_val_loss=float(info["final_val_loss"]),
        n_params=int(info["n_params"]),
        evals=[(int(r["step"]), float(r["val_loss"])) for r in rows],
        train_losses=train_losses,
        config=info["config"],
        full_val_loss=None if full_val is None else float(full_val["val_loss"]),
        extrapolation=_read_final_eval(run_dir / "extrapolation.json", max_iters),
    )


def _read_final_eval(path: Path, max_iters: int) -> dict | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    # 只认训练结束时那个存档的分数：评成中途存档或放错目录时直接报错，不混进汇总
    if result["ckpt_step"] != max_iters:
        raise ValueError(
            f"{path} 评的是第 {result['ckpt_step']} 步的存档，不是最终的第 {max_iters} 步"
        )
    return result


# 字段是后来加的（如 M3 的 norm 开关）时，之前存的 run.json 里没有它；新字段的默认值
# 就是加之前的行为，所以缺了按 TrainConfig 的默认值比较，旧种子和新种子才对得上
DEFAULT_CONFIG = asdict(TrainConfig())


def _shared_config(config: dict) -> dict:
    full = {**DEFAULT_CONFIG, **config}
    return {k: v for k, v in full.items() if k not in PER_SEED_KEYS}


def _check_against_config_file(run: SeedRun, name: str, config_dir: Path) -> None:
    # 目录名说是这个变体，实际配置就得和它的配置文件一致（只有种子和输出目录可以不同）。
    # 正弦和 RoPE 参数量相同，只看数字发现不了跑错结构。名单外的变体没有配置文件，不核对
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        return
    expected = _shared_config(asdict(load_config(path)))
    actual = _shared_config(run.config)
    diff = sorted(
        k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)
    )
    if diff:
        raise ValueError(f"{name} 的种子 {run.seed} 与配置文件 {path} 不同：{diff}")


def load_variants(root: Path, config_dir: Path | None = None) -> dict[str, Variant]:
    """读取 root 下所有 <变体>/seed<种子>/，按变体名合并；给了 config_dir 就与配置文件核对。"""
    variants: dict[str, Variant] = {}
    for run_json in sorted(Path(root).glob("*/seed*/run.json")):
        run = _read_run(run_json.parent)
        name = run_json.parent.parent.name
        if config_dir is not None:
            _check_against_config_file(run, name, Path(config_dir))
        variant = variants.setdefault(name, Variant(name))
        # 同一变体的种子之间，除 seed、out_dir 外配置必须相同：防止复制错目录、混进别的实验
        if variant.runs:
            first = _shared_config(next(iter(variant.runs.values())).config)
            this = _shared_config(run.config)
            diff = sorted(
                k for k in first.keys() | this.keys() if first.get(k) != this.get(k)
            )
            if diff:
                raise ValueError(f"{name} 的种子 {run.seed} 与其他种子配置不同：{diff}")
        variant.runs[run.seed] = run
    return variants


# ---------- 比较依据 ----------

FULL_VAL = "完整验证集"
SAMPLED = "20 batch 抽样估计"


def _with_full_val(variant: Variant) -> Variant:
    # 每个种子的最终 val 换成完整验证集的分数，其余（评估曲线、配置）不变
    runs = {
        seed: replace(run, final_val_loss=run.full_val_loss)
        for seed, run in variant.runs.items()
    }
    return Variant(variant.name, runs)


def _on_same_basis(*variants: Variant) -> tuple[list[Variant], str]:
    """每个种子都有完整验证集分数时都换成它；缺一个就都用 run.json 的抽样估计，不混着比。"""
    if all(v.has_full_val for v in variants):
        return [_with_full_val(v) for v in variants], FULL_VAL
    return list(variants), SAMPLED


# ---------- Markdown 汇总表 ----------

CONVERGE_COLUMN = f"首次 val≤{CONVERGE_AT} 的步数"
VARIANT_HEADER = (
    f"| 组 | 变体 | 参数量 | 各种子最终 val | 均值 | 种子间差距 | 与对照差 | 结论 | 依据 "
    f"| {CONVERGE_COLUMN} |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _per_seed(variant: Variant, fmt: Callable[[SeedRun], str]) -> str:
    # 按种子从小到大，用 " / " 隔开
    return " / ".join(fmt(variant.runs[s]) for s in sorted(variant.runs))


def _converge_cell(run: SeedRun) -> str:
    step = steps_to_threshold(run.evals, CONVERGE_AT)
    return "—" if step is None else str(step)


def _stress_status(run: SeedRun) -> str:
    # 是否发散按事先定的规则；没发散、却一直没降到收敛线的另外标出，判定本身不改
    reason = divergence(run.evals, run.final_val_loss, run.train_losses)
    if reason:
        return reason
    # 周期评估和训练结束时的最终评估都没到收敛线，才算未收敛
    if (
        steps_to_threshold(run.evals, CONVERGE_AT) is None
        and run.final_val_loss > CONVERGE_AT
    ):
        return f"未发散，但未收敛（从未达到 val≤{CONVERGE_AT}）"
    return "未发散"


def _variant_row(group: str, variant: Variant, control: Variant | None) -> str:
    if control is None or variant is control:
        (shown,), basis = _on_same_basis(variant)
        diff, verdict = "—", "—"
    else:
        (shown, shown_control), basis = _on_same_basis(variant, control)
        if variant.has_nan_in_either_score or control.has_nan_in_either_score:
            verdict = "无法比较"
        else:
            verdict = compare(shown, shown_control)
        gap = shown.mean_val - shown_control.mean_val
        diff = f"{gap:+.4f}" if math.isfinite(gap) else "—"
        # 对照那一行显示的是完整分数、这一行却退回抽样估计时，写出对照的抽样均值，
        # 读表的人才能对上“与对照差”
        if basis == SAMPLED and control.has_full_val:
            basis += f"（对照抽样均值 {shown_control.mean_val:.4f}）"
    return _md_row(
        [
            group,
            variant.name,
            f"{variant.n_params:,}",
            _per_seed(shown, lambda r: f"{r.final_val_loss:.4f}"),
            f"{shown.mean_val:.4f}",
            f"{shown.spread:.4f}",
            diff,
            verdict,
            basis,
            _per_seed(variant, _converge_cell),
        ]
    )


def _extrapolation_lines(variants: dict[str, Variant]) -> list[str]:
    runs = [
        (variant.name, seed, run)
        for variant in variants.values()
        for seed, run in sorted(variant.runs.items())
        if run.extrapolation is not None
    ]
    if not runs:
        return []
    lines = [
        "",
        "## A1 外推",
        "",
        "窗口长度是训练长度的 2 倍：前半段的位置都训练过，后半段超出了训练长度。"
        "注意后半段的上下文也更长，所以差值同时包含“位置超出训练范围”和“能看的内容变多”"
        "两个影响，不能全算到其中一个头上。",
        "",
        "| 变体 | 种子 | 窗口长度 | 完整验证集 val | 前半段 | 后半段 | 后半段 − 前半段 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, seed, run in runs:
        first = run.extrapolation["first_half_loss"]
        second = run.extrapolation["second_half_loss"]
        full = "—" if run.full_val_loss is None else f"{run.full_val_loss:.4f}"
        lines.append(
            _md_row(
                [
                    name,
                    str(seed),
                    str(run.extrapolation["length"]),
                    full,
                    f"{first:.4f}",
                    f"{second:.4f}",
                    f"{second - first:+.4f}",
                ]
            )
        )
    return lines + ["", f"{CONTROL}：可学习位置表只有训练长度那么多行，不支持外推。"]


def load_bench(root: Path) -> dict[str, dict] | None:
    """读第 5 批的短测 bench.json：变体 → 结果；没有这个文件时返回 None。"""
    path = root / "bench.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache_speed(root: Path) -> list[tuple[str, dict]] | None:
    """读第 5 批补充 bench_batch/ 下的每个 json：[(变体, 结果)]；没有时返回 None。"""
    # 这个目录里还有 Codex 的执行记录，短测结果的文件名是 b<B>.json、cpu_b<B>.json
    rows = [
        item
        for path in sorted((root / "bench_batch").glob("*.json"))
        if re.fullmatch(r"(cpu_)?b\d+", path.stem)
        for item in json.loads(path.read_text(encoding="utf-8")).items()
    ]
    return rows or None


def _format_or_dash(value: float | None, fmt: str) -> str:
    return "—" if value is None else format(value, fmt)


def _bench_lines(bench: dict[str, dict]) -> list[str]:
    control = bench[CONTROL]["train"]
    runs = [run["tokens_per_sec"] for run in control["runs"]]
    spread = (max(runs) - min(runs)) / control["tokens_per_sec"]
    lines = [
        "",
        "## 速度与显存短测",
        "",
        "用随机权重和随机 token，不读数据，和正式训练相同地累积、裁剪、更新；"
        "每项取中位数，原始数字见 bench.json。",
        f"对照组自己 {len(runs)} 次的训练吞吐相差 {spread:.1%}，比这小的差别不算结论。",
        "",
        "| 变体 | 参数量 | 训练 tokens/s | 相对对照 | 训练峰值 MiB | 相对对照 "
        "| 完整缓存 MiB（B=1） | 缓存类型（k / v） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in bench.items():
        speed = result["train"]["tokens_per_sec"]
        peak = result["train"]["peak_memory_mb"]
        relative_speed = relative_peak = "—"
        if name != CONTROL:
            base = control["tokens_per_sec"]
            relative_speed = f"{(speed - base) / base:+.1%}"
            if peak is not None and control["peak_memory_mb"] is not None:
                relative_peak = f"{peak - control['peak_memory_mb']:+,.0f}"
        dtype = result["cache_dtype"]
        lines.append(
            _md_row(
                [
                    name,
                    f"{result['n_params']:,}",
                    f"{speed:,.0f}",
                    relative_speed,
                    _format_or_dash(peak, ",.0f"),
                    relative_peak,
                    f"{result['cache_bytes_full_window'] / 2**20:.2f}",
                    f"{dtype['k']} / {dtype['v']}",
                ]
            )
        )
    return lines


def _cache_speed_lines(rows: list[tuple[str, dict]]) -> list[str]:
    first = rows[0][1]
    lines = [
        "",
        "## A7 KV 缓存的生成速度",
        "",
        f"随机权重，贪心生成：开头 {first['prompt_len']} 个 token，"
        f"新生成 {first['new_tokens']} 个。tokens/s 是整个 batch 的总和，"
        "倍数是有缓存除以无缓存，都取中位数。",
        "",
        "| 设备 | 变体 | B | 无缓存 tokens/s | 有缓存 tokens/s | 倍数 "
        "| 有缓存最慢–最快 | 无缓存峰值 MiB | 有缓存峰值 MiB | 次数 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    # GPU 在前；同一设备按变体和 B 排
    ordered = sorted(
        rows,
        key=lambda row: (
            row[1]["device"].startswith("cpu"),
            row[1]["device"],
            row[0],
            row[1]["generate_batch_size"],
        ),
    )
    for name, result in ordered:
        plain, cached = result["generate_no_cache"], result["generate_cache"]
        speeds = [run["tokens_per_sec"] for run in cached["runs"]]
        lines.append(
            _md_row(
                [
                    result["device"],
                    name,
                    str(result["generate_batch_size"]),
                    f"{plain['tokens_per_sec']:,.0f}",
                    f"{cached['tokens_per_sec']:,.0f}",
                    f"{cached['tokens_per_sec'] / plain['tokens_per_sec']:.2f}",
                    f"{min(speeds):,.0f}–{max(speeds):,.0f}",
                    _format_or_dash(plain["peak_memory_mb"], ",.0f"),
                    _format_or_dash(cached["peak_memory_mb"], ",.0f"),
                    str(len(speeds)),
                ]
            )
        )
    return lines


def render_summary(
    variants: dict[str, Variant],
    bench: dict[str, dict] | None = None,
    cache_speed: list[tuple[str, dict]] | None = None,
) -> str:
    """生成 SUMMARY.md 的内容；还没跑的变体直接跳过，名单外的变体单独列出。"""
    control = variants.get(CONTROL)
    lines = [
        "# M3 消融汇总",
        "",
        "由 `scripts/summarize_ablations.py` 自动生成，数字取自各次训练回传的 run.json 与 full_val.json；"
        "判定规则见 docs/PLAN.md「M3 实施计划」。速度与显存另做短测，不在本表。",
        "",
        f"最终 val 优先用{FULL_VAL}的分数（full_val.json）；一行里对照或变体只要有一个"
        f"种子缺，这一行两边都用 run.json 的 {SAMPLED}，“依据”一列注明。",
        "",
        f"## 单变量对比（对照组 {CONTROL}）",
        "",
    ]
    if control is None:
        lines += [f"尚无对照组 {CONTROL}，暂不能给出结论。", ""]
    lines.append(VARIANT_HEADER)
    if control is not None:
        lines.append(_variant_row("对照", control, control))
    for group, names in GROUPS.items():
        for name in names:
            if name in variants:
                lines.append(_variant_row(group, variants[name], control))

    lines += _extrapolation_lines(variants)
    lines += [
        "",
        "## A2 高学习率加测",
        "",
        f"| 变体 | 种子 | 最终 val | {CONVERGE_COLUMN} | 是否发散 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in STRESS:
        for seed, run in sorted(variants.get(name, Variant(name)).runs.items()):
            final = f"{run.final_val_loss:.4f}"
            lines.append(
                _md_row(
                    [name, str(seed), final, _converge_cell(run), _stress_status(run)]
                )
            )

    reference = [variants[n] for n in REFERENCE if n in variants]
    planned = {CONTROL, *STRESS, *REFERENCE, *(n for g in GROUPS.values() for n in g)}
    others = [v for n, v in sorted(variants.items()) if n not in planned]
    for title, extra in (
        ("参考：组合（不算单变量结论）", reference),
        ("其他（不在计划名单里）", others),
    ):
        if extra:
            lines += ["", f"## {title}", "", VARIANT_HEADER]
            lines += [_variant_row("—", v, control) for v in extra]
    # 短测只加在最后，前面几节不变
    if bench is not None:
        lines += _bench_lines(bench)
    if cache_speed is not None:
        lines += _cache_speed_lines(cache_speed)
    return "\n".join(lines) + "\n"
