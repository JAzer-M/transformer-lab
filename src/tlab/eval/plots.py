"""M3 消融的 loss 曲线：每组一张图，把各变体叠在一起（M3 第 6 步）。

val loss 取自 metrics.csv，是训练中每 250 步按 20 个 batch 抽样的估计，所以末端和
结果表里的完整验证集分数略有差别。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tlab.eval.ablation import Variant, load_variants

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# 图的文件名 → (标题, 变体)。图上的文字用英文，不依赖中文字体
FIGURES = {
    "a1": (
        "A1 position encoding",
        ["a0_control", "a1_sinusoidal", "a1_rope"],
    ),
    "a2": ("A2 normalization", ["a0_control", "a2_postln", "a2_rmsnorm"]),
    "a2_lr": (
        "A2 normalization at learning rate 3e-3",
        ["a2_preln_lr3e-3", "a2_postln_lr3e-3", "a2_rmsnorm_lr3e-3"],
    ),
    "a3": ("A3 feed-forward", ["a0_control", "a3_swiglu"]),
    "a4": ("A4 attention", ["a0_control", "a4_gqa2", "a4_mqa"]),
    "a9": ("Combination (reference only)", ["a0_control", "a9_modern"]),
}
LABELS = {
    "a0_control": "control",
    "a1_sinusoidal": "sinusoidal",
    "a1_rope": "RoPE",
    "a2_postln": "Post-LN",
    "a2_rmsnorm": "RMSNorm",
    "a3_swiglu": "SwiGLU",
    "a4_gqa2": "GQA (2 groups)",
    "a4_mqa": "MQA (1 group)",
    "a9_modern": "RoPE + RMSNorm + SwiGLU + GQA",
    "a2_preln_lr3e-3": "Pre-LN LayerNorm",
    "a2_postln_lr3e-3": "Post-LN",
    "a2_rmsnorm_lr3e-3": "RMSNorm",
}
# 图上不画第 0 步：loss 约 9，会把后面的差别压扁
FIRST_SHOWN_STEP = 250
# 各变体最后的差别只有 0.01–0.05，整条曲线上看不出来，右边再放大这一段
ZOOM_FROM_STEP = 2000
# 放大时不管比最低的曲线高这么多的曲线（高学习率的 Post-LN 停在 5.8）
ZOOM_IGNORES_ABOVE = 1.0

# 配色取自 dataviz 技能的参考色板：前 3 个颜色两两之间都通过了色盲校验，每张图最多 3 条线。
# 颜色跟着名字走，同一个名字在每张图里都是同一个颜色
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {
    "control": BLUE,
    "Pre-LN LayerNorm": BLUE,
    "sinusoidal": ORANGE,
    "RoPE": AQUA,
    "Post-LN": ORANGE,
    "RMSNorm": AQUA,
    "SwiGLU": ORANGE,
    "GQA (2 groups)": ORANGE,
    "MQA (1 group)": AQUA,
    "RoPE + RMSNorm + SwiGLU + GQA": ORANGE,
}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


@dataclass
class Curve:
    """一个变体的 val loss 曲线：每个种子一条，加上它们的均值。"""

    label: str
    steps: list[int]
    seeds: dict[int, list[float]]
    mean: list[float]


def curve_of(variant: Variant) -> Curve:
    """把一个变体各种子的评估记录排成曲线；各种子的步数必须相同才能平均。"""
    runs = sorted(variant.runs.items())
    steps = [step for step, _ in runs[0][1].evals]
    seeds = {}
    for seed, run in runs:
        if [step for step, _ in run.evals] != steps:
            raise ValueError(f"{variant.name} 各种子的评估步数不同，不能平均")
        seeds[seed] = [val for _, val in run.evals]
    mean = [sum(vals) / len(vals) for vals in zip(*seeds.values(), strict=True)]
    return Curve(LABELS[variant.name], steps, seeds, mean)


def _draw_curves(axes: "Axes", curves: list[Curve], from_step: int) -> None:
    """粗线是均值，淡线是每个种子，末端的点标出最后一次评估。"""
    for curve in curves:
        color = COLORS[curve.label]
        shown = [i for i, s in enumerate(curve.steps) if s >= from_step]
        xs = [curve.steps[i] for i in shown]
        for vals in curve.seeds.values():
            ys = [vals[i] for i in shown]
            axes.plot(xs, ys, color=color, linewidth=0.8, alpha=0.35, label="_seed")
        axes.plot(
            xs,
            [curve.mean[i] for i in shown],
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            marker="o",
            markevery=[len(xs) - 1],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=curve.label,
        )


def _zoom_ylim(curves: list[Curve]) -> tuple[float, float]:
    """放大区的纵轴范围：只看和最低的曲线相差不远的那些。"""
    tails = [
        [
            val
            for step, val in zip(curve.steps, vals, strict=True)
            if step >= ZOOM_FROM_STEP
        ]
        for curve in curves
        for vals in curve.seeds.values()
    ]
    lowest = min(min(tail) for tail in tails)
    near = [
        v for tail in tails if min(tail) - lowest < ZOOM_IGNORES_ABOVE for v in tail
    ]
    margin = (max(near) - lowest) * 0.08 or 0.01
    return lowest - margin, max(near) + margin


def _style_axes(axes: "Axes") -> None:
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axes.grid(axis="y", color=GRIDLINE, linewidth=1)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE)
    axes.set_xlabel("Training step", color=TEXT_SECONDARY, fontsize=9)


def build_figure(title: str, curves: list[Curve]) -> "Figure":
    """画一张图：左边是整条曲线，右边放大最后一段。"""
    # 只在画图时才用 matplotlib（dev 依赖）；用 Figure 而不是 pyplot，不碰全局状态和窗口
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 4), facecolor=SURFACE, layout="constrained")
    figure.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=12)
    full, zoom = figure.subplots(1, 2, width_ratios=[3, 2])

    _draw_curves(full, curves, FIRST_SHOWN_STEP)
    _style_axes(full)
    full.set_facecolor(SURFACE)
    full.set_ylabel("Validation loss (sampled)", color=TEXT_SECONDARY, fontsize=9)
    full.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)

    _draw_curves(zoom, curves, ZOOM_FROM_STEP)
    _style_axes(zoom)
    zoom.set_facecolor(SURFACE)
    zoom.set_title(
        f"From step {ZOOM_FROM_STEP} (zoomed)",
        loc="left",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    last_step = max(curve.steps[-1] for curve in curves)
    zoom.set_xlim(ZOOM_FROM_STEP, last_step * 1.02)
    zoom.set_ylim(*_zoom_ylim(curves))
    return figure


def plot_all(root: Path, out_dir: Path, config_dir: Path | None = None) -> list[Path]:
    """读取结果目录，把 FIGURES 里的每张图写成 PNG，返回写出的路径。"""
    variants = load_variants(root, config_dir=config_dir)
    # 先检查再画，缺变体时不留下只画了一半的目录
    for _, names in FIGURES.values():
        for name in names:
            if name not in variants:
                raise ValueError(f"结果目录 {root} 里没有变体 {name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, (title, names) in FIGURES.items():
        figure = build_figure(title, [curve_of(variants[name]) for name in names])
        path = out_dir / f"{key}.png"
        figure.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
        paths.append(path)
    return paths
