import json
import subprocess
import sys
from pathlib import Path

import pytest

# 待实现模块在各测试内导入，与其他测试文件保持一致

ROOT = Path(__file__).resolve().parents[1]
EVALS = [(0, 9.0), (250, 2.5), (3051, 1.6)]


def write_run(
    root: Path, variant: str, seed: int, evals: list[tuple[int, float]] = EVALS
) -> None:
    """按回传的目录结构写一次假训练：<变体>/seed<种子>/run.json 和 metrics.csv。"""
    run_dir = root / variant / f"seed{seed}"
    run_dir.mkdir(parents=True)
    info = {
        "config": {"seed": seed, "max_iters": 3052, "out_dir": str(run_dir)},
        "n_params": 100,
        "final_val_loss": evals[-1][1],
    }
    (run_dir / "run.json").write_text(json.dumps(info), encoding="utf-8")
    rows = ["step,train_loss,val_loss,lr"]
    rows += [f"{step},{val},{val},0.001" for step, val in evals]
    (run_dir / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_all_variants(root: Path, skip: str | None = None) -> None:
    from tlab.eval.plots import FIGURES

    names = {name for _, variants in FIGURES.values() for name in variants}
    for name in names - {skip}:
        write_run(root, name, 1337)
        write_run(root, name, 2024)


def test_curve_is_the_mean_of_the_seeds(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants
    from tlab.eval.plots import curve_of

    write_run(tmp_path, "a1_rope", 1337, [(0, 9.0), (250, 3.0), (3051, 2.0)])
    write_run(tmp_path, "a1_rope", 2024, [(0, 9.0), (250, 2.0), (3051, 1.0)])

    curve = curve_of(load_variants(tmp_path)["a1_rope"])

    assert curve.steps == [0, 250, 3051]
    assert curve.mean == [9.0, 2.5, 1.5]
    assert curve.seeds == {1337: [9.0, 3.0, 2.0], 2024: [9.0, 2.0, 1.0]}


def test_curve_rejects_seeds_with_different_steps(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants
    from tlab.eval.plots import curve_of

    write_run(tmp_path, "a1_rope", 1337, [(0, 9.0), (250, 3.0)])
    write_run(tmp_path, "a1_rope", 2024, [(0, 9.0), (500, 3.0)])

    # 不报错的话，不同步数的 loss 会被悄悄平均到一起
    with pytest.raises(ValueError, match="a1_rope"):
        curve_of(load_variants(tmp_path)["a1_rope"])


def test_missing_metrics_names_the_variant(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants

    write_run(tmp_path, "a1_rope", 1337)
    (tmp_path / "a1_rope" / "seed1337" / "metrics.csv").unlink()

    with pytest.raises(FileNotFoundError, match="a1_rope"):
        load_variants(tmp_path)


def test_figure_draws_the_mean_of_each_variant_from_the_first_shown_step(
    tmp_path: Path,
) -> None:
    from tlab.eval.ablation import load_variants
    from tlab.eval.plots import build_figure, curve_of

    write_run(tmp_path, "a1_rope", 1337, [(0, 9.0), (250, 3.0), (3051, 2.0)])
    write_run(tmp_path, "a1_rope", 2024, [(0, 9.0), (250, 2.0), (3051, 1.0)])

    figure = build_figure("title", [curve_of(load_variants(tmp_path)["a1_rope"])])

    lines = {line.get_label(): line for line in figure.axes[0].lines}
    mean = lines["RoPE"]
    # 第 0 步的 loss 约 9，会把后面的差别压扁，所以不画
    assert list(mean.get_xdata()) == [250, 3051]
    assert list(mean.get_ydata()) == [2.5, 1.5]
    # 每个种子各有一条淡线，图例里只有均值
    assert len(figure.axes[0].lines) == 3
    legend = figure.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["RoPE"]


def test_each_variant_keeps_its_color_in_every_figure(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants
    from tlab.eval.plots import FIGURES, build_figure, curve_of

    write_all_variants(tmp_path)
    variants = load_variants(tmp_path)

    colors: dict[str, set[str]] = {}
    for title, names in FIGURES.values():
        figure = build_figure(title, [curve_of(variants[name]) for name in names])
        for line in figure.axes[0].lines:
            if not line.get_label().startswith("_"):
                colors.setdefault(line.get_label(), set()).add(line.get_color())

    # 同一个名字（比如 control、Post-LN）不管出现在哪张图都是同一个颜色
    assert all(len(found) == 1 for found in colors.values()), colors


def test_each_figure_has_distinct_labels_and_colors() -> None:
    from tlab.eval.plots import COLORS, FIGURES, LABELS

    for _, names in FIGURES.values():
        labels = [LABELS[name] for name in names]
        assert len(set(labels)) == len(labels), labels
        assert len({COLORS[label] for label in labels}) == len(labels), labels
    # 高学习率那张图和普通那张图，同一种结构用同一个名字
    assert LABELS["a2_postln_lr3e-3"] == LABELS["a2_postln"]
    assert LABELS["a2_rmsnorm_lr3e-3"] == LABELS["a2_rmsnorm"]


def run_script(root: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_ablations.py"),
            str(root),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


def test_script_writes_every_figure_as_png(tmp_path: Path) -> None:
    from tlab.eval.plots import FIGURES

    write_all_variants(tmp_path / "results")
    out = tmp_path / "figures"

    done = run_script(tmp_path / "results", out)

    assert done.returncode == 0, done.stderr
    assert {path.name for path in out.iterdir()} == {f"{key}.png" for key in FIGURES}
    for path in out.iterdir():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_script_names_the_missing_variant(tmp_path: Path) -> None:
    write_all_variants(tmp_path / "results", skip="a4_mqa")

    done = run_script(tmp_path / "results", tmp_path / "figures")

    assert done.returncode != 0
    assert "a4_mqa" in done.stderr
    # 先检查再画，缺变体时不留下只画了一半的目录
    assert not (tmp_path / "figures").exists()


def test_zoom_panel_shows_the_end_and_ignores_curves_far_above() -> None:
    from tlab.eval.plots import Curve, build_figure

    steps = [250, 2000, 3051]
    near = Curve("control", steps, {1: [3.0, 1.6, 1.5]}, [3.0, 1.6, 1.5])
    close = Curve("RMSNorm", steps, {1: [3.1, 1.7, 1.6]}, [3.1, 1.7, 1.6])
    # 高学习率的 Post-LN 停在 5.8，算进去的话其他曲线的差别又被压扁了
    far = Curve("Post-LN", steps, {1: [5.8, 5.8, 5.8]}, [5.8, 5.8, 5.8])

    zoom = build_figure("title", [near, close, far]).axes[1]

    assert zoom.get_xlim()[0] == 2000
    low, high = zoom.get_ylim()
    assert 1.4 < low < 1.5
    assert 1.7 < high < 1.8
