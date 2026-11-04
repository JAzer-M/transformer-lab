import csv
import json
from pathlib import Path

import pytest

# 消融结果汇总（agent 编写）；预期值都是手算好的具体数字，不用实现里的算法重算


def write_run(
    root: Path,
    variant: str,
    seed: int,
    final_val: float,
    evals: list[tuple[int, float]] | None = None,
    tokens_per_sec: float = 80000.0,
    n_params: int = 13969152,
    **config_overrides,
) -> None:
    """按 Codex 回传的目录结构写一次假训练：<变体>/seed<种子>/run.json 与 metrics.csv。"""
    run_dir = root / variant / f"seed{seed}"
    run_dir.mkdir(parents=True)
    config = {
        "d_model": 384,
        "max_iters": 3052,
        "seed": seed,
        "out_dir": f"runs/m3/{variant}/seed{seed}",
    }
    config.update(config_overrides)
    run_info = {
        "config": config,
        "n_params": n_params,
        "tokens_per_sec": tokens_per_sec,
        "final_val_loss": final_val,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info), encoding="utf-8")
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "lr"])
        for step, val in evals or [(0, 9.0), (3051, final_val)]:
            writer.writerow([step, val, val, 6e-4])


# ---------- 1. 读取结果目录 ----------


def test_load_variants_merges_seeds_of_one_variant(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants

    write_run(tmp_path, "a0_control", 1337, 1.50)
    write_run(tmp_path, "a0_control", 2024, 1.54)

    variants = load_variants(tmp_path)

    control = variants["a0_control"]
    assert sorted(control.runs) == [1337, 2024]
    assert control.mean_val == pytest.approx(1.52)
    # 种子间差距 = 最大 − 最小
    assert control.spread == pytest.approx(0.04)
    assert control.n_params == 13969152


def test_runs_saved_before_a_new_switch_count_as_its_default(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants

    # 第 1 批对照组是加 norm 开关之前存的，config 里没有这两项；新代码补跑的种子有，取默认值
    write_run(tmp_path, "a0_control", 1337, 1.50)
    write_run(tmp_path, "a0_control", 2024, 1.54)
    write_run(tmp_path, "a0_control", 7, 1.52, norm="layernorm", norm_position="pre")

    assert sorted(load_variants(tmp_path)["a0_control"].runs) == [7, 1337, 2024]

    # 结构真的不同时仍要报错
    write_run(tmp_path, "a0_control", 8, 1.52, norm="rmsnorm", norm_position="pre")
    with pytest.raises(ValueError):
        load_variants(tmp_path)


# ---------- 2. 配置一致性 ----------


def test_load_variants_rejects_seeds_with_different_configs(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants

    # seed 与 out_dir 本来就不同，其余字段必须一样；这里第二个种子的步数被改了
    write_run(tmp_path, "a1_rope", 1337, 1.48)
    write_run(tmp_path, "a1_rope", 2024, 1.49, max_iters=1000)

    with pytest.raises(ValueError, match="max_iters"):
        load_variants(tmp_path)


# ---------- 3. 收敛速度 ----------


def test_steps_to_threshold_finds_the_first_eval_at_or_below_it() -> None:
    from tlab.eval.ablation import steps_to_threshold

    evals = [(0, 9.0), (250, 2.6), (500, 2.0), (750, 1.9), (1000, 2.1)]

    # 第一次 ≤ 2.0 发生在第 500 步（恰好等于也算）；之后回升不影响
    assert steps_to_threshold(evals, 2.0) == 500
    # 一直没降到阈值：返回 None
    assert steps_to_threshold(evals, 1.5) is None


# ---------- 4. 与对照组比较 ----------


@pytest.mark.parametrize(
    ("finals", "verdict"),
    [
        # 对照组 1.50 / 1.54：均值 1.52，种子间差距 0.04
        ((1.40, 1.42), "更好"),  # 均值差 −0.11，超过 0.04，两个种子都更低
        ((1.60, 1.62), "更差"),  # 均值差 +0.09，超过 0.04，两个种子都更高
        ((1.49, 1.53), "差异不显著"),  # 两个种子都低 0.01，但均值差 0.01 小于 0.04
        # 均值差 −0.06 虽大于对照的 0.04，但小于变体自己的种子间差距 0.12：取较大者
        ((1.40, 1.52), "差异不显著"),
        ((1.45, 1.58), "需补种子"),  # 种子 1337 低 0.05，种子 2024 高 0.04，方向相反
    ],
)
def test_compare_with_control(
    tmp_path: Path, finals: tuple[float, float], verdict: str
) -> None:
    from tlab.eval.ablation import compare, load_variants

    write_run(tmp_path, "a0_control", 1337, 1.50)
    write_run(tmp_path, "a0_control", 2024, 1.54)
    write_run(tmp_path, "a3_swiglu", 1337, finals[0])
    write_run(tmp_path, "a3_swiglu", 2024, finals[1])

    variants = load_variants(tmp_path)

    assert compare(variants["a3_swiglu"], variants["a0_control"]) == verdict


def test_compare_needs_two_seeds_on_both_sides(tmp_path: Path) -> None:
    from tlab.eval.ablation import compare, load_variants

    write_run(tmp_path, "a0_control", 1337, 1.50)
    write_run(tmp_path, "a0_control", 2024, 1.54)
    write_run(tmp_path, "a3_swiglu", 1337, 1.30)  # 只跑完一个种子

    variants = load_variants(tmp_path)

    # 规范要求主结论至少 2 个种子：只有一个时不给“更好/更差”
    assert compare(variants["a3_swiglu"], variants["a0_control"]) == "种子不足"
    assert compare(variants["a0_control"], variants["a3_swiglu"]) == "种子不足"


def test_compare_with_nan_is_not_comparable(tmp_path: Path) -> None:
    from tlab.eval.ablation import compare, load_variants

    write_run(tmp_path, "a0_control", 1337, 1.50)
    write_run(tmp_path, "a0_control", 2024, 1.54)
    write_run(tmp_path, "a1_rope", 1337, float("nan"))
    write_run(tmp_path, "a1_rope", 2024, 1.40)

    variants = load_variants(tmp_path)

    # 有一次训练发散成 NaN：不能和对照比，也不能反过来拿它当对照
    assert compare(variants["a1_rope"], variants["a0_control"]) == "无法比较"
    assert compare(variants["a0_control"], variants["a1_rope"]) == "无法比较"


def test_compare_with_three_seeds_uses_the_mean_rule(tmp_path: Path) -> None:
    from tlab.eval.ablation import compare, load_variants

    # 对照 1.50 / 1.54 / 1.52：均值 1.52，差距 0.04
    # 变体 1.40 / 1.42 / 1.53：种子 1337、2024 更低，种子 7 更高（方向不一致）
    for seed, control, variant in [
        (1337, 1.50, 1.40),
        (2024, 1.54, 1.42),
        (7, 1.52, 1.53),
    ]:
        write_run(tmp_path, "a0_control", seed, control)
        write_run(tmp_path, "a3_swiglu", seed, variant)

    variants = load_variants(tmp_path)

    # 补到第 3 个种子后不再看配对方向，只按均值规则：
    # 均值差 −0.07 小于变体自身差距 0.13 → 差异不显著（而不是一直“需补种子”）
    assert compare(variants["a3_swiglu"], variants["a0_control"]) == "差异不显著"


# ---------- 5. 是否发散（A2 高学习率加测） ----------


@pytest.mark.parametrize(
    ("evals", "final_val", "reason"),
    [
        # 正常下降：不算发散
        ([(0, 9.0), (250, 3.0), (500, 2.2), (3051, 1.6)], 1.6, None),
        # 出现 NaN
        ([(0, 9.0), (250, float("nan"))], float("nan"), "NaN"),
        # 最终 3.2 比第 250 步的 3.0 还高（反弹 0.4，没超过 0.5）
        ([(0, 9.0), (250, 3.0), (500, 2.8), (3051, 3.2)], 3.2, "第 250 步"),
        # 第 750 步 2.6 比此前最低的 2.0 高出 0.6；最终虽然降回 1.9，仍算发散过
        ([(0, 9.0), (250, 3.0), (500, 2.0), (750, 2.6), (3051, 1.9)], 1.9, "反弹"),
        # 最终 2.1 比此前最低的 1.5 高出 0.6：最后一次评估也要检查
        ([(0, 9.0), (250, 3.0), (500, 1.5)], 2.1, "反弹"),
        # 正好高 0.5 也算（“0.5 以上”包含 0.5）
        ([(0, 9.0), (250, 3.0), (500, 2.0), (750, 2.5)], 1.9, "反弹"),
        # 没有第 250 步的评估：不能说“未发散”，要说无法判定
        ([(0, 9.0), (500, 2.0)], 1.9, "无法判定"),
    ],
)
def test_divergence_reasons(
    evals: list[tuple[int, float]], final_val: float, reason: str | None
) -> None:
    from tlab.eval.ablation import divergence

    result = divergence(evals, final_val)

    if reason is None:
        assert result is None
    else:
        assert reason in result


def test_divergence_checks_train_loss_too() -> None:
    from tlab.eval.ablation import divergence

    evals = [(0, 9.0), (250, 3.0), (500, 2.0)]

    # val 正常，但 train loss 出现了 NaN：PLAN 写的是“任一 loss 为 NaN”
    assert "NaN" in divergence(evals, 1.9, train_losses=[9.0, float("nan")])


# ---------- 6. 汇总表 ----------


def test_render_summary_lists_control_variants_stress_and_others(
    tmp_path: Path,
) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    write_run(tmp_path, "a0_control", 1337, 1.50, evals=[(0, 9.0), (500, 2.0)])
    write_run(tmp_path, "a0_control", 2024, 1.54, evals=[(0, 9.0), (750, 1.95)])
    write_run(tmp_path, "a3_swiglu", 1337, 1.40, n_params=13969152)
    write_run(tmp_path, "a3_swiglu", 2024, 1.42, n_params=13969152)
    nan = float("nan")
    write_run(tmp_path, "a2_postln_lr3e-3", 1337, nan, evals=[(0, 9.0), (250, nan)])
    write_run(tmp_path, "x_misc", 1337, 1.70)

    text = render_summary(load_variants(tmp_path))

    # 只在表格行里找（标题里也会出现变体名）
    lines = [line for line in text.splitlines() if line.startswith("|")]
    control_row = next(line for line in lines if "a0_control" in line)
    swiglu_row = next(line for line in lines if "a3_swiglu" in line)
    stress_row = next(line for line in lines if "a2_postln_lr3e-3" in line)
    # 对照组：两个种子、均值 1.5200、差距 0.0400、首次 val ≤ 2.0 的步数 500 / 750
    assert "1.5000" in control_row and "1.5400" in control_row
    assert "1.5200" in control_row and "0.0400" in control_row
    assert "500 / 750" in control_row
    # 变体：均值 1.4100、比对照低 0.1100，结论“更好”
    assert "1.4100" in swiglu_row and "-0.1100" in swiglu_row and "更好" in swiglu_row
    # 高学习率加测：写出发散原因
    assert "NaN" in stress_row
    # 不在名单里的变体单独列出，不会被悄悄丢掉
    assert "x_misc" in text
    # 速度另做短测（PLAN「M3 实施计划」），汇总表不放长训练的 tokens/s，免得被拿来下结论
    assert "tokens/s" not in text


def test_render_summary_marks_nan_variant_as_not_comparable(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    write_run(tmp_path, "a0_control", 1337, 1.50)
    write_run(tmp_path, "a0_control", 2024, 1.54)
    write_run(tmp_path, "a1_rope", 1337, float("nan"))
    write_run(tmp_path, "a1_rope", 2024, 1.40)

    text = render_summary(load_variants(tmp_path))

    rope_row = next(line for line in text.splitlines() if line.startswith("| A1"))
    assert "无法比较" in rope_row and "更差" not in rope_row


def test_render_summary_says_when_control_is_missing(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    write_run(tmp_path, "a3_swiglu", 1337, 1.40)

    text = render_summary(load_variants(tmp_path))

    # 还没有对照组时不能给出结论，要明确写出来
    assert "尚无对照组" in text
    assert "更好" not in text


def test_summarize_script_writes_summary_file(tmp_path: Path) -> None:
    import subprocess
    import sys

    write_run(tmp_path, "a0_control", 1337, 1.50)
    out = tmp_path / "SUMMARY.md"

    # Codex 每跑完一批就运行这条命令（docs/m3-windows-run.md）
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_ablations.py",
            str(tmp_path),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )

    assert "a0_control" in out.read_text(encoding="utf-8")


# ---------- 完整验证集评估（第 2 批起，见 scripts/eval_full_val.py） ----------


def write_full_val(
    root: Path, variant: str, seed: int, val: float, ckpt_step: int = 3052
) -> None:
    """在已有的种子目录里放一份 full_val.json，与 Codex 回传的一致。"""
    path = root / variant / f"seed{seed}" / "full_val.json"
    # ckpt_step 与 write_run 的 max_iters 一致：是训练结束时那个存档的分数
    result = {"val_loss": val, "n_tokens": 5414547, "ckpt_step": ckpt_step}
    path.write_text(json.dumps(result), encoding="utf-8")


def write_close_pair(root: Path) -> None:
    # 抽样估计（run.json）：对照均值 1.5776、差距 0.0176，变体均值 1.5725，
    # 低 0.0051 没超过 0.0176 → 差异不显著
    write_run(root, "a0_control", 1337, 1.5864)
    write_run(root, "a0_control", 2024, 1.5688)
    write_run(root, "a2_rmsnorm", 1337, 1.5800)
    write_run(root, "a2_rmsnorm", 2024, 1.5650)


def test_render_summary_prefers_full_validation_scores(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    write_close_pair(tmp_path)
    # 完整验证集：对照 1.5780 / 1.5770（均值 1.5775、差距 0.0010），
    # 变体 1.5700 / 1.5694（均值 1.5697、差距 0.0006）；均值低 0.0078 > 0.0010，
    # 两个种子配对都是变体更低 → 更好
    for variant, seed, val in [
        ("a0_control", 1337, 1.5780),
        ("a0_control", 2024, 1.5770),
        ("a2_rmsnorm", 1337, 1.5700),
        ("a2_rmsnorm", 2024, 1.5694),
    ]:
        write_full_val(tmp_path, variant, seed, val)

    text = render_summary(load_variants(tmp_path))

    control_row = next(line for line in text.splitlines() if "| a0_control |" in line)
    rms_row = next(line for line in text.splitlines() if "| a2_rmsnorm |" in line)
    assert "1.5780 / 1.5770" in control_row and "完整验证集" in control_row
    assert "1.5700 / 1.5694" in rms_row and "1.5697" in rms_row
    assert "-0.0078" in rms_row and "更好" in rms_row and "完整验证集" in rms_row


def test_render_summary_falls_back_when_any_seed_lacks_full_val(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    write_close_pair(tmp_path)
    # 对照两个种子都有完整分数，变体只有一个：这一行两边都退回抽样估计，免得混着比
    write_full_val(tmp_path, "a0_control", 1337, 1.5780)
    write_full_val(tmp_path, "a0_control", 2024, 1.5770)
    write_full_val(tmp_path, "a2_rmsnorm", 1337, 1.5700)

    text = render_summary(load_variants(tmp_path))

    rms_row = next(line for line in text.splitlines() if "| a2_rmsnorm |" in line)
    assert "1.5800 / 1.5650" in rms_row and "-0.0051" in rms_row
    assert "差异不显著" in rms_row and "20 batch" in rms_row
    # 对照那一行显示的是完整分数，这一行要写出对照的抽样均值，读表的人才能对上差值
    assert "1.5776" in rms_row


def test_nan_in_either_score_is_not_comparable(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    write_close_pair(tmp_path)
    for variant, seed, val in [
        ("a0_control", 1337, 1.5780),
        ("a0_control", 2024, 1.5770),
        ("a2_rmsnorm", 1337, 1.5700),
        ("a2_rmsnorm", 2024, 1.5694),
    ]:
        write_full_val(tmp_path, variant, seed, val)
    # 训练末尾的抽样估计是 NaN（训练中出过 NaN），即使完整分数有限，也不能下结论
    run_json = tmp_path / "a2_rmsnorm" / "seed2024" / "run.json"
    info = json.loads(run_json.read_text(encoding="utf-8"))
    info["final_val_loss"] = float("nan")
    run_json.write_text(json.dumps(info), encoding="utf-8")

    text = render_summary(load_variants(tmp_path))

    rms_row = next(line for line in text.splitlines() if "| a2_rmsnorm |" in line)
    assert "无法比较" in rms_row and "更好" not in rms_row


def test_full_val_of_an_earlier_checkpoint_is_rejected(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants

    write_run(tmp_path, "a0_control", 1337, 1.50)
    # 评的是第 1000 步的中途存档，不是训练结束（max_iters = 3052）的模型：不能混进汇总
    write_full_val(tmp_path, "a0_control", 1337, 1.49, ckpt_step=1000)

    with pytest.raises(ValueError):
        load_variants(tmp_path)


def test_stress_table_marks_runs_that_never_converged(tmp_path: Path) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    # Post-LN 高学习率的实测形状：没有 NaN、没有反弹、最终不高于第 250 步，但一直停在高位
    write_run(
        tmp_path,
        "a2_postln_lr3e-3",
        1337,
        5.84,
        evals=[(0, 9.1), (250, 5.85), (3051, 5.84)],
    )
    write_run(
        tmp_path,
        "a2_preln_lr3e-3",
        1337,
        1.47,
        evals=[(0, 9.1), (250, 2.7), (750, 1.95), (3051, 1.47)],
    )
    nan = float("nan")
    write_run(tmp_path, "a2_rmsnorm_lr3e-3", 1337, nan, evals=[(0, 9.1), (250, nan)])

    lines = render_summary(load_variants(tmp_path)).splitlines()

    stuck = next(line for line in lines if "| a2_postln_lr3e-3 |" in line)
    healthy = next(line for line in lines if "| a2_preln_lr3e-3 |" in line)
    broken = next(line for line in lines if "| a2_rmsnorm_lr3e-3 |" in line)
    # 事先定的发散判定不变，“未收敛”是另外标出来的
    assert "未发散" in stuck and "未收敛" in stuck
    assert "未发散" in healthy and "未收敛" not in healthy
    # 真发散的写发散原因，不再多标未收敛
    assert "NaN" in broken and "未收敛" not in broken


def test_stress_table_does_not_call_a_run_unconverged_if_final_val_reached_it(
    tmp_path: Path,
) -> None:
    from tlab.eval.ablation import load_variants, render_summary

    # 周期评估都没到 2.0，但最终那次评估到了：不能写“从未达到”
    write_run(
        tmp_path,
        "a2_preln_lr3e-3",
        1337,
        1.99,
        evals=[(0, 9.1), (250, 2.5), (3000, 2.01)],
    )

    lines = render_summary(load_variants(tmp_path)).splitlines()

    row = next(line for line in lines if "| a2_preln_lr3e-3 |" in line)
    assert "未发散" in row and "未收敛" not in row
