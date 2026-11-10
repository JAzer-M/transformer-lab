from pathlib import Path

# 核对 README 里的结果和原始结果一致，防止抄错或过期

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "m3_ablations"


def readme_rows() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.startswith("|")]


def cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_readme_scores_match_the_full_validation_results() -> None:
    from tlab.eval.ablation import load_variants

    rows = readme_rows()
    for name, variant in load_variants(RESULTS).items():
        scores = [run.full_val_loss for _, run in sorted(variant.runs.items())]
        expected = [f"{s:.4f}" for s in scores] + [f"{sum(scores) / len(scores):.4f}"]
        matching = [r for r in rows if r.startswith(f"| `{name}` |")]
        assert any(all(value in cells(r) for value in expected) for r in matching), name


def summary_text() -> str:
    return (RESULTS / "SUMMARY.md").read_text(encoding="utf-8")


def test_summary_file_matches_the_raw_results() -> None:
    from tlab.eval.ablation import (
        load_bench,
        load_cache_speed,
        load_variants,
        render_summary,
    )

    # 结果表可复现：磁盘上的 SUMMARY.md 必须是脚本用原始结果生成的，不是手改或过期的
    variants = load_variants(RESULTS, config_dir=ROOT / "configs" / "ablations")
    generated = render_summary(variants, load_bench(RESULTS), load_cache_speed(RESULTS))
    assert summary_text() == generated
    # 短测只加在最后，前面的几节和没有短测时逐字相同
    assert generated.startswith(render_summary(variants).rstrip("\n"))


def test_readme_means_match_the_summary_file() -> None:
    rows = readme_rows()
    checked = 0
    # 后面短测的表也有 10 列，这里只看前面的结果表
    results_part = summary_text().split("## 速度与显存短测")[0]
    for line in results_part.splitlines():
        parts = cells(line)
        # 汇总表的行：组 | 变体 | 参数量 | 各种子 | 均值 | 差距 | 与对照差 | 结论 | 依据 | 步数
        if len(parts) == 10 and parts[1].startswith("a") and parts[1][1].isdigit():
            name, params, mean, diff, verdict = (
                parts[1],
                parts[2],
                parts[4],
                parts[6],
                parts[7],
            )
            matching = [r for r in rows if r.startswith(f"| `{name}` |")]
            assert any(
                all(v in cells(r) for v in (params, mean, diff, verdict))
                for r in matching
            ), name
            checked += 1
    assert checked == 9


def test_readme_extrapolation_matches_the_results() -> None:
    from tlab.eval.ablation import load_variants

    rows = readme_rows()
    checked = 0
    for name, variant in load_variants(RESULTS).items():
        for seed, run in variant.runs.items():
            if run.extrapolation is None:
                continue
            expected = [
                str(seed),
                f"{run.extrapolation['first_half_loss']:.4f}",
                f"{run.extrapolation['second_half_loss']:.4f}",
            ]
            matching = [r for r in rows if r.startswith(f"| `{name}` |")]
            assert any(all(v in cells(r) for v in expected) for r in matching), name
            checked += 1
    assert checked == 4


def test_readme_cache_speed_matches_the_benchmark_results() -> None:
    from tlab.eval.ablation import load_cache_speed

    rows = readme_rows()
    speed = load_cache_speed(RESULTS)
    assert len(speed) == 7
    for name, result in speed:
        plain = result["generate_no_cache"]["tokens_per_sec"]
        cached = result["generate_cache"]["tokens_per_sec"]
        expected = [
            f"`{name}`",
            str(result["generate_batch_size"]),
            f"{plain:,.0f}",
            f"{cached:,.0f}",
            f"{cached / plain:.2f}",
        ]
        assert any(all(v in cells(r) for v in expected) for r in rows), expected


def test_readme_train_speed_matches_the_benchmark_results() -> None:
    from tlab.eval.ablation import load_bench

    bench = load_bench(RESULTS)
    base = bench["a0_control"]["train"]
    for name, result in bench.items():
        rows = [r for r in readme_rows() if r.startswith(f"| `{name}` |")]
        train = result["train"]
        if name in ("a1_sinusoidal", "a2_postln"):
            continue  # README 只列差别够大或和 A4 有关的变体
        expected = [
            f"{train['tokens_per_sec']:,.0f}",
            f"{train['peak_memory_mb']:,.0f}",
        ]
        if name != "a0_control":
            speed = train["tokens_per_sec"] - base["tokens_per_sec"]
            expected.append(f"{speed / base['tokens_per_sec']:+.1%}")
        assert any(all(v in cells(r) for v in expected) for r in rows), name


def test_readme_links_and_figures_exist() -> None:
    import re

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)#]+)\)", text)
    local = [t for t in targets if not t.startswith("http")]
    assert len(local) > 10
    for target in local:
        assert (ROOT / target).exists(), target


def test_readme_sample_excerpts_come_from_the_samples_file() -> None:
    import re

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    samples = (RESULTS / "samples.md").read_text(encoding="utf-8")
    section = text.split("### 生成样例")[1].split("###")[0]

    excerpts = re.findall(r"```text\n(.*?)\n```", section, re.S)

    assert len(excerpts) == 3
    # 节选必须一字不差地来自模型的输出，结尾的省略号是 README 加的
    for excerpt in excerpts:
        assert excerpt.endswith("……")
        assert excerpt.removesuffix("……") in samples


def test_readme_alignment_matches_the_report() -> None:
    import json

    report = json.loads(
        (ROOT / "experiments" / "m4_qwen3" / "alignment.json").read_text(
            encoding="utf-8"
        )
    )
    # 这份记录要来自干净的提交，并且通过了验收，才能写进 README
    assert report["passed"] is True
    assert not report["git_commit"].endswith("-dirty")
    assert report["dtype"] == "torch.float32"
    assert report["n_params"] == 596_049_920

    rows = readme_rows()
    for prompt in report["prompts"]:
        assert prompt["same_with_cache"] and prompt["same_without_cache"]
        expected = [
            str(prompt["n_tokens"]),
            f"{prompt['logits_max_error']:.2e}",
            f"{prompt['logits_max_abs']:.1f}",
            f"{max(prompt['layer_relative_errors']):.2e}",
        ]
        assert any(all(v in cells(r) for v in expected) for r in rows), expected


def test_official_scaling_run_really_has_no_error() -> None:
    import json

    # README 说改成官方写法后误差是 0，证据在这个文件里
    path = ROOT / "experiments" / "m4_qwen3" / "alignment_official_scaling.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["max_error"] == 0.0
    assert all(max(p["layer_max_errors"]) == 0.0 for p in report["prompts"])
