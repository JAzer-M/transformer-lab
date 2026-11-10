"""M5 的汇总：SUMMARY.md 和 README 里的数字必须和原始结果一致。"""

import json
from pathlib import Path

import pytest

from tlab.eval.m5_summary import (
    f1_table,
    f2_table,
    f4_table,
    load_m5,
    render_m5_summary,
    verdict,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "m5_gsm8k"


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_summary_file_matches_the_raw_results() -> None:
    expected = render_m5_summary(RESULTS)

    assert (RESULTS / "SUMMARY.md").read_text(encoding="utf-8") == expected


@pytest.mark.parametrize(
    ("diff", "gap", "expected"),
    [
        (0.013, 0.032, "差异不显著"),  # 差 1.3 个百分点，种子差距 3.2
        (-0.002, 0.032, "差异不显著"),
        (0.05, 0.03, "更好"),
        (-0.05, 0.03, "更差"),
        (0.03, 0.03, "差异不显著"),  # 相等时不算有结论
    ],
)
def test_verdict_follows_the_m3_rule(diff: float, gap: float, expected: str) -> None:
    assert verdict(diff, gap) == expected


def raw_eval(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def table_rows(table: str) -> list[list[str]]:
    return [
        [c.strip() for c in line.strip("|").split("|")]
        for line in table.splitlines()[2:]
    ]


def test_f1_table_counts_come_from_the_eval_files() -> None:
    rows = table_rows(f1_table(load_m5(RESULTS)))
    few = raw_eval("f1_base_fewshot_rmsnorm32.json")
    zero = raw_eval("f1_base_zeroshot_rmsnorm32.json")
    sft = raw_eval("f2/r8_seed1337_eval.json")

    # 直接从每题的对错数，不用汇总里的函数
    assert rows[0][2] == f"{sum(i['correct'] for i in few['items'])}/500" == "234/500"
    assert rows[1][2] == f"{sum(i['correct'] for i in zero['items'])}/500" == "242/500"
    assert rows[2][2] == f"{sum(i['correct'] for i in sft['items'])}/500" == "254/500"
    total = sum(
        raw_eval(f"f2/r{r}_seed{s}_eval.json")["correct"]
        for r in (4, 8, 16, 32)
        for s in (1337, 2024)
    )
    assert rows[4][2] == f"{total}/4000"


def test_f2_table_uses_both_seeds_of_every_rank() -> None:
    rows = {row[0]: row for row in table_rows(f2_table(load_m5(RESULTS)))}

    # r=8：50.8% 和 47.6%，均值 49.2%，差 3.2 个百分点（手算）
    assert rows["8"][3:7] == ["50.8%", "47.6%", "49.2%", "3.2"]
    assert rows["8"][8] == "对照"
    assert rows["8"][1] == "2,293,760"  # M5 计划里手算的 286,720 × 8
    assert {rows[r][8] for r in ("4", "16", "32")} == {"差异不显著"}


def test_f4_table_reports_the_comparison_file() -> None:
    compare = raw_eval("f4/compare.json")
    table = f4_table(load_m5(RESULTS))

    assert compare["passed"] is True
    assert (
        f"| 每步 train loss 差的平均 / 最大 | {compare['mean_abs_train_diff']} / {compare['max_abs_train_diff']} |"
        in table
    )
    assert "| 结论 | 达标 |" in table


@pytest.mark.parametrize("table", [f1_table, f2_table, f4_table])
def test_readme_tables_are_the_generated_ones(table) -> None:
    assert table(load_m5(RESULTS)) in readme()


def test_readme_bf16_alignment_matches_the_reports() -> None:
    before = raw_eval("base_alignment_cuda_bf16.json")["max_error"]
    after = raw_eval("base_alignment_cuda_bf16_rmsnorm_fp32.json")["max_error"]

    assert f"bf16 的分数最大误差从 {before} 变成 {after}" in readme()
