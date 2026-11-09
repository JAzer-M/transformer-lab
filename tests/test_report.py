"""实验报告 docs/report.md 里的表和数字必须来自原始结果（会被简历和面试引用）。

原始结果优先查 experiments/ 下的文件；只在 README 或 PROGRESS 留了记录的
（比如没有存档的 40 题试跑），查那里。
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tlab.eval.m5_summary import f1_table, f2_table, f4_table, load_m5

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report.md"


def report() -> str:
    return REPORT.read_text(encoding="utf-8")


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_m3_table_is_copied_from_the_summary_file() -> None:
    summary = source("experiments/m3_ablations/SUMMARY.md").splitlines()
    start = next(
        i for i, line in enumerate(summary) if line.startswith("| 组 | 变体 |")
    )
    end = summary.index("", start)  # 表格到第一个空行为止
    table = "\n".join(summary[start:end])

    assert table in report()


@pytest.mark.parametrize("table", [f1_table, f2_table, f4_table])
def test_m5_tables_are_the_generated_ones(
    table: Callable[[dict[str, Any]], str],
) -> None:
    assert table(load_m5(ROOT / "experiments" / "m5_gsm8k")) in report()


# 报告正文里的数字 → 它在原始结果里的样子
M3 = "experiments/m3_ablations"
M4 = "experiments/m4_qwen3/alignment.json"
NUMBERS = [
    ("10,750,080", "experiments/m1_shakespeare_char/run.json", '"n_params": 10750080'),
    ("1.5858", "experiments/results.md", "val 1.5858"),
    ("105,912 tokens/s", "experiments/results.md", "| 105912 |"),
    ("2,597 MiB", "experiments/results.md", "| 2597 MB |"),
    ("1.3531", "experiments/results.md", "val 1.3531"),
    ("82,756 tokens/s", "experiments/results.md", "| 82756 |"),
    ("5,007 MiB", "experiments/results.md", "| 5007 MB |"),
    ("13,969,152", f"{M3}/SUMMARY.md", "13,969,152"),
    ("5,414,547", f"{M3}/BATCH2_SUPPLEMENT_REPORT.md", "5,414,547"),
    ("1.73e-04", M4, '"logits_max_error": 0.00017261505126953125'),
    ("4.77e-05", M4, '"logits_max_error": 4.76837158203125e-05'),
    ("4.24e-05", M4, '"logits_max_error": 4.2438507080078125e-05'),
    ("596,049,920", M4, "596049920"),
    (
        "0.65625",
        "experiments/m5_gsm8k/base_alignment_cuda_bf16.json",
        '"max_error": 0.65625',
    ),
    ("后半段 loss 是 2.30 和 2.13", f"{M3}/SUMMARY.md", "| 2.3010 |"),
    ("后半段 loss 是 2.30 和 2.13", f"{M3}/SUMMARY.md", "| 2.1253 |"),
    ("正弦是 4.93 和 5.06", f"{M3}/SUMMARY.md", "| 4.9251 |"),
    ("正弦是 4.93 和 5.06", f"{M3}/SUMMARY.md", "| 5.0564 |"),
    ("5.83–5.85", "README.md", "| 5.8343 | 5.8352 |"),
    ("快 6.80 倍", f"{M3}/bench_batch/b32.md", "| 6.80 | 662 | 267 | 4.50 |"),
    ("从 662 MiB 降到 267 MiB", f"{M3}/bench_batch/b32.md", "| 662 | 267 |"),
    ("快 7.42 倍", f"{M3}/bench_batch/cpu_b1.md", "| 7.42 |"),
    ("MHA 的 4.50 MiB", f"{M3}/bench_batch/b32.md", "| 267 | 4.50 |"),
    ("GQA-2 的 1.50 MiB", f"{M3}/bench_batch/b32_console.txt", "| 167 | 1.50 |"),
    (
        "267 MiB 降到 GQA-2 的 167 MiB",
        f"{M3}/bench_batch/b32_console.txt",
        "| 658 | 167 |",
    ),
    ("7,331 MiB", "experiments/m5_gsm8k/f2/probe_r32_b2_run.json", "7330.5"),
    ("均值 49.2%", "experiments/m5_gsm8k/SUMMARY.md", "两个种子的均值 49.2%"),
    (
        "不超过两个种子之间的 3.2",
        "experiments/m5_gsm8k/SUMMARY.md",
        "两个种子之间的 3.2",
    ),
    ("旧规则 5/40", "PROGRESS.md", "旧规则 5/40"),
    ("新规则 22/40", "PROGRESS.md", "新规则 22/40"),
]


@pytest.mark.parametrize(("in_report", "path", "in_source"), NUMBERS)
def test_numbers_in_the_text_come_from_the_results(
    in_report: str, path: str, in_source: str
) -> None:
    assert in_report in report()
    assert in_source in source(path)
