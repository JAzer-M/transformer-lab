"""F4：比较两次训练的 loss 曲线，看是否“基本重合”。

参数给训练的输出目录（里面有 metrics.csv、run.json、lora_init.pt）。先核对两次训练
真的可比：起点相同（PEFT 那次的 init_lora 和自研那次的 lora_init.pt 是同一个文件），
配置除了 out_dir 和 init_lora 都相同；不可比就报错，不算结果。

门槛在正式训练之前就定下来（2026-11-08 用户确认），免得看到结果再改：
每一步 train loss 之差的平均 ≤ 0.01，最后的 val loss 之差 ≤ 0.01。
另外报告第 1 步的差：那时 B 还是 0，两边只差基座本身的数值，可以当底线。
--reference 给自研换一个种子的那次，作为参照：两种实现的差距应该比换种子小得多。
不达标时退出码是 1，结果照样写出来。

用法：
    uv run python scripts/compare_curves.py --ours runs/m5/f2_r8_seed1337 \
        --theirs runs/m5/f4_peft_r8_seed1337 --reference runs/m5/f2_r8_seed2024 \
        --out experiments/m5_gsm8k/f4/compare.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from tlab.train.utils import file_sha256

THRESHOLDS = {"mean_abs_train_diff": 0.01, "final_val_diff": 0.01}
# 两次训练允许不同的配置项：输出目录和起点的来源
ALLOWED_TO_DIFFER = {"out_dir", "init_lora"}


def read_metrics(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_run(run_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return read_metrics(run_dir / "metrics.csv"), info


def val_points(rows: list[dict[str, str]]) -> dict[int, float]:
    return {int(r["step"]): float(r["val_loss"]) for r in rows if r["val_loss"] != ""}


def check_same_setup(
    ours_dir: Path, ours: dict[str, Any], theirs: dict[str, Any]
) -> None:
    """起点和配置不同的话，曲线对不上也说明不了 LoRA 的实现有没有问题。"""
    start = (theirs.get("init_lora") or {}).get("sha256")
    if start != file_sha256(ours_dir / "lora_init.pt"):
        raise ValueError("起点不同：theirs 的 init_lora 不是 ours 的 lora_init.pt")
    differ = sorted(
        key
        for key in set(ours["config"]) | set(theirs["config"])
        if key not in ALLOWED_TO_DIFFER
        and ours["config"].get(key) != theirs["config"].get(key)
    )
    if differ:
        raise ValueError(f"配置不同：{differ}")


def compare(
    ours: list[dict[str, str]],
    theirs: list[dict[str, str]],
    reference: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """两次训练必须走完同样的步数，看 val loss 的步数也要相同。"""
    steps = [int(r["step"]) for r in ours]
    if steps != [int(r["step"]) for r in theirs]:
        raise ValueError(f"两次训练的步数不同：{len(ours)} 步和 {len(theirs)} 步")
    ours_val, theirs_val = val_points(ours), val_points(theirs)
    if ours_val.keys() != theirs_val.keys() or not ours_val:
        raise ValueError("两次训练看 val loss 的步数不同")

    diffs = [
        abs(float(a["train_loss"]) - float(b["train_loss"]))
        for a, b in zip(ours, theirs, strict=True)
    ]
    last = max(ours_val)
    report: dict[str, Any] = {
        "steps": len(steps),
        # 第 1 步时 B 还是 0：两边只差基座本身的数值
        "first_step_train_diff": diffs[0],
        "mean_abs_train_diff": sum(diffs) / len(diffs),
        "max_abs_train_diff": max(diffs),
        "val": [
            {
                "step": step,
                "ours": ours_val[step],
                "theirs": theirs_val[step],
                "diff": abs(ours_val[step] - theirs_val[step]),
            }
            for step in sorted(ours_val)
        ],
        "final_val_diff": abs(ours_val[last] - theirs_val[last]),
        "thresholds": THRESHOLDS,
    }
    if reference is not None:
        # 参照：同一个实现换一个种子。数据顺序不同，所以只比 val loss
        reference_val = val_points(reference)
        if last not in reference_val:
            raise ValueError(f"参照的那次训练在第 {last} 步没有 val loss")
        report["seed_gap_val"] = [
            {"step": step, "diff": abs(ours_val[step] - reference_val[step])}
            for step in sorted(ours_val)
            if step in reference_val
        ]
        report["seed_gap_final_val"] = abs(ours_val[last] - reference_val[last])
    report["passed"] = (
        report["mean_abs_train_diff"] <= THRESHOLDS["mean_abs_train_diff"]
        and report["final_val_diff"] <= THRESHOLDS["final_val_diff"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True, help="自研 LoRA 的输出目录")
    parser.add_argument("--theirs", type=Path, required=True, help="PEFT 的输出目录")
    parser.add_argument("--reference", type=Path, default=None, help="自研换种子的那次")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        ours_rows, ours_info = read_run(args.ours)
        theirs_rows, theirs_info = read_run(args.theirs)
        check_same_setup(args.ours, ours_info, theirs_info)
        reference = read_run(args.reference)[0] if args.reference else None
        report = compare(ours_rows, theirs_rows, reference)
    except ValueError as error:
        sys.exit(str(error))
    report |= {
        "ours": str(args.ours),
        "theirs": str(args.theirs),
        "reference": str(args.reference) if args.reference else None,
        "ours_commit": ours_info.get("git_commit"),
        "theirs_commit": theirs_info.get("git_commit"),
        "peft_version": theirs_info.get("peft_version"),
        "transformers_version": theirs_info.get("transformers_version"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"第 1 步差 {report['first_step_train_diff']:.2e}；每步 train loss 差：平均 "
        f"{report['mean_abs_train_diff']:.4f}，最大 {report['max_abs_train_diff']:.4f}；"
        f"最后的 val loss 差 {report['final_val_diff']:.4f}；"
        f"{'达标' if report['passed'] else '未达标'}"
    )
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
