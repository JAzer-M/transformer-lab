"""给每个变体的存档生成文字样例，写成一个 Markdown 文件。

要在存放存档的机器上运行（Windows + RTX 5060 的 WSL），数据目录按存档里的配置读取。

用法：
    uv run python scripts/sample_ablations.py --out experiments/m3_ablations/samples.md
"""

import argparse
from pathlib import Path

import torch

from tlab.device import get_device
from tlab.eval.samples import VARIANTS, SampleParams, render_samples, sample_checkpoint
from tlab.train.utils import git_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/m3"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--run-seed", type=int, default=1337, help="用哪个种子的存档")
    # 下面的默认值与 M2 的样例相同
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # 先确认存档都在，免得跑到一半才发现缺一个
    paths = {
        name: args.runs / name / f"seed{args.run_seed}" / "ckpt.pt"
        for name in args.variants
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} 的存档不存在：{path}")

    device = get_device() if args.device == "auto" else torch.device(args.device)
    params = SampleParams(
        args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seeds
    )
    results = {}
    for name, path in paths.items():
        results[name] = sample_checkpoint(path, params, device)
        print(f"{name}：第 {results[name]['ckpt_step']} 步的存档", flush=True)
    text = render_samples(results, params, git_commit(), str(device), args.run_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
