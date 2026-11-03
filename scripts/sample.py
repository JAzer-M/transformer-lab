"""从 checkpoint 载入模型，给一个开头，生成文本。

用法：
    uv run python scripts/sample.py --ckpt runs/shakespeare_char/ckpt.pt
    uv run python scripts/sample.py --ckpt ... --prompt "ROMEO:" --temperature 0.8
    uv run python scripts/sample.py --ckpt runs/tinystories_smoke/ckpt.pt \
        --prompt "Once upon a time"
"""

import argparse
from pathlib import Path

import torch

from tlab.data.loader import load_tokenizer
from tlab.device import get_device
from tlab.eval.generate import generate
from tlab.train.utils import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", default="\n", help="开头文字，默认一个换行")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)

    # 按 checkpoint 里保存的配置重建模型，再载入参数
    model, ckpt = load_model(args.ckpt, device)
    config = ckpt["config"]

    # 词表来自训练时的数据目录：字符级是 meta.json，BPE 是 tokenizer.json
    tokenizer = load_tokenizer(Path(config["data_dir"]))
    prompt = torch.tensor([tokenizer.encode(args.prompt)], device=device)  # (1, T)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    out = generate(
        model,
        prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        generator=generator,
    )  # (1, T + max_new_tokens)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
