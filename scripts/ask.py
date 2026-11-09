"""自己提问：把 Qwen3 载入我们写的 GPT（可选装上 LoRA），贪心回答一道题。

提示格式和 SFT 训练、评测时一样：Question: {你的问题}\\nAnswer:
抽答案的规则也和评测一样：先截掉模型接着编的下一题，再取 #### 后面的数。

用法：
    # 一道题
    uv run python scripts/ask.py --lora runs/m5/f2_r8_seed1337/lora.pt "Tom has ..."
    # 不给题就一题接一题地问（模型只载一次），空行退出
    uv run python scripts/ask.py --lora runs/m5/f2_r8_seed1337/lora.pt
    # 去掉 --lora 就是原始的基座，可以对比 SFT 改变了什么

不给 --model-dir 时用 HuggingFace 缓存里的基座（没有就下载，约 1.2 GB）；
LoRA 路径写错时，在找基座之前就会报错。
"""

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tlab.convert.qwen3 import resolve_model_dir
from tlab.device import get_device
from tlab.eval.gsm8k import generate_answer, truncate_continuation, zero_shot_prompt
from tlab.finetune.lora_model import check_lora_file, load_qwen3_with_lora
from tlab.tokenizer.hf_bpe import HFTokenizer

REPO = "Qwen/Qwen3-0.6B-Base"  # 和评测、SFT 配置里的默认基座相同
EOS = 151643  # Qwen3 的 <|endoftext|>，SFT 时答案后面接的就是它


def questions_from_input() -> Iterator[str]:
    """一行一题；空行、输入结束（Ctrl-D）或 Ctrl-C 就停。"""
    while True:
        try:
            line = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        yield line


def describe_reply(reply: dict[str, Any], seconds: float) -> str:
    """把 generate_answer 的结果写成几行给人看的文字。"""
    output = reply["output"]
    shown = truncate_continuation(output)
    lines = [f"回答：{shown}"]
    if len(shown) < len(output):
        # 基座答完常接着编下一题；抽答案时这部分也截掉了
        lines.append(f"（后面又编了 {len(output) - len(shown)} 个字符的新题，没显示）")
    lines.append(f"抽出的答案：{reply['pred'] or '没有找到数'}")
    stop = "在 eos 停下" if reply["stopped_by_eos"] else "到了长度上限还没停"
    lines.append(f"{stop}；生成 {reply['new_tokens']} 个 token，用时 {seconds:.1f} 秒")
    return "\n".join(lines)


def ask_one(
    model: nn.Module,
    tokenizer: HFTokenizer,
    question: str,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
) -> None:
    # 和 SFT 训练、评测时同一个格式；用 repr 显示，换行写成 \n，看得出格式的每个字符
    prompt = zero_shot_prompt(question)
    print(f"提示：{prompt!r}")
    start = time.perf_counter()
    reply = generate_answer(
        model, tokenizer, prompt, max_new_tokens, eos_token_id, device
    )
    print(describe_reply(reply, time.perf_counter() - start) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question", nargs="?", help="不给就从输入里一题一题地读")
    parser.add_argument(
        "--lora", type=Path, default=None, help="LoRA 存档；不给就是基座"
    )
    parser.add_argument("--model-dir", type=Path, default=None, help="不给就用 HF 缓存")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eos-token-id", type=int, default=EOS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    check_lora_file(args.lora)  # 路径写错时不用等找基座、载模型才知道
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model_dir, model_name = resolve_model_dir(REPO, args.model_dir)
    model, lora = load_qwen3_with_lora(model_dir, device, args.lora)
    tokenizer = HFTokenizer.load(model_dir / "tokenizer.json")
    lora_note = f"LoRA {lora['path']}（r={lora['r']}）" if lora else "没有 LoRA"
    print(f"模型 {model_name}，{lora_note}，设备 {device}\n")

    questions = [args.question] if args.question else questions_from_input()
    for question in questions:
        ask_one(
            model, tokenizer, question, args.max_new_tokens, args.eos_token_id, device
        )


if __name__ == "__main__":
    main()
