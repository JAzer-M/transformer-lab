"""M4 验收：自研模型载入 Qwen3 权重，和 transformers 逐层对比，再比较贪心生成。

transformers 只在这个脚本里当对照，src/tlab 不依赖它（AGENTS 硬性规则 1）。
超过门槛时退出码不是 0。

用法（第一次会下载约 1.5 GB，放在 HuggingFace 默认缓存目录）：
    uv run python scripts/check_qwen3.py --out experiments/m4_qwen3/alignment.json
别的模型用 --repo，比如 --repo Qwen/Qwen3-0.6B-Base。
--autocast bf16 时两边都在 bf16 autocast 下算（GPU 上训练就是这样），只报告差多少，不判定。

环境里设了 SOCKS 代理（ALL_PROXY=socks5://...）时，下载会报缺 socksio。如果同时有 HTTP
代理，在命令前加 env -u ALL_PROXY -u all_proxy，这条命令就只走 HTTP 代理。
"""

import argparse
import contextlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import snapshot_download
from torch import nn
from transformers import Qwen3ForCausalLM

from tlab.convert.qwen3 import MODEL_FILES, load_qwen3
from tlab.eval.generate import generate
from tlab.model.gpt import count_parameters
from tlab.model.layers import RMSNorm
from tlab.tokenizer.hf_bpe import HFTokenizer
from tlab.train.utils import git_commit

REPO = "Qwen/Qwen3-0.6B"
# 长度不同，英文、中文都有；只比较两边的输出，内容本身不重要
PROMPTS = [
    "Give me a short introduction to large language models.",
    "请用一句话解释什么是注意力机制。",
    "Once upon a time, in a small town by the sea, there lived an old fisherman "
    "who went out every morning before the sun came up. One day he found",
]


def record_outputs(
    modules: list[nn.Module], run: Callable[[], torch.Tensor]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """运行模型，记下每个模块的输出；返回这些输出和最后的分数。"""
    outputs: list[torch.Tensor] = []

    def hook(_: nn.Module, __: Any, output: Any) -> None:
        # HF 的层有时返回元组，第一个才是隐藏状态
        outputs.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        logits = run()  # (B, T, V)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, logits


@torch.no_grad()
def compare_layers(
    ours: nn.Module, theirs: nn.Module, ids: torch.Tensor
) -> dict[str, Any]:
    """同样的输入 ids (B, T)，逐层比较：词表、每一层、最终归一化，再比较分数。"""
    our_outputs, our_logits = record_outputs(
        [ours.tok_emb, *ours.blocks, ours.ln_f], lambda: ours(ids)
    )
    their_outputs, their_logits = record_outputs(
        [theirs.model.embed_tokens, *theirs.model.layers, theirs.model.norm],
        lambda: theirs(ids, use_cache=False).logits,
    )
    assert len(our_outputs) == len(their_outputs)
    errors = [
        (a - b).abs().max().item()  # a、b 都是 (B, T, C)
        for a, b in zip(our_outputs, their_outputs, strict=True)
    ]
    sizes = [output.abs().max().item() for output in their_outputs]
    return {
        "layer_max_errors": errors,
        # 隐藏状态的数值可能上千，绝对误差说明不了什么，要除以它本身的大小
        "layer_max_abs": sizes,
        "layer_relative_errors": [
            error / size for error, size in zip(errors, sizes, strict=True)
        ],
        "logits_max_error": (our_logits - their_logits).abs().max().item(),  # ()
        "logits_max_abs": their_logits.abs().max().item(),  # ()
        # autocast 生效时两边都是 bfloat16，用来确认这次确实都在低精度下算
        "logits_dtype": str(our_logits.dtype),
        "their_logits_dtype": str(their_logits.dtype),
    }


@torch.no_grad()
def compare_generation(
    ours: nn.Module, theirs: nn.Module, ids: torch.Tensor, new_tokens: int
) -> dict[str, Any]:
    """贪心生成：我们有无缓存各一次，和 transformers 比较。ids 是 (1, T)。"""
    plain = generate(ours, ids, new_tokens, top_k=1)  # (1, T + new_tokens)
    cached = generate(ours, ids, new_tokens, top_k=1, use_cache=True)  # 同上
    # 模型自带的配置是抽样，这里明确改成贪心，也关掉重复惩罚；不让 eos 提前结束，长度才一样
    their_ids = theirs.generate(
        ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        eos_token_id=None,
    )  # (1, T + new_tokens)
    return {
        "new_tokens": new_tokens,
        "same_without_cache": torch.equal(plain, their_ids),
        "same_with_cache": torch.equal(cached, their_ids),
        "tokens": their_ids[0].tolist(),
    }


def prompt_ids(
    tokenizer: HFTokenizer | None, vocab_size: int
) -> list[tuple[str, list[int]]]:
    """有分词器就编码 PROMPTS；测试用的小模型没有分词器，用固定种子的随机 token。"""
    if tokenizer is not None:
        return [(text, tokenizer.encode(text)) for text in PROMPTS]
    generator = torch.Generator().manual_seed(0)
    return [
        (
            f"随机 {length} 个 token",
            torch.randint(0, vocab_size, (length,), generator=generator).tolist(),
        )
        for length in (5, 17)
    ]


def judge(prompts: list[dict[str, Any]], max_error: float) -> tuple[float, bool]:
    """验收标准说的是分数和贪心生成，所以只看这两样；隐藏状态的误差只报告，不算失败。"""
    worst = max(prompt["logits_max_error"] for prompt in prompts)
    same = all(
        prompt["same_with_cache"] and prompt["same_without_cache"] for prompt in prompts
    )
    return worst, worst < max_error and same


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=None, help="不给就下载 --repo"
    )
    parser.add_argument("--repo", default=REPO, help="不给 --model-dir 时下载哪个模型")
    parser.add_argument(
        "--autocast",
        choices=["bf16"],
        default=None,
        help="两边都在 autocast 下算，只报告差多少，不判定是否通过",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-error",
        type=float,
        default=1e-3,
        help="验收标准是 1e-4 量级，即小于 1e-3",
    )
    args = parser.parse_args()

    # 先建好目录：不建的话要等比较完才报错
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 下载到的目录名就是这份权重在 HF 上的版本号
    model_dir = args.model_dir or Path(
        snapshot_download(args.repo, allow_patterns=MODEL_FILES)
    )
    model_name = (
        str(args.model_dir) if args.model_dir else f"{args.repo}@{model_dir.name}"
    )
    device = torch.device(args.device)
    ours = load_qwen3(model_dir, device)
    theirs = (
        Qwen3ForCausalLM.from_pretrained(
            model_dir, attn_implementation="eager", dtype=torch.float32
        )
        .to(device)
        .eval()
    )

    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = HFTokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    prompts = []
    for i, (text, token_ids) in enumerate(
        prompt_ids(tokenizer, ours.tok_emb.num_embeddings)
    ):
        ids = torch.tensor([token_ids], device=device)  # (1, T)
        result = {"prompt": text, "n_tokens": len(token_ids)}
        # autocast 的类型和设备有关，每次新建
        autocast = (
            torch.autocast(device.type, dtype=torch.bfloat16)
            if args.autocast
            else contextlib.nullcontext()
        )
        with autocast:
            result |= compare_layers(ours, theirs, ids)
            result |= compare_generation(ours, theirs, ids, args.new_tokens)
        if tokenizer is not None:
            result["generated_text"] = tokenizer.decode(result["tokens"])
        prompts.append(result)
        print(
            f"提示 {i + 1}（{len(token_ids)} 个 token）：每层最大相对误差 "
            f"{max(result['layer_relative_errors']):.2e}，分数最大误差 "
            f"{result['logits_max_error']:.2e}（分数最大约 {result['logits_max_abs']:.1f}）；"
            f"贪心相同：有缓存 {result['same_with_cache']}，无缓存 {result['same_without_cache']}",
            flush=True,
        )

    worst, passed = judge(prompts, args.max_error)
    # 验收标准说的是 fp32；bf16 下本来就有舍入，这次只看差多少
    verdict: bool | None = None if args.autocast else passed
    report = {
        "passed": verdict,
        "autocast": args.autocast,
        "max_error": worst,
        "max_error_allowed": args.max_error,
        "git_commit": git_commit(),
        "model": model_name,
        "device": str(device),
        "dtype": str(next(ours.parameters()).dtype),
        # 开关对所有 RMSNorm 一起设置；用 any：模型里没有 RMSNorm 时是 False
        "rmsnorm_fp32": any(
            m.compute_fp32 for m in ours.modules() if isinstance(m, RMSNorm)
        ),
        "n_params": count_parameters(ours),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "prompts": prompts,
    }
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if verdict is None:
        print(f"{args.autocast} 下分数最大误差 {worst:.2e}，只报告，不判定")
        return
    print(
        f"分数最大误差 {worst:.2e}，门槛 {args.max_error:.0e}，"
        f"{'通过' if verdict else '未通过'}"
    )
    if not verdict:
        sys.exit(1)


if __name__ == "__main__":
    main()
