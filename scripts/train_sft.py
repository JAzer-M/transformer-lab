"""LoRA SFT 的入口：在 GSM8K 训练集上微调 Qwen3-0.6B-Base，只训练 A、B。

用法（第一次会下载基座权重，约 1.2 GB；数据先用 prepare_gsm8k.py 准备）：
    uv run python scripts/train_sft.py --config configs/finetune/sft_r8.yaml
短测显存和速度：在命令后面加 max_steps=5；换种子：seed=2024。
输出在 out_dir：metrics.csv、run.json、lora.pt（只有 A、B）。

环境里设了 SOCKS 代理（ALL_PROXY=socks5://...）时，下载会报缺 socksio。如果同时有 HTTP
代理，在命令前加 env -u ALL_PROXY -u all_proxy，这条命令就只走 HTTP 代理。
"""

import argparse
from pathlib import Path

from tlab.finetune.sft import load_sft_config, run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--model-dir", type=Path, default=None, help="不给就下载")
    parser.add_argument("overrides", nargs="*", help="覆盖单个字段，如 max_steps=5")
    args = parser.parse_args()

    info = run_sft(load_sft_config(args.config, args.overrides), args.model_dir)
    print(
        f"完成 {info['steps_run']}/{info['total_steps']} 步：train {info['final_train_loss']:.4f}，"
        f"val {info['final_val_loss']:.4f}，{info['tokens_per_second']:.0f} tokens/s，"
        f"显存峰值 {info['peak_memory_mb']} MiB，用时 {info['seconds']:.0f} 秒"
    )


if __name__ == "__main__":
    main()
