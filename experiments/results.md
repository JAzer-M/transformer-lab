| 日期 | 编号 | 配置文件 | commit | 种子 | val loss / 准确率 | tokens/s | 显存峰值 | 一句结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-30 | M1-1 | configs/shakespeare_char.yaml | 54549658efcc8562b56d2505a1f49034eae79a47-dirty | 1337 | val 1.5858 | 105912 | 2597 MB | val 持续下降；样例有剧本格式，词形与语义仍有错误 |
| 2026-11-02 | M2-1 | configs/tinystories_base.yaml | 45e78f1e718f70b322d0e817e47eb8e7272d5773 | 1337 | val 1.3531 | 82756 | 5007 MB | val 持续下降；可生成故事，部分因果矛盾与截断 |
| 2026-11-02 | M3-a0_control-s1337 | configs/ablations/a0_control.yaml（seed=1337） | b7d9989a4642d8a5d702321755fef18399ddd055 | 1337 | val 1.5864 | 82179 | 5007 MB | 对照组；固定 100,007,936 token；未见发散 |
| 2026-11-03 | M3-a0_control-s2024 | configs/ablations/a0_control.yaml（seed=2024） | b7d9989a4642d8a5d702321755fef18399ddd055 | 2024 | val 1.5689 | 80068 | 5007 MB | 对照组；固定 100,007,936 token；未见发散 |
| 2026-11-03 | M3-a2_postln-s1337 | configs/ablations/a2_postln.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.6207（完整验证集 1.6124） | 79285 | 4982 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_postln-s2024 | configs/ablations/a2_postln.yaml（seed=2024） | c72d450997f3e830fe36c895ed6e1bd55448213d | 2024 | val 1.6019（完整验证集 1.6150） | 78602 | 4982 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_rmsnorm-s1337 | configs/ablations/a2_rmsnorm.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.5915（完整验证集 1.5829） | 72475 | 5317 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_rmsnorm-s2024 | configs/ablations/a2_rmsnorm.yaml（seed=2024） | c72d450997f3e830fe36c895ed6e1bd55448213d | 2024 | val 1.5690（完整验证集 1.5814） | 72382 | 5317 MB | A2归一化对比；未见发散 |
| 2026-11-03 | M3-a2_preln_lr3e-3-s1337 | configs/ablations/a2_preln_lr3e-3.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.4732（完整验证集 1.4651） | 78303 | 5007 MB | 高学习率加测；未发散 |
| 2026-11-03 | M3-a2_postln_lr3e-3-s1337 | configs/ablations/a2_postln_lr3e-3.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 5.8359（完整验证集 5.8343） | 79707 | 4982 MB | 高学习率加测；未发散；验证loss停留在约5.84 |
| 2026-11-03 | M3-a2_rmsnorm_lr3e-3-s1337 | configs/ablations/a2_rmsnorm_lr3e-3.yaml（seed=1337） | c72d450997f3e830fe36c895ed6e1bd55448213d | 1337 | val 1.4770（完整验证集 1.4694） | 73628 | 5317 MB | 高学习率加测；未发散 |
| 2026-11-04 | M3-a2_preln_lr3e-3-s2024 | configs/ablations/a2_preln_lr3e-3.yaml（seed=2024） | b9491f2661ab0c904eb8de73c490f21eef55b799 | 2024 | val 1.4567（完整验证集 1.4653） | 83828 | 5007 MB | 高学习率第2种子；未发散；首次抽样val≤2.0在第750步 |
| 2026-11-04 | M3-a2_postln_lr3e-3-s2024 | configs/ablations/a2_postln_lr3e-3.yaml（seed=2024） | b9491f2661ab0c904eb8de73c490f21eef55b799 | 2024 | val 5.8236（完整验证集 5.8352） | 84395 | 4982 MB | 高学习率第2种子；未发散；全程记录的抽样val未达到2.0，未收敛 |
| 2026-11-04 | M3-a2_rmsnorm_lr3e-3-s2024 | configs/ablations/a2_rmsnorm_lr3e-3.yaml（seed=2024） | b9491f2661ab0c904eb8de73c490f21eef55b799 | 2024 | val 1.4624（完整验证集 1.4717） | 76311 | 5317 MB | 高学习率第2种子；未发散；首次抽样val≤2.0在第750步 |
| 2026-11-05 | M3-a3_swiglu-s1337 | configs/ablations/a3_swiglu.yaml（seed=1337） | 6629ec8847f6446aa142979bbccd5e4e63a63563 | 1337 | val 1.5547（完整验证集 1.5469） | 78751 | 5270 MB | A3前馈对比；mlp=swiglu；未见发散 |
| 2026-11-05 | M3-a3_swiglu-s2024 | configs/ablations/a3_swiglu.yaml（seed=2024） | 6629ec8847f6446aa142979bbccd5e4e63a63563 | 2024 | val 1.5328（完整验证集 1.5451） | 79158 | 5270 MB | A3前馈对比；mlp=swiglu；未见发散 |
| 2026-11-05 | M3-a1_sinusoidal-s1337 | configs/ablations/a1_sinusoidal.yaml（seed=1337） | 6629ec8847f6446aa142979bbccd5e4e63a63563 | 1337 | val 1.6047（完整验证集 1.5960） | 79529 | 5004 MB | A1位置编码；pos_encoding=sinusoidal；1024外推已完成；未见发散 |
| 2026-11-05 | M3-a1_sinusoidal-s2024 | configs/ablations/a1_sinusoidal.yaml（seed=2024） | 6629ec8847f6446aa142979bbccd5e4e63a63563 | 2024 | val 1.5799（完整验证集 1.5911） | 79122 | 5004 MB | A1位置编码；pos_encoding=sinusoidal；1024外推已完成；未见发散 |
| 2026-11-05 | M3-a1_rope-s1337 | configs/ablations/a1_rope.yaml（seed=1337） | 6629ec8847f6446aa142979bbccd5e4e63a63563 | 1337 | val 1.5333（完整验证集 1.5281） | 73108 | 5007 MB | A1位置编码；pos_encoding=rope；1024外推已完成；未见发散 |
| 2026-11-05 | M3-a1_rope-s2024 | configs/ablations/a1_rope.yaml（seed=2024） | 6629ec8847f6446aa142979bbccd5e4e63a63563 | 2024 | val 1.5132（完整验证集 1.5272） | 73330 | 5007 MB | A1位置编码；pos_encoding=rope；1024外推已完成；未见发散 |
| 2026-11-06 | M3-a4_gqa2-s1337 | configs/ablations/a4_gqa2.yaml（seed=1337） | ca964f766726708765e244e3829365e946fe5f9b | 1337 | val 1.6007（完整验证集 1.5943） | 83812 | 4986 MB | A4注意力；n_kv_heads=2；未见发散 |
| 2026-11-06 | M3-a4_gqa2-s2024 | configs/ablations/a4_gqa2.yaml（seed=2024） | ca964f766726708765e244e3829365e946fe5f9b | 2024 | val 1.5751（完整验证集 1.5883） | 85395 | 4986 MB | A4注意力；n_kv_heads=2；未见发散 |
| 2026-11-06 | M3-a4_mqa-s1337 | configs/ablations/a4_mqa.yaml（seed=1337） | ca964f766726708765e244e3829365e946fe5f9b | 1337 | val 1.5969（完整验证集 1.5902） | 85782 | 4982 MB | A4注意力；n_kv_heads=1；未见发散 |
| 2026-11-06 | M3-a4_mqa-s2024 | configs/ablations/a4_mqa.yaml（seed=2024） | ca964f766726708765e244e3829365e946fe5f9b | 2024 | val 1.5805（完整验证集 1.5943） | 85563 | 4982 MB | A4注意力；n_kv_heads=1；未见发散 |
| 2026-11-06 | M3-a9_modern-s1337 | configs/ablations/a9_modern.yaml（seed=1337） | ca964f766726708765e244e3829365e946fe5f9b | 1337 | val 1.5082（完整验证集 1.5001） | 70359 | 5561 MB | 组合仅作参考，不作单变量归因；未见发散 |
| 2026-11-06 | M3-a9_modern-s2024 | configs/ablations/a9_modern.yaml（seed=2024） | ca964f766726708765e244e3829365e946fe5f9b | 2024 | val 1.4922（完整验证集 1.5044） | 70504 | 5561 MB | 组合仅作参考，不作单变量归因；未见发散 |
| 2026-11-06 | M3-bench | scripts/bench_attention.py（9 个配置） | ca964f766726708765e244e3829365e946fe5f9b | 1337 | — | 330 / 223 | 102.62 MB | 详见 experiments/m3_ablations/bench.md |
| 2026-11-06 | M3-bench-batch | scripts/bench_attention.py（a0_control、a4_gqa2；B=1、8、32；CPU B=1） | 4e6bb179f182d9c65d4ba7e828c84810117f7b5e | 1337 | — | 1156.58 / 7866.12 | 267.01 MiB | 详见 experiments/m3_ablations/bench_batch/ |
| 2026-11-07 | M4-align | Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca（scripts/check_qwen3.py） | 6917765b8abf7156a76ed6842f65a0e78f3ce560 | — | 分数最大误差 1.73e-04（fp32，CPU） | — | —（CPU，未测显存） | 自研模型载入官方权重，分数误差在 1e-4 量级，贪心生成和 transformers 相同 |
| 2026-11-07 | F1-base-fewshot | configs/finetune/f1_base_fewshot.yaml | d47ab580911e647f7a6a126e12b898e57742a683 | — | 0.468（234/500） | 24.082784926455183 | 3788.81787109375 MiB | 基座 few-shot |
| 2026-11-07 | F1-base-zeroshot | configs/finetune/f1_base_zeroshot.yaml | d47ab580911e647f7a6a126e12b898e57742a683 | — | 0.498（249/500） | 29.832448260595154 | 3498.3583984375 MiB | 基座不给例子 |
| 2026-11-07 | M5-align-bf16 | Qwen/Qwen3-0.6B-Base（scripts/check_qwen3.py --autocast bf16） | d47ab580911e647f7a6a126e12b898e57742a683 | — | 分数最大误差 0.65625（bf16，GPU） | — | — | 只报告，不判定 |
| 2026-11-07 | M5-align-cpu | Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd（scripts/check_qwen3.py --repo） | d47ab580911e647f7a6a126e12b898e57742a683 | — | 分数最大误差 1.11e-04（fp32，CPU） | — | —（CPU，未测显存） | 基座也能载入自研架构，贪心生成和 transformers 相同 |
| 2026-11-07 | M5-align-cuda-fp32 | Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd（scripts/check_qwen3.py --repo --device cuda） | d47ab580911e647f7a6a126e12b898e57742a683 | — | 分数最大误差 0.0（fp32，GPU） | — | — | GPU 上每层逐位相同（由 Codex 运行，Claude 从 json 核对） |
| 2026-11-08 | M5-align-bf16-rmsnorm32 | Qwen/Qwen3-0.6B-Base（scripts/check_qwen3.py --autocast bf16，rmsnorm_fp32 打开） | 5db2d8f25df615540d6f46ae3fc49847e1000bd4 | — | 分数最大误差 0.0（bf16，GPU） | — | — | 只报告，不判定 |
