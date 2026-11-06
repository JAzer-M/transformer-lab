# M3第5批补充：生成batch与CPU短测

Codex在WSL2 Ubuntu执行；来源main `4e6bb179f182d9c65d4ba7e828c84810117f7b5e`。4条命令、7个设备/配置结果、99次测量均已完成，退出码均为0，总耗时879.062秒（含每配置60秒降温）。执行者已完成99次测量及文件核对；独立只读复核因会话中断未完成，用户已明确要求直接提交；Claude本批审查待进行。

设备：RTX5060 Laptop 8151MiB，PyTorch2.9.1+cu128/CUDA12.8；同机CPU为Intel Core i7-14650HX（16核24线程），自检时PyTorch默认12线程。uv sync --locked成功；CUDA分配通过；367 passed in20.85s。

## 生成结果

表中为原始测量中位数；GPU每项5次，CPU每项3次。tokens/s为batch内全部新生成token的总吞吐，每条序列prompt32+新增480。CPU显存峰值未测，以—表示。

| 设备 | 变体 | B | 无缓存tokens/s | 有缓存tokens/s | 有缓存÷无缓存 | 无缓存峰值MiB | 有缓存峰值MiB | 完整缓存MiB（B=1） | k / v |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cuda | a0_control | 1 | 321.85 | 320.48 | 0.996 | 115.20 | 102.62 | 4.50 | torch.bfloat16 / torch.bfloat16 |
| cuda | a4_gqa2 | 1 | 306.92 | 278.31 | 0.907 | 109.21 | 93.50 | 1.50 | torch.bfloat16 / torch.bfloat16 |
| cuda | a0_control | 8 | 1261.48 | 2333.96 | 1.850 | 238.57 | 139.97 | 4.50 | torch.bfloat16 / torch.bfloat16 |
| cuda | a4_gqa2 | 8 | 1288.77 | 2334.85 | 1.812 | 232.45 | 109.80 | 1.50 | torch.bfloat16 / torch.bfloat16 |
| cuda | a0_control | 32 | 1156.58 | 7866.12 | 6.801 | 662.30 | 267.01 | 4.50 | torch.bfloat16 / torch.bfloat16 |
| cuda | a4_gqa2 | 32 | 1196.83 | 8348.19 | 6.975 | 657.78 | 166.93 | 1.50 | torch.bfloat16 / torch.bfloat16 |
| cpu | a0_control | 1 | 38.78 | 287.70 | 7.419 | — | — | 9.00 | torch.float32 / torch.float32 |

完整缓存始终按B=1计算，生成峰值取实际生成batch。GPU使用bf16，CPU使用fp32。全部实测值原样保留；本报告不对性能机制下结论，未追加重测。

## 温度与耗时

每次计时前由nvidia-smi读取GPU温度，以下范围不是连续监控极值。CPU项也读取GPU温度，不代表CPU温度。

| 任务 | 变体 | 配置前/后GPU温度°C | 测量起始GPU温度范围°C | 缺失次数 |
| --- | --- | --- | --- | --- |
| b1 | a0_control | 51.5 / 64.4 | 53.5–68.7 | 0 |
| b1 | a4_gqa2 | 54.1 / 65.3 | 54.1–71.6 | 0 |
| b8 | a0_control | 54.5 / 70.0 | 57.1–78.8 | 0 |
| b8 | a4_gqa2 | 56.5 / 71.2 | 56.5–80.4 | 0 |
| b32 | a0_control | 56.7 / 79.8 | 58.7–86.3 | 0 |
| b32 | a4_gqa2 | 58.9 / 80.2 | 58.9–86.3 | 0 |
| cpu_b1 | a0_control | 59.2 / 60.2 | 60.1–62.9 | 0 |

| 输出 | 秒（含降温） | 退出码 |
| --- | --- | --- |
| b1.json | 182.028 | 0 |
| b8.json | 196.031 | 0 |
| b32.json | 300.038 | 0 |
| cpu_b1.json | 200.739 | 0 |

## 实际命令

`/home/j/.local/bin/uv run python scripts/bench_attention.py --configs configs/ablations/a0_control.yaml configs/ablations/a4_gqa2.yaml --generate-batch-size 1 --repeats 5 --train-steps 5 --out experiments/m3_ablations/bench_batch/b1.json`

`/home/j/.local/bin/uv run python scripts/bench_attention.py --configs configs/ablations/a0_control.yaml configs/ablations/a4_gqa2.yaml --generate-batch-size 8 --repeats 5 --train-steps 5 --out experiments/m3_ablations/bench_batch/b8.json`

`/home/j/.local/bin/uv run python scripts/bench_attention.py --configs configs/ablations/a0_control.yaml configs/ablations/a4_gqa2.yaml --generate-batch-size 32 --repeats 5 --train-steps 5 --out experiments/m3_ablations/bench_batch/b32.json`

`/home/j/.local/bin/uv run python scripts/bench_attention.py --configs configs/ablations/a0_control.yaml --device cpu --repeats 3 --train-steps 1 --out experiments/m3_ablations/bench_batch/cpu_b1.json`

uv使用已确认绝对路径；逐条捕获原始控制台输出、检查退出码，每个配置写出JSON后核对缓存。GPU自检和pytest结束后才启动短测，未并行启动其它计算工作。脚本使用随机权重和随机token，没有启动正式训练或读取正式存档/数据。GPU的train-steps5及CPU的train-steps1为脚本自带短测参数，本批不据此评价训练吞吐。

正式命令前，日志启动器出现uv路径错误及一次内联诊断引号错误，两次均未启动GPU自检、pytest或短测。修正后首次实际前置通过；4条短测各执行一次。设置阶段记录见launcher_setup_failures.json。

4份JSON、4份自动MD、4份原始控制台及前置/执行/指纹证据一同归档。99次原始测量的中位数、配置、来源和7个缓存尺寸/类型逐项核对。运行期间360个已跟踪文件未变。results仅追加M3-bench-batch一行；原bench.json、bench.md与SUMMARY保持原文。用户手写文件和配置保持来源提交内容，模型、数据和runs目录不入Git。
