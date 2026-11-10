# 噪声互相关 CPU/CUDA 性能（0.6.0.dev16）

[English](PERFORMANCE_0.6.0.dev16.md) · [接口](NOISE_CORRELATION.zh-CN.md) · [逐次计时](../verification/preview-0.6.0.dev16/performance.json) · [公开 API 测速脚本](../benchmarks/data-processing/benchmark_noise_correlation.py)

2026-11-10 在实际安装的 Linux CPython 3.13 编译包上测量。CPU 为 AMD Ryzen
Threadripper PRO 3955WX（16 核/32 线程），GPU 为 RTX 5090；PyTorch 2.8.0+cu128、
CUDA 12.8、NumPy 2.1.3。服务器为共享环境，未停止其他任务。
CPU 基线取 NumPy、Torch 单线程和 8 线程实测中的最快者，不代表最优的全部 CPU 实现或线程数。

端到端流程包括去均值、两端各 5% 余弦渐消、平滑谱白化（11 bins，0.2–10 Hz，
外侧 0.1 Hz 过渡带）、系数归一化、全部 i<j 的线性互相关和加权均值叠加。
采样间隔 0.02 s、50% 重叠、输出 ±5 s，按 8 窗口/256 台站对分块；输入已校验
（`check_finite=False`）。本表是线性叠加正向处理，未测 PWS 或 backward 耗时。
排除预热（CPU 2 次；CUDA 至少 8 次且不少于 0.25 s），取 3 次同步墙钟计时中位数，
包含 Python 调度和设备内存分配，不含磁盘 I/O。逐次时间全部保留。

| 规模 | 精度 | 最快实测 CPU ms | 驻留 CUDA ms | CUDA 含传输 ms | 驻留加速比 | 含传输加速比 |
|---|---|---:|---:|---:|---:|---:|
| small | float32 | 1.000 | 2.642 | 2.699 | 0.38× | 0.37× |
| small | float64 | 1.084 | 2.644 | 2.738 | 0.41× | 0.40× |
| array | float32 | 265.478 | 9.893 | 10.956 | 26.83× | 24.23× |
| array | float64 | 418.939 | 12.278 | 14.191 | 34.12× | 29.52× |
| batch4 | float32 | 916.556 | 18.154 | 21.594 | 50.49× | 42.44× |
| batch4 | float64 | 1737.672 | 37.670 | 43.268 | 46.13× | 40.16× |
| dense128 | float32 | 1935.915 | 33.407 | 34.901 | 57.95× | 55.47× |
| dense128 | float64 | 3556.845 | 67.575 | 72.328 | 52.64× | 49.18× |

| Case | B | Stations | Record samples | Window samples | Windows | Pairs per batch |
|---|---:|---:|---:|---:|---:|---:|
| small | 1 | 4 | 2048 | 512 | 7 | 6 |
| array | 1 | 32 | 65536 | 4096 | 31 | 496 |
| batch4 | 4 | 32 | 65536 | 4096 | 31 | 496 |
| dense128 | 1 | 128 | 32768 | 4096 | 15 | 8128 |


“含传输”包括波形上传和 CCF 下载，台站对/时延元数据仍留在 GPU。
小任务的 CUDA 启动和调度成本占主导，CPU 更快。较大台阵/batch 利用台站 FFT 复用、
GPU 台站对乘积，以及反 FFT 前的互谱叠加取得收益。加速比仅针对本实现所测 CPU 路径，
不宣称相对 FastXC/NoisePy 的性能。

CUDA 相对 NumPy 的最大相对 L2 误差：**FP32 4.13e-07，FP64 1.24e-15**。
这对应表内的系数归一化 CCF，不是任意输入的误差保证。峰值 CUDA 已分配显存约从
1 MiB（小任务 FP32）到 2.84 GiB（B=4 FP64），原始 JSON 列出各项。
线性叠加需保留全部输出台站对的频谱累积量及分块临时量；特别大的台阵应指定所需台站对。

安装包回归每平台 236 项：macOS 190 通过/46 跳过，Linux/CUDA 214 通过/22 跳过。
新增互相关测试 CUDA 15/15 通过，Mac 跳过其中 CUDA 专属项；见[验证记录](../verification/preview-0.6.0.dev16/verification.json)。
GitHub Actions 本轮被取消，上述结果来自两台实际机器的编译包测试。

与 `source_field.receiver_cross_spectrum` 理论互谱比较时，其定义为 `E[Y_i conj(Y_j)]`，
而本 CCF 使用 `conj(Y_i)*Y_j`，需转置/共轭台站索引，并对齐单边 PSD 与 FFT 的归一化。
非线性白化及有限窗处理也会改变所比较的统计量。
