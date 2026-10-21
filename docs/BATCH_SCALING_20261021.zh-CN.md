# 模型批处理：延迟、吞吐量与加速比 — 2026-10-21

[English](BATCH_SCALING_20261021.md) | 简体中文

批处理能提高接收函数、面波的 CUDA 吞吐量。格林函数当前通过主机线程逐模型调度，结构 VJP 仍在 CPU，
增加外层 batch 不会自动变成融合的 GPU 伴随计算。本次使用已安装的 **0.6.0.dev4**，
FP64/complex128、物理算法及 1e-10 km/s 根容差均未改变，无需更换编译包。

## 相同批量下，CUDA 相对 CPU 的加速比

每格为 **CPU 整批时间 / CUDA 整批时间**，模型数量相同。
接收函数基线使用 4 个 CPU Torch 线程，CSV 另列单线程结果；面波使用 16 个模型工作线程。
格林函数总 CPU 预算为 16 线程：B1 为 1 模型工作线程×16 原生线程，B16/B64 为 4×4。

| 任务及 CPU 基线 | B=1 | B=16 | B=64 | B=256 |
| --- | --- | --- | --- | --- |
| 接收函数（CPU4） 正演 | 11.69× | 41.01× | 86.75× | 81.75× |
| 接收函数（CPU4） 正演＋VJP | 7.57× | 24.29× | 13.87× | 27.25× |
| 相速度（CPU16） 正演 | 0.28× | 0.35× | 0.84× | 1.50× |
| 相速度（CPU16）＋全部层 Vs 核 | 1.23× | 1.72× | 3.11× | 4.40× |
| 群速度（CPU16） 正演 | 0.19× | 0.42× | 0.86× | 1.55× |
| 群速度（CPU16）＋全部层 Vs 核 | 2.87× | 3.66× | 5.88× | 6.67× |
| 格林函数（CPU16）† 正演 | 1.14× | 1.41× | 1.93× | — |
| 格林函数（CPU16）† 正演＋VJP | 1.02× | 1.04× | — | — |

小于 1 表示 CUDA 更慢。† 格林函数 CUDA 项包含 CPU 准备及主机输入输出，
双方结构 VJP 都是 CPU 计算。未测 B64 的格林函数 VJP、B256 的格林函数，破折号不代表估算。
此前主表使用了 B1 格林函数与 B16 接收函数/频散，未展示完整批量扩展曲线。

## GPU 自身吞吐量及相对单模型的提升

单位为 **模型/秒**。吞吐提升计算为 `(B/T_B)/(1/T_1) = B*T_1/T_B`，各时间均为实测整批中位数。
这与“GPU 相对 CPU 加速几倍”是两个不同指标。

| 任务 | B=1 | B=16 | B=64 | B=256 | 最大已测批量/B1 吞吐提升 |
| --- | --- | --- | --- | --- | --- |
| 接收函数 正演 | 480.8 | 7157.2 | 17199.8 | 17755.9 | 36.93× (B=256) |
| 接收函数 正演＋VJP | 85.5 | 1353.5 | 981.8 | 1857.8 | 21.73× (B=256) |
| 相速度 正演 | 83.6 | 1225.5 | 3326.3 | 6475.7 | 77.43× (B=256) |
| 相速度＋全部层 Vs 核 | 51.3 | 700.0 | 1482.5 | 2130.8 | 41.50× (B=256) |
| 群速度 正演 | 73.4 | 1090.8 | 3035.1 | 6178.5 | 84.21× (B=256) |
| 群速度＋全部层 Vs 核 | 19.4 | 241.4 | 429.3 | 539.6 | 27.87× (B=256) |
| 格林函数† 正演 | 11.5 | 27.1 | 38.5 | — | 3.34× (B=64) |
| 格林函数† 正演＋VJP | 1.4 | 2.2 | — | — | 1.64× (B=16) |

格林函数的吞吐收益也来自 CPU 预算重新分配及主机并发，不能称为原生 GPU 模型融合。
本轮 RF 的 B64 梯度每模型吞吐低于 B16：batch 增大不保证单调提速，正演和反向各有饱和点与内存开销。

## 为什么有些项目仍加速不大

接收函数、频散 CUDA 核有模型维度，更多独立模型能摊薄接口校验、启动、同步开销，增加 GPU 并行工作。
纯频散正演计算量较小；CPU 基阶寻根可利用前一周期减少搜索，而 CUDA 当前按周期并行试算更多网格点。
完整敏感核有更多参数任务，通常更容易体现 GPU 收益。
格林函数 CUDA 入口尚未融合模型维，每个模型仍有独立分配和传输，结构微分也在 CPU；外层批处理无法消除这一边界。

上一轮 GPU 有约 80% 的其他工作负载；本轮开始采样利用率为 0%，原始 JSON 记录各配置前后的 GPU 状态，
其中采样可能包含本基准自身，机器仍是共享的。与昨天绝对耗时的差别不能全部归因于批处理，
上面所有批量比值只使用本轮数据。

## 测量口径与数值核验

Threadripper PRO 3955WX、RTX 5090 32 GB（`cuda:1`），Python 3.13.5、Torch 2.8.0+cu128。
均为 32 层，固定模型集合取前 B 个组成 1/16/64/256 批量；每项预热一次、计时三次，轮换后端与操作顺序，
取 GPU 同步后的主机墙钟中位数，包含公开接口校验。接收函数/频散复用设备模型和已准备搜索网格，
不计初始构造与搬运；格林函数现有接口的主机读写和传输计入。显存峰值为 Torch 计数，不含原生格林函数 CUDA malloc。
没有停止其他用户作业。

接收函数为 1 射线、6 高斯频带、nfft=2048、81 时间点、41 质点运动窗口；固定 RF/视 Vs 余切对应的
顶深/Vp/Vs/密度 VJP，首层顶深固定。频散为 Rayleigh 基阶、2–50 秒 12 周期，全部层 Vs 核含寻根。
格林函数为 4 接收点、64 采样、固定 20205 波数节点，Vp/Vs VJP 包含正演；不声称测了完整波形稠密 Jacobian。

34 条后端/批量记录均通过 CPU/CUDA 数值对照：

| 类别 | 正演最大相对差 | 微分最大相对差 |
| --- | --- | --- |
| receiver_function  | 1.159e-15 | 2.634e-15 |
| dispersion phase | 5.194e-13 | 3.168e-11 |
| dispersion group | 1.808e-11 | 2.740e-09 |
| green  | 2.442e-14 | 0.000e+00 |

另外核验 RF 原生 batch16 与逐模型调用、频散 batch256 的线程寻根与单模型 warp 寻根，
以及格林函数 CPU 并发伴随与串行伴随；选择代表模型核验：

| 类别 | 指标 | 最大值 |
| --- | --- | --- |
| receiver_function | forward_relative_error | 0.000e+00 |
| receiver_function | vjp_relative_error | 1.346e-15 |
| dispersion | forward_relative_error | 2.239e-11 |
| dispersion | kernel_relative_error | 2.936e-09 |
| green | vjp_relative_error | 4.309e-19 |

这些是实测检查，不是任意模型的严格误差界。70 位精度审计、原始 disba 对照仍见
[dev4 完整报告](PERFORMANCE_0.6.0.dev4.zh-CN.md)，其独立测量批次不混入本次批量比值。

## 复现与原始数据

```sh
python benchmarks/forward-derivatives/benchmark_batch_scaling.py --device cuda:1 --layers 32 --batches 1 16 64 256 --repeat 3 --output batch.json
python benchmarks/forward-derivatives/validate_batch_scaling.py --device cuda:1 --output batch-validation.json
```

安装匹配的 dev4 wheel、Torch 和 NumPy，不设置指向源码的 `PYTHONPATH`。
脚本需要同目录的 `benchmark_forward_derivatives.py` 辅助函数，均通过公开 API 调用。

- [batch-scaling-20261021.json](../verification/batch-scaling-20261021/batch-scaling-20261021.json)
- [summary.csv](../verification/batch-scaling-20261021/summary.csv)
- [batch-validation-20261021.json](../verification/batch-scaling-20261021/batch-validation-20261021.json)
- [batch-smoke-20261021.json](../verification/batch-scaling-20261021/batch-smoke-20261021.json)
- [documentation-examples-20261021.json](../verification/batch-scaling-20261021/documentation-examples-20261021.json)
