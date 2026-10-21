# 解析伴随与设备内微分实测 — 0.6.0.dev7

[算法、公式与接口](ANALYTIC_METHODS.zh-CN.md) · [设备边界](DEVICE_EXECUTION.zh-CN.md) · [二进制安装](PREVIEW_0.6.0.dev7.zh-CN.md)

本次对比正式编译的 dev6 与 dev7 wheel，分别在 Apple M4 Max CPU、Threadripper PRO 3955WX CPU 和 RTX 5090 上测量。**加速比为同设备 dev6 时间 / dev7 时间**，小于 1 表示变慢。CPU/GPU 比值单独列出，不能与版本优化倍数混用。

## 计时范围

全程 FP64；2 次预热、5 次同步墙钟计时取中位数；较大 Green 案例为 3 次。CPU 原生 4 线程、Torch 1 线程。各版本串行运行，未与本任务的编译或其他计算重叠；不保证共享机器无外部负载。

面波是含寻根的 8 个周期、全部厚度/Vp/Vs/密度核；表格使用 32 层。RF 是 32 层、2 条射线、256 FFT 点的完整正演或损失 VJP。小 Green 是 8 层、2 接收点、32 时间点、16 频率、固定 64 个波数步、全部结构参数 VJP。Green 批量仍逐模型派发；固定网格用于公平算法比较，不能据此证明某个物理场的截止积分已收敛。

## Apple M4 Max CPU

| 计算 | B | dev6 ms | dev7 ms | 加速比 |
|---|---:|---:|---:|---:|
| surface 相速度＋全部参数核 | 1 | 11.135 | 2.820 | 3.95× |
| surface 群速度＋全部参数核 | 1 | 45.149 | 24.252 | 1.86× |
| rf 仅正演 | 1 | 0.457 | 0.471 | 0.97× |
| rf 正演＋VJP | 1 | 1.785 | 1.268 | 1.41× |
| green 仅正演 | 1 | 1.204 | 1.221 | 0.99× |
| green 正演＋VJP | 1 | 14.187 | 14.073 | 1.01× |
| surface 相速度＋全部参数核 | 32 | 305.028 | 43.468 | 7.02× |
| surface 群速度＋全部参数核 | 32 | 1393.913 | 728.848 | 1.91× |
| rf 仅正演 | 32 | 7.846 | 7.978 | 0.98× |
| rf 正演＋VJP | 32 | 28.488 | 27.386 | 1.04× |
| green 仅正演 | 32 | 35.722 | 35.088 | 1.02× |
| green 正演＋VJP | 32 | 451.048 | 462.044 | 0.98× |

## Threadripper PRO 3955WX CPU

| 计算 | B | dev6 ms | dev7 ms | 加速比 |
|---|---:|---:|---:|---:|
| surface 相速度＋全部参数核 | 1 | 23.877 | 9.733 | 2.45× |
| surface 群速度＋全部参数核 | 1 | 105.440 | 60.153 | 1.75× |
| rf 仅正演 | 1 | 1.489 | 1.372 | 1.09× |
| rf 正演＋VJP | 1 | 6.363 | 4.464 | 1.43× |
| green 仅正演 | 1 | 2.636 | 2.880 | 0.92× |
| green 正演＋VJP | 1 | 18.884 | 19.431 | 0.97× |
| surface 相速度＋全部参数核 | 32 | 539.796 | 108.964 | 4.95× |
| surface 群速度＋全部参数核 | 32 | 3257.861 | 1688.588 | 1.93× |
| rf 仅正演 | 32 | 18.142 | 18.993 | 0.96× |
| rf 正演＋VJP | 32 | 60.095 | 58.043 | 1.04× |
| green 仅正演 | 32 | 86.825 | 87.743 | 0.99× |
| green 正演＋VJP | 32 | 641.002 | 634.238 | 1.01× |

## RTX 5090 CUDA

| 计算 | B | dev6 ms | dev7 ms | 加速比 |
|---|---:|---:|---:|---:|
| surface 相速度＋全部参数核 | 1 | 11.256 | 12.365 | 0.91× |
| surface 群速度＋全部参数核 | 1 | 43.899 | 54.015 | 0.81× |
| rf 仅正演 | 1 | 1.944 | 1.861 | 1.04× |
| rf 正演＋VJP | 1 | 12.093 | 5.930 | 2.04× |
| green 仅正演 | 1 | 2.502 | 4.707 | 0.53× |
| green 正演＋VJP | 1 | 118.257 | 48.042 | 2.46× |
| surface 相速度＋全部参数核 | 32 | 17.723 | 14.847 | 1.19× |
| surface 群速度＋全部参数核 | 32 | 69.901 | 56.851 | 1.23× |
| rf 仅正演 | 32 | 1.871 | 1.878 | 1.00× |
| rf 正演＋VJP | 32 | 11.797 | 5.970 | 1.98× |
| green 仅正演 | 32 | 73.828 | 146.912 | 0.50× |
| green 正演＋VJP | 32 | 3765.435 | 1462.136 | 2.58× |

## 较大格林函数网格

8 层、128 时间点、64 频率、512 个固定波数步、2 接收点。

| Device | Operation | B | dev6 ms | dev7 ms | Speedup |
|---|---|---:|---:|---:|---:|
| cpu | 仅正演 | 1 | 75.507 | 68.236 | 1.11× |
| cpu | 正演＋VJP | 1 | 631.076 | 687.425 | 0.92× |
| cuda:1 | 仅正演 | 1 | 20.507 | 36.146 | 0.57× |
| cuda:1 | 正演＋VJP | 1 | 3717.588 | 453.250 | 8.20× |
| cpu | 仅正演 | 4 | 277.501 | 273.830 | 1.01× |
| cpu | 正演＋VJP | 4 | 2559.849 | 2760.621 | 0.93× |
| cuda:1 | 仅正演 | 4 | 79.694 | 146.436 | 0.54× |
| cuda:1 | 正演＋VJP | 4 | 14892.448 | 1810.759 | 8.22× |

## 新版本同机 CPU / CUDA

| Operation | B | CPU / CUDA |
|---|---:|---:|
| surface 相速度＋全部参数核 | 1 | 0.79× |
| surface 群速度＋全部参数核 | 1 | 1.11× |
| rf 仅正演 | 1 | 0.74× |
| rf 正演＋VJP | 1 | 0.75× |
| green 仅正演 | 1 | 0.61× |
| green 正演＋VJP | 1 | 0.40× |
| surface 相速度＋全部参数核 | 32 | 7.34× |
| surface 群速度＋全部参数核 | 32 | 29.70× |
| rf 仅正演 | 32 | 10.11× |
| rf 正演＋VJP | 32 | 9.72× |
| green 仅正演 | 32 | 0.60× |
| green 正演＋VJP | 32 | 0.43× |

## 精度与限制

整体相对差异定义为 `max(abs(new-old))/max(abs(old))`，不是每个接近零分量的相对误差，也不是地质反演准确率。下表覆盖本报告所有模型/设备/批次。

| Family | Maximum normalized difference |
|---|---:|
| surface | 9.562e-11 |
| rf | 5.280e-15 |
| green | 3.028e-12 |

同机 CPU/CUDA 的最大整体相对差异（包括较大网格）：

| Family | dev6 | dev7 |
|---|---:|---:|
| surface | 1.738e-07 | 1.738e-07 |
| rf | 3.738e-15 | 4.953e-15 |
| green | 2.945e-12 | 4.271e-13 |

独立测试还覆盖：高阶频散重寻根差分与旧解析核、RF 独立 Torch 图、Levin 波形 JVP 的密度缩放恒等式和 Gauss 差分、Green 全物理场/非默认 CUDA 流/应力直接项/共享源批量/梯度掩码。较长周期测试使统一低频保护阈值从 3e-3 加严为 5e-3，正演/JVP/伴随一致，未降低 FP64 或根容差。完整二进制测试及日志摘要见[发布验证](../verification/performance-0.6.0.dev7/verification.json)。

**未获得全面提速。** 小任务中 CUDA 启动、同步和 Torch 合成开销可超过收益；Green 纯正演可用 `device_pipeline=False` 复查主机合成路径。面波 `differentiation="forward"` 保留原解析核，部分小批次更快。临界群速度坐标的保护会回到原解析路径，因此不能用理想 O(L) 或单个非临界根的速度预测完整调用。CPU Green 结构算法基本未变；该项时差不能解释成伴随算法收益。

自适应 Levin 仍在 CPU，当前新 CUDA 反向路径使用固定网格。Levin 只适用于实际含振荡波数积分的 Green；RF FFT 与频散根没有被改称 Levin。上一版 [Levin 与 PTAM 的误差/速度范围](QUADRATURE_0.6.0.dev6.zh-CN.md) 和 [disba/pyfk 对照](PERFORMANCE_0.6.0.dev5.zh-CN.md) 保留为历史结果，未冒充本版重新测试；高阶模 disba 核不作真值。MPS RF/面波为 CPU FP64 桥接。

## 可复查数据

- [summary.json](../verification/performance-0.6.0.dev7/benchmarks/summary.json)
- [before-apple-final.json](../verification/performance-0.6.0.dev7/benchmarks/before-apple-final.json)
- [after-apple-final.json](../verification/performance-0.6.0.dev7/benchmarks/after-apple-final.json)
- [before-cuda-final.json](../verification/performance-0.6.0.dev7/benchmarks/before-cuda-final.json)
- [after-cuda-final.json](../verification/performance-0.6.0.dev7/benchmarks/after-cuda-final.json)
- [before-cuda-large.json](../verification/performance-0.6.0.dev7/benchmarks/before-cuda-large.json)
- [after-cuda-large.json](../verification/performance-0.6.0.dev7/benchmarks/after-cuda-large.json)

同名 NPZ 保存完整输出；JSON 保存全部重复计时和 SHA-256。源码仓库中的 `optimization/analytic-adjoint-20261021/measure.py` 可复现配置。
