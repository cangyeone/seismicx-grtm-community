# 振荡波数积分优化：0.6.0.dev6

[English](QUADRATURE_0.6.0.dev6.md) | 简体中文

本轮优化已有的 **Bessel–Levin 配点积分**，并增加正演/结构 JVP 联合 Levin 积分和 `method="auto"`。强振荡时直接处理 Bessel 结构，比逐个采样振荡周期更合适；PTAM、Shanks 和 Gauss 接口继续保留。Levin 方法依据 [Levin (1997)](https://www.tau.ac.il/~levin/OscIntAnal.pdf)，Chebyshev/截断 SVD 的相关分析见 [Chen、Serkh、Bremer](https://arxiv.org/abs/2211.13400)。文献的适用假设不等同于任意 GRTM 输入的无限积分保证。

## 本次改动

- 相邻面板合并采样，模型和材料准备次数减少；每次求解重新计算所有 GRTM 核值。
- 缓存与材料无关的 Gauss/Chebyshev 基和实数 SVD；同一几何供所有复数通道、频率及方向导数复用。SVD 缓存限 256 项。
- 正演与解析结构 JVP 共享配点矩阵、积分网格及残差检查。没有引入参数差分，也不对自适应选点流程求导。
- 保留原来的奇异值阈值 `1e-13`、独立残差采样、舍入误差诊断下限、逐个最坏面板细分和预算耗尽报错。
- `method="auto"` 按 `cutoff * max(distance) > 512` 选择 Levin，否则选择批量 Gauss。这只是成本启发式；不会放宽误差检查，仍可显式选方法。默认方法保持 Gauss。

## CPU 单频率对比

Apple M4 Max；源代码基线 `54cf35f`，同一原生库、相同模型，原生扩展精度/稳定模态表示均未改变。4 层原生正演、2 层联合正演/JVP，全部 30 个矩张量 Green 系数。区间 `[0,4] km⁻¹`，`atol=1e-10, rtol=0, order=12`；预热后 5 次中位数。首轮时间、全部重复计时、模型和输出数组均保存在原始 JSON。缓存仅包含与材料无关的几何，每轮仍重新求解传播。

参考为另一个 Gauss 网格，20 阶、`atol=1e-12`。下表误差是原生系数的**绝对差**，不是相对误差或 SI 波形误差。联合同深度算例加回解析直接场，低频算例使用稳定模态表示。

| Case | Old method / ms | New method / ms | Ratio | Old / new nodes | Max field / JVP absolute difference vs reference |
|---|---:|---:|---:|---:|---:|
| near | levin 24.058 | Levin 7.801 | 3.08× | 1388 / 1388 | 2.52e-14 |
| far | levin 24.572 | Levin 7.767 | 3.16× | 1388 / 1388 | 6.04e-14 |
| shallow_far | levin 26.093 | Levin 8.138 | 3.21× | 1438 / 1438 | 4.31e-14 |
| joint_separated | gauss 188.974 | Levin 17.962 | 10.52× | 9360 / 1488 | 1.82e-13 / 3.67e-14 |
| joint_equal_depth | gauss 1879.250 | Levin 239.727 | 7.84× | 6408 / 1388 | 2.1e-13 / 3.32e-13 |
| joint_quasistatic | gauss 1143.052 | Levin 56.145 | 20.36× | 3888 / 610 | 1.06e-13 / 1.11e-13 |

近场（10/30 km）新 Gauss 为 2.020 ms，新 Levin 为 7.801 ms，Gauss 更快；`auto` 会选择 Gauss。远场（100/1000 km）新 Gauss 为 33.994 ms、新 Levin 为 7.767 ms，Levin 快 4.38 倍。浅源远场对应 101.378/8.138 ms，快 12.46 倍。不能把旧 Gauss 到新 Levin 的联合算法收益说成同一种算法的底层加速。

## 与原生 PTAM 的完整计算对照

另一个固定模型：4 层、震源深 8 km、距离 10/30/100 km、16 点、dt=0.4 s、全部 30 通道。原生 PTAM 使用 8 个极值，单线程；表中计时包括模型准备和 FFT，预热后 3 次中位数。独立 Gauss 参考为 20 阶、`atol=1e-13`、cutoff=8；cutoff=4 到 8 的变化保存在 JSON。Levin/auto 用 cutoff=4、`atol=1e-12`。

| Method | Period length (km) | ms | Max absolute spectrum error | Relative L2 spectrum error |
|---|---:|---:|---:|---:|
| levin | — | 158.118 | 2.45e-15 | 7.59e-12 |
| auto | — | 38.751 | 2.45e-15 | 7.51e-12 |
| ptam | 400.0 | 2.731 | 1.25e-07 | 0.000585 |
| ptam | 800.0 | 4.642 | 6.39e-08 | 0.000174 |
| ptam | 1600.0 | 7.819 | 6.16e-08 | 0.000151 |
| ptam | 3200.0 | 12.876 | 7.74e-08 | 0.000202 |
| ptam | 6400.0 | 24.260 | 1.36e-07 | 0.00025 |
| ptam | 12800.0 | 46.715 | 1.35e-07 | 0.000248 |

**没有得出“同误差下全面快于 PTAM”的结论。** 本例粗 PTAM 更快，但扫描设置没有达到新积分的误差；加密积分步长也没有消除它的误差平台。旧 PTAM 仍使用原有 Bessel 近似，新积分用系统 Bessel 函数，因此该误差差异属于完整实现的对照，不能全部归因于 PTAM 的数学变换。近场完整计算用 `auto` 比固定 Levin 更合适。

保留了两项额外诊断：1000 km 接收点配短时间窗导致阻尼场接近零，此时相对误差失去实用意义；更长时间窗的极低频原生 Gauss 参考在 2048 面板内未达到 1e-13，明确拒绝返回，未用于加速结论。低频稳定场/JVP 的独立验收见上表及原有连续积分测试。

## 接口与范围

```python
import grtm
import numpy as np

solver = grtm.Solver(backend="cpu")
model = grtm.example_model()  # native units: km, km/s, g/cm^3
model.update(log2_samples=5, dt=0.4, distances=[100., 1000.],
             t0=[0., 0.], source_type=1, ptam=0)
with solver.wavenumber_kernel(model, frequency_index=4) as kernel:
    field = kernel.integrate(4., method="auto", atol=1e-10, rtol=0)
    direction = np.zeros_like(model["layers"])
    direction[0, 2] = 1.  # top-layer Vs direction, analytic chain rule
    joint = kernel.integrate_jvp(4., direction, method="auto",
                                 atol=1e-10, rtol=0)
    print(field["method"], field["kernel_calls"], field["evaluations"])
    print(joint["value"].shape, joint["jvp"].shape)
```


`cutoff` 始终以 km⁻¹ 表示；生产计算还应检查增大 cutoff 后的变化。`value`/`jvp` 为 `[receiver,30]` 原生系数。完整波形可调用 `solver.compute_quadrature(model, cutoff=4., method="auto", ...)`；NED/SI 合成使用 `forward_quadrature`。

**这条自适应积分路径在 CPU 上运行。** 即使 Solver 载入 CUDA/Metal 库，也不会称其为 GPU 积分。本轮相速度/群速度 CUDA 优化见[独立报告](PERFORMANCE_0.6.0.dev6.zh-CN.md)，不得将其比值挪用到这里。已有 CUDA 固定网格正演/微分接口未改为 CPU 自适应路径。

残差和嵌套规则提供有限区间误差估计；没有无穷尾项和全部核舍入误差的严格证书。`certified=False`、`tail_bound=None` 保留，`require_certified=True` 拒绝执行。同深度原始通道不能只靠振荡外推证明普通积分存在；`integrate_jvp` 保留解析直接场恢复。

证据：[单频率数据](../verification/performance-0.6.0.dev6/quadrature/cpu.json)、[PTAM 完整计算](../verification/performance-0.6.0.dev6/quadrature/ptam-cpu.json)、[低频失败记录](../verification/performance-0.6.0.dev6/quadrature/ptam-low-frequency-failure.json)。私有复现脚本位于相同目录的 `measure.py` 和 `ptam_check.py`。
