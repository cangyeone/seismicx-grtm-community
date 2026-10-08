# 0.3 接口、梯度与速度验证

[English](FORWARD_VERIFICATION.md) | 简体中文

2026-10-06。本版完成 NED / SI 矩张量和 SDR、流式/张量 Earth batch、结构 Jacobian/VJP、
PyTorch/JAX forward 和应力应变。接口说明见 [PYTHON_FORWARD.md](PYTHON_FORWARD.zh-CN.md)。

## 功能与正确性

- macOS ARM64、Python 3.14、NumPy 2.5.3、Torch 2.14.1、JAX 0.11.2：17 项测试中 15 项通过，2 项 GPU 测试跳过。
- 远程 Ubuntu/WSL x86_64、Python 3.8.10、NumPy 1.24.4：CPU wheel 13 项通过，GPU/框架 4 项跳过；CUDA wheel 15 项通过，未安装的 Torch/JAX 2 项跳过。
- 源码 tar.gz 在独立 macOS venv 中通过 pip 安装并运行 CPU 测试。
- Torch：普通 backward、共享源 Earth batch、SDR 角度链式梯度、复频谱损失；JAX：jit、grad、JVP、jacfwd、vmap、显式 batch、共享源梯度及复频谱损失。均与 NumPy VJP 一致；CPU 框架梯度也与独立固定网格差分损失对比。
- CUDA 实测覆盖物理场合成和结构 Jacobian；框架测试在 macOS CPU 运行，没有声称测试了 Torch CUDA tensor 或 JAX GPU device 的完整链路。
- 独立单力源空间偶合验证六个 NED 矩张量分量的符号、因子和归一化，相对差异小于 2×10⁻⁶，包含震源上方、下方和跨材料层接收点。
- 完整 Cartesian 接收点差分验证位移梯度，包含柱坐标基向量导数；应变/应力张量对称。
- 所有六个矩张量及三个单力基源满足自由表面零牵引，测试阈值 2×10⁻¹¹，实测典型值约 10⁻¹⁵。
- Vp、Vs、ρ、Qp、Qs、界面深度、深层参数，以及应力的接收层直接本构导数，均作独立中心差分校验。
- SI/native 换算、无量纲源历史因果卷积、batch 顺序/错误索引、固定积分网格及界面拓扑拒绝均验证。
- ASan/UBSan：原生 kernel/CLI 测试及六个结构参数的 C API 所有权/切线工作区 smoke test 通过。
- pip check 在本地和远程均通过；CUDA runtime 静态链接，ldd 未显示 libcudart 或私有构建目录。

## 修正的物理问题

1. 矩张量轴对称径向项原来用了垂向振幅，修正为水平振幅，同时修正对应 dz/dr 通道。
2. 矩张量 SH 上行 order-1 反射项改为负号，下行 order-2 直接项改为正号；由独立单力偶合验证。
3. 跨层下行 PSV 使用下行递推矩阵，接收层反射传播使用接收层厚度，修正震源以下接收点的结果。
4. 单力源旧垂向基函数按向上符号存储，高层合成时转换为 NED 向下；原 compute() 的原始通道契约保持不变。
5. 高层应变采用对称梯度，并包含方位角和柱坐标基向量项；不沿用旧 green.py 中的应变公式。

这些修正会改变原矩张量源的相关输出。历史测速使用外部 [YunyiQian/grtm](https://github.com/YunyiQian/grtm) 的 Fortran 参考程序；该实现不随本仓库分发。

## 新版 wheel 与 Fortran

同机 Threadripper PRO 3955WX（16 核/32 线程）和单张 RTX 5090，三次重复中位数，预热一次。
近场四层模型：hs=8 km、hr=0、r=10 km、dt=0.04 s、2048 点、PTAM=0，积分周期与原 Fortran 对齐。
所有实现仅计算单力源位移基函数；表中为积分时间，完整调用另见原始 JSON。

| 实现 | 积分时间 / s | 相对 Fortran -O3 |
|---|---:|---:|
| 原 Fortran -O3 | 11.682756 | 1× |
| 0.3 C wheel，32 线程 | 0.565346 | 20.7× |
| 0.3 CUDA wheel，device 0 | 0.117619 | 99.3× |

wheel 使用 portable CPU 构建（GRTM_NATIVE=OFF），不同于此前启用本机 CPU 调优的原生库。
更多距离/采样规模的原生库测速见 [另行冻结的公开 FK 对比](../benchmarks/fk/README.zh-CN.md)。
本表不把应力、Torch/JAX callback 或结构 Jacobian 的额外工作算入纯积分加速比。

## Forward、Earth batch 与 Jacobian

四层模型，64 点、dt=0.2 s、2 个距离（10/30 km），合成矩张量位移、应变、应力；
以下计时包含 Python 调用、内存分配、FFT 和物理场合成。

| 工作量 | CPU / s | CUDA / s |
|---|---:|---:|
| 单模型 forward，32 主机线程 | 0.005295 | 0.009077 |
| 32 模型，逐模型调用 | 0.147209 | 0.277019 |
| 32 模型，workers=4 / threads=8 | 0.120489 | 0.156355 |

这种小任务 CPU 更快；CUDA 启动/传输成本尚未摊薄。batch 目前逐模型调用，不是模型融合 kernel。
本组 CPU/CUDA 物理场相对 L2 差异：位移 4.391×10⁻¹⁴、应变 1.878×10⁻¹⁴、应力 1.904×10⁻¹⁴。

同一模型的复频谱应力 Jacobian，选择 7 个结构参数，主机 32 线程：

| 方法 | 完整 Jacobian 时间 / s |
|---|---:|
| 半解析链式导数，P 次切线积分 | 0.070646 |
| 固定网格中心差分，2P 次求解 | 0.076579 |

本组半解析方法约快 1.08×；不能推广为所有规模都更快。常规频率使用 double 切线，
准静态病态频率使用 long double 切线，工作区在线程内复用。
梯度相对 L2 差异为 1.435×10⁻⁵（差分步长 1e-4 相对参数值）。
解析方法的主要价值还包括不依赖差分步长、矩张量解析梯度复用基函数，以及可收缩的 VJP。

## 实际数值边界

PTAM=3 的密度导数与差分相对差异约 2.51×10⁻⁹，但 PTAM 极值分支切换仍可能不光滑。
额外的极端算例（4 点、dt=1000 s、r=1000 km）中，密度结构梯度的解析/差分诊断差异约
0.98（Linux）和 1.00（macOS ARM64）。这种准静态病态区域的结构梯度**不能视为已验证**。
两种方法的差异不足以单独判定哪个更准确，需要独立更高精度参考和积分收敛检查。
macOS ARM64 的 long double 没有额外精度；Linux 的 80 位路径也不是任意参数下的保证。
接口提供 check_step 和可复用固定网格，使用非常长周期/病态材料时应检查这些诊断。

本历史 0.3 报告的原始记录保留在私有开发仓库，不随公开指南附带。
另行冻结的 [FK 对比](../benchmarks/fk/README.zh-CN.md)提供自己的公开数据和适用范围；
二进制安装检查见[发行验证](../verification/README.md)。历史计时不是 0.6 预览版的新测速。
