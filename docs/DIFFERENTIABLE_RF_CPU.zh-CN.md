# CPU 可微接收函数预览版 0.6.0.dev2

**以下为 dev2 历史说明。**当前 [dev4 预览版](PREVIEW_0.6.0.dev4.zh-CN.md)已支持原生 CUDA 正演/VJP。
最新[批量示例与执行指南](BATCH_EXECUTION.zh-CN.md)及 [2026-10-21 吞吐测试](BATCH_SCALING_20261021.zh-CN.md)
说明现有接口；下文“仅支持 CPU”的限制属于 dev2，不适用于 dev4。

新增公开接口 `JointRFForward`、`joint_receiver_forward`、`hmc_sample`、`dram_sample`、`rank_diagnostics` 及浅部控制点模型/联合反演问题。旧公开版没有提供项目中可微 RF / HMC 接口，此次补入自己的 CPU 实现。

发布二进制轮包：macOS Apple Silicon / Python 3.12、Linux x86_64 / Python 3.13（manylinux_2_28）。请从 [v0.6.0.dev2](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev2) 下载匹配包，并检查 SHA-256；安装 NumPy、SciPy、PyTorch 后通过 pip 安装。Linux 可先从官方 CPU 源安装 PyTorch。

RF 使用 float64 / complex128 的 CPU 散射递推与 FFT，支持一阶自动微分和多次反射。新 RF 接口仅接受 `device='cpu'`。原 0.6.0.dev1 的格林函数二进制保持原样；没有把开发目录的其他改动打入发布包。

层表为深度 km、Vp/Vs km/s、密度 g/cm³，可追加 Qp/Qs；最后一行是半空间。4 列为弹性，6 列可使用恒定 Q 衰减。层顶从 0 严格递增，需满足固体与非临界射线条件。输出 RF 为 `[batch,ray,band,time]`；P 到时径向归一化属于调用端约定，需与观测处理一致。

HMC 采用物理目标、logit 边界变换及 Jacobian、一阶梯度和 Metropolis 校正；自适应在预热后停止。DRAM 是我们根据论文独立实现的程序，未收录第三方作者反演源码。完整使用示例见 [English instructions](DIFFERENTIABLE_RF_CPU.md)。

面向用户的反演与模型接口见 [FlowRF](https://github.com/cangyeone/flow-assisted-hmc-receiver-function-inversion)，该项目通过依赖安装 GRTM，不复制求解器。GRTM 原有研究/非商业许可证保持不变，发布的是二进制模块，不是求解器开源源码。
