# 格林函数、接收函数、频散与微分的 CPU/CUDA 测试 — 0.6.0.dev4

[English](PERFORMANCE_0.6.0.dev4.md) | 简体中文

本次寻根优化保留原精度要求，大幅缩短 GRTM 自身的批量频散耗时。
CPU 并行与 CUDA 的收益随任务规模变化。**格林函数的结构微分仍在 CPU，即使正演使用 CUDA；**
接收函数和频散的 CUDA 微分则确实在 GPU 执行。以下速度与数值精度结果不等于后验收敛速度或地质反演准确率。

## 同一 GRTM 的正演与微分

2026-10-20；Threadripper PRO 3955WX（16 物理核/32 逻辑核）、RTX 5090
32 GB（`cuda:1`），Linux/WSL，Python 3.13.5、Torch 2.8.0+cu128。
主机和 GPU 与其他任务共享，未停止无关作业。全程 FP64/complex128。
复用已准备对象，每项预热一次，完整调用重复 5 次取主机墙钟中位数，CUDA 在计时前后同步。
主表不计构造和模型上传，包含公开接口校验、主机调度和输出处理；JSON 保留逐次耗时和单独记录的准备时间。
每轮轮换操作顺序，各后端顺序运行。共享机器负载会影响绝对时间与加速比。

格林函数：时域位移、1 个模型、4 接收点、64 采样、固定 20205 个波数积分节点，32 层的 Vp/Vs 共 64 参数。
测试一个方向 JVP 与标量目标的伴随 VJP，不构造完整稠密 Jacobian。
接收函数：16 模型、32 层、1 条射线、6 个高斯频带、81 个时间点、nfft=2048、41 个质点运动窗口；
计算固定 RF/视 Vs 余切对应的顶深、Vp、Vs、密度梯度，首层顶深固定。
频散：16 模型、Rayleigh 基阶、2–50 秒的 12 个周期，Vs 核包含全部层。
三类微分任务的输出不同，不能直接横比各类原始耗时。

| 任务（32 层） | CPU 单线程 (s) | CPU 多线程 (s) | CUDA / 混合 † (s) | CPU 单线程/目标 | CPU 多线程/目标 |
| --- | --- | --- | --- | --- | --- |
| 格林函数 / B=1 正演 | 0.738855 | 0.178445 | 0.342739 | 2.16× | 0.52× |
| 格林函数 / B=1 仅 JVP † | 1.654211 | 0.407322 | 0.429144 | 3.85× | 0.95× |
| 格林函数 / B=1 仅 VJP † | 7.721161 | 1.430158 | 1.281868 | 6.02× | 1.12× |
| 格林函数 / B=1 正演＋VJP † | 8.528672 | 1.403220 | 1.754349 | 4.86× | 0.80× |
| 接收函数 / B=16 正演 | 0.442220 | 0.421455 | 0.024811 | 17.82× | 16.99× |
| 接收函数 / B=16 仅 VJP | 0.674402 | 0.741328 | 0.023473 | 28.73× | 31.58× |
| 接收函数 / B=16 正演＋VJP | 1.085292 | 1.103559 | 0.052522 | 20.66× | 21.01× |
| 相速度 / B=16 正演 | 0.056294 | 0.014906 | 0.039368 | 1.43× | 0.38× |
| 相速度 / B=16 正演＋全部层 Vs 核 | 0.549999 | 0.120235 | 0.060831 | 9.04× | 1.98× |
| 群速度 / B=16 正演 | 0.063379 | 0.019523 | 0.039908 | 1.59× | 0.49× |
| 群速度 / B=16 正演＋全部层 Vs 核 | 3.902394 | 0.557266 | 0.148562 | 26.27× | 3.75× |

加速比小于 1 表示目标更慢。† 格林函数目标列为 CUDA 正演＋CPU 16 线程微分；
JVP/VJP 本身仍是 CPU 计算，不能称为原生 CUDA 结构微分。
其他 CUDA 行使用 1 个主机 Torch 线程，传播和微分在 GPU 执行。
CPU 多线程列中格林函数/频散使用 16 线程，接收函数使用 4 个 Torch 线程，其收益不能当作单核算法收益。
初次 RF16 基准在 8 层/B16 配置中持续数分钟后停止；已完成的 12 条预试记录单独保留，
没有把累计等待时间伪造为单次超时耗时。
“仅 VJP”复用已有线性化/计算图；“正演＋VJP”重新构造，两者分别实测，不由总时间相减。
共享负载随时间变化，中位数不保证可以相加。

全部 8 层、单模型数据、格林函数两种 CUDA 主机线程配置及逐次计时保留在综合 JSON。
小任务的 GPU 延迟可能高于 CPU，主表的批量结果不能推及任意规模。

## 只算频散：与原始 disba 比较

32 层、12 周期，独立批次预热后重复 5 次。每个 CUDA 比值使用该批次交错测量的 disba 基线，
不与另一批次的 CPU 时间混除。disba 0.7.0 不修改、不增加多线程：Dunkin、dc=.005、群速度 dt=.025。
GRTM 根容差 1e-10 km/s，群速度采用解析周期导数。**这是各自原始默认方法的比较，精度预算并不相等。**
disba 原生寻根停止条件使用约 1e-6 的相对速度尺度；dc 是搜索间隔，不是最终根误差。

CPU 单线程：

| B | kind | disba CPU1 (s) | GRTM CPU1 (s) | speedup |
| --- | --- | --- | --- | --- |
| 1 | phase | 0.002621 | 0.005037 | 0.52× |
| 1 | group | 0.005139 | 0.003489 | 1.47× |
| 16 | phase | 0.044729 | 0.059479 | 0.75× |
| 16 | group | 0.082886 | 0.061486 | 1.35× |
| 64 | phase | 0.167922 | 0.238703 | 0.70× |
| 64 | group | 0.342791 | 0.250233 | 1.37× |

GRTM 使用 16 个模型工作线程，基线仍为原始单线程 disba：

| B | kind | disba CPU1 (s) | GRTM CPU16 (s) | speedup |
| --- | --- | --- | --- | --- |
| 16 | phase | 0.043589 | 0.021545 | 2.02× |
| 16 | group | 0.086080 | 0.019837 | 4.34× |
| 64 | phase | 0.168736 | 0.046853 | 3.60× |
| 64 | group | 0.341671 | 0.053518 | 6.38× |

CUDA 与该批次自身交错测量的单线程 disba：

| B | kind | disba prepared (s) | CUDA prepared (s) | speedup | disba full (s) | CUDA full (s) | full speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | phase | 0.002694 | 0.038230 | 0.07× | 0.002739 | 0.060934 | 0.04× |
| 1 | group | 0.005248 | 0.040977 | 0.13× | 0.005633 | 0.062361 | 0.09× |
| 16 | phase | 0.043558 | 0.040237 | 1.08× | 0.046289 | 0.057985 | 0.80× |
| 16 | group | 0.084485 | 0.042696 | 1.98× | 0.083519 | 0.065132 | 1.28× |
| 64 | phase | 0.172809 | 0.054301 | 3.18× | 0.171897 | 0.054602 | 3.15× |
| 64 | group | 0.346198 | 0.055410 | 6.25× | 0.338251 | 0.078363 | 4.32× |

CUDA 完整调用从主机速度模型开始，包含对象构造、搬运和网格规划；常态调用复用设备模型与网格。
CPU16 是 GRTM 显式模型级并行，disba 对照始终保持原单线程。
32 层/B16 时，旧版 GRTM CPU 相/群速度耗时 3.303400/3.459268 秒，
新版为 0.059479/0.061486 秒，约提速 55.5/56.3 倍。
旧 CUDA 为 .225457/.258540 秒，新版 .040237/.042696 秒，约提速 5.6/6.1 倍。
旧新结果为同一共享硬件上的不同测量批次，并非配对实验。
此规模下纯 CPU 单线程相速度仍慢于默认 disba；CUDA 单模型仍受延迟限制，B64 批处理收益更大。

## 完整 Vs 敏感核：与原始 disba 比较

32 层/B16/12 周期，包含寻根及全部层 Vs 导数；各自预热后测 5 次。
CPU 和 CUDA 比较批次分别列出 disba 基线，不混用中位数。

| kind | disba CPU1 (s) | GRTM CPU1 (s) | CPU speedup | disba CPU1, CUDA campaign (s) | CUDA (s) | CUDA speedup |
| --- | --- | --- | --- | --- | --- | --- |
| phase | 6.944697 | 0.571227 | 12.16× | 6.850183 | 0.052390 | 130.75× |
| group | 13.634333 | 3.909884 | 3.49× | 13.643346 | 0.134783 | 101.22× |

disba 保留默认 2.5% 单边模型扰动，GRTM 为解析 dual/hyper-dual 微分。
与 disba 核的差不能直接当成独立精度界，也不能把完整核加速比用于声称纯频散提速。

## 精度与执行设备核验

129 个模型、4339 个可用模态点，覆盖正常梯度、低速层、薄层、密度跳变、厚层/短周期；
包含两种波、0/1 阶、缺失掩码、乱序/重复周期与群速度。
优化相速度与原始复数全扫描（更严的 1e-12 km/s 容差）最大差约 1.73e-12 km/s。
独立 70 位计算采用一般 4×4 本征矩阵求逆及六个外积子式，不使用优化后的五状态递推。
48 个相/群速度测试点结果如下：

| Method | Cases | Max phase error (km/s) | Max group error (km/s) |
| --- | --- | --- | --- |
| GRTM auto CUDA | 48 | 1.502e-12 | 1.202e-10 |
| Original complex scan CPU | 48 | 1.783e-11 | 1.358e-09 |
| Original disba CPU (available cases) | 45 | 2.312e-06 | 1.438e-03 |

此 70 位子审计中的原始复数扫描使用默认 1e-10 根容差。
disba 有 45 个共同可比较点：一个两层 Love 模型的 3 周期曲线在相、群速度调用均抛出 DispersionError。
失败原样记录，没有记成零误差或静默删除。GRTM 全部 48 点与共同 45 点的上述最大误差相同。
结果支持测试模型中的精度保持，不构成任意模型的模态完备证明、严格浮点误差界，也不证明地质反演更准。

安装包综合基准同时核验 CPU/CUDA 一致性、伴随恒等式与中心差分（差分只用于验证）：

| Family | Metric | Maximum over benchmark cases |
| --- | --- | --- |
| green | adjoint_identity_relative_error | 6.925e-12 |
| green | forward_relative_cpu_error | 3.753e-13 |
| green | jvp_central_difference_relative_error | 4.572e-07 |
| green | vjp_relative_cpu_error | 1.724e-19 |
| receiver_function | directional_vjp_central_difference_relative_error | 1.687e-08 |
| receiver_function | rf_max_abs_cpu_error | 5.204e-16 |
| receiver_function | rf_relative_cpu_error | 1.159e-15 |
| receiver_function | vjp_relative_cpu_error | 2.575e-15 |
| dispersion | forward_relative_cpu_error | 2.118e-11 |
| dispersion | forward_vs_kernel_relative_cpu_error | 2.371e-08 |

10 个二进制构建任务均通过安装后的 CPU 套件，每项发现 105 测试：Linux 63 通过/42 跳过，
macOS 64 通过/41 跳过。实际 Linux/CUDA 96 通过/9 跳过，macOS/Metal 89 通过/16 跳过。
可选依赖及平台条件导致跳过，详见[二进制验证记录](../verification/preview-0.6.0.dev4.json)，
其中说明了只涉及 Metal 并发舍入检查的测试断言修正，求解器与数值精度未改。

## 复现与原始记录

安装[匹配的 dev4 编译包](PREVIEW_0.6.0.dev4.zh-CN.md)；稳定 PyPI 仍为 0.5.1。
二进制源码提交为 `fbf130f9ce259faa89df6557c002dfb16acc8230`。
综合数据来自安装后的 CI wheel，不是源码路径导入。

```sh
python benchmarks/forward-derivatives/benchmark_forward_derivatives.py --device cuda:1 --threads 1 16 --layers 8 32 --batches 1 16 --repeat 5 --output comprehensive.json
python benchmarks/surface-wave/benchmark_surface_cpu.py --threads 1 --layers 32 --batches 16 --tasks sensitivity --repeat 5 --output kernels-cpu.json
python benchmarks/surface-wave/benchmark_surface_wave.py --device cuda:1 --layers 32 --batches 16 --repeat 5 --output kernels-cuda.json
python benchmarks/surface-wave/benchmark_surface_dispersion.py --device cuda:1 --repeat 5 --output dispersion-cuda.json
```

外部对照脚本另装仅测速用的 `disba==0.7.0`。GRTM 生产路径不依赖 disba。
完整实现仓库还提供 `examples/validate_surface_search.py --models 128 --mp-models 8` 重跑 70 位审计。
[面波接口与数学说明](SURFACE_WAVES.zh-CN.md)解释半解析方法和适用范围。

- [comprehensive-dev4.json](../verification/performance-0.6.0.dev4/comprehensive-dev4.json)
- [comprehensive-preliminary-rf16.json](../verification/performance-0.6.0.dev4/comprehensive-preliminary-rf16.json)
- [comprehensive-rf-surface-dev4.json](../verification/performance-0.6.0.dev4/comprehensive-rf-surface-dev4.json)
- [search-cpu-final.json](../verification/performance-0.6.0.dev4/search-cpu-final.json)
- [search-cpu16-final.json](../verification/performance-0.6.0.dev4/search-cpu16-final.json)
- [search-cuda-final.json](../verification/performance-0.6.0.dev4/search-cuda-final.json)
- [cpu-validation.json](../verification/performance-0.6.0.dev4/cpu-validation.json)
- [search-final-validation.json](../verification/performance-0.6.0.dev4/search-final-validation.json)
- [kernels-cpu-disba.json](../verification/performance-0.6.0.dev4/kernels-cpu-disba.json)
- [kernels-cuda-disba.json](../verification/performance-0.6.0.dev4/kernels-cuda-disba.json)
