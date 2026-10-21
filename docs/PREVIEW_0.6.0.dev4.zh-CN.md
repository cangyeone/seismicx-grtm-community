# CPU/CUDA 寻根与微分预览版 — 0.6.0.dev4

[English](PREVIEW_0.6.0.dev4.md) | 简体中文

本版优化 GRTM 面波寻根与群速度解析切线，并统一测试格林函数、接收函数、频散及其结构微分。
具体速度和精度见[完整测试报告](PERFORMANCE_0.6.0.dev4.zh-CN.md)。稳定 PyPI 仍为 0.5.1。

[批量接口指南](BATCH_EXECUTION.zh-CN.md)与 [2026-10-21 批量扩展测试](BATCH_SCALING_20261021.zh-CN.md)
补充每模型耗时、每秒模型数及真实执行设备。这些测试使用同一编译包，无需升级版本。

从[二进制发布页](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev4)
下载与 Python、平台匹配的 wheel，校验发布页中的 SHA-256，再安装实际文件：

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev4-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

尖括号内容是占位符。提供 CPython 3.10–3.14、Apple Silicon macOS 11+（CPU/Metal）、
Linux/WSL x86_64 glibc 2.28+（CPU/CUDA）的包。CUDA 目标计算能力 12.0，已在 RTX 5090 验证。
CUDA 面波、联合 RF 和框架适配需要与 Python、GPU、驱动兼容的 PyTorch。
沿用现有科研及非商业许可证；社区仓库分发编译包，不包含求解器实现源码。

```python
import numpy as np
from grtm import SurfaceWave

# 顶深 km、Vp/Vs km/s、密度 g/cm³；最后一层为半空间。
layers = np.array([[0.,3.2,1.65,2.1], [.8,5.6,3.15,2.6],
                   [5.,6.7,3.8,2.9], [25.,8.1,4.6,3.3]])
models = np.repeat(layers[None], 16, axis=0)
sw = SurfaceWave(models, threads=16)  # 显式启用 CPU 模型级并行
c = sw.phase_velocity([5.,10.,20.])
u = sw.group_velocity([5.,10.,20.])
k = sw.sensitivity([5.,10.,20.], kind='group', parameter='all')['kernel']
print(c.shape, u.shape, k.shape)

# CUDA 不设置 threads=16，使用默认 threads=1。
# gpu = SurfaceWave(models, device='cuda:0')
# c_gpu = gpu.phase_velocity([5.,10.,20.])

# 原始复数全扫描参考路径：
reference = SurfaceWave(layers, search='scan', root_tolerance=1e-12)
```

默认 `search="auto"` 保留 GRTM 本征矩阵、FP64/complex128 和 1e-10 km/s 根容差。
实数试算结果使用原始复数 GRTM 方程作根区间与残差核验。
CPU 基阶连续追踪有适用条件与回退；低速层、强跳变、高阶模态仍执行全局发现。
群速度正演只计算所需解析一阶导数；结构核保留原复数 dual/hyper-dual 微分。
生产路径不调用 disba，也不对每个参数扰动模型后重求根。
参数、单位与模态限制见[面波接口说明](SURFACE_WAVES.zh-CN.md)。

CPU `threads` 默认为 1，按独立模型行并行；`search="scan"` 的旧寻根保持串行。
CUDA 使用调用方设备和 stream，网格规划在主机执行。单条短曲线仍可能 CPU 更快。
与 disba 的比较保留各自公开默认精度设置，不能当作相同误差预算的性能竞赛。

格林函数结构 JVP/VJP 当前仍在 **CPU**，使用 CUDA 正演时也是混合流程。
联合 RF 的正演/反向与面波敏感核在选择 CUDA 后确实于 GPU 计算。
Metal 用于点源格林函数正演；RF 与面波的 GPU 接口当前为 CUDA。
原生自定义 backward 只支持一阶微分。

发布附件包含二进制审计、设备测试说明和校验和。完整报告附原始重复耗时、精度审计及公开接口测速脚本。
其他接口见[物理正演](PYTHON_FORWARD.zh-CN.md)、[格林函数微分](RESEARCH_PREVIEW.zh-CN.md)、
[接收函数](DIFFERENTIABLE_RF_CPU.zh-CN.md)。
