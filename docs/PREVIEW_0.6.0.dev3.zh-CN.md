# 完整方法二进制预览版 — 0.6.0.dev3

本版整合 GRTM 格林函数、结构微分、接收函数与 HMC/DRAM 接口，并新增 Rayleigh/Love 面波频散及半解析敏感核。
完整实现源码保存在维护者的主仓库；此 community 仓库提供编译包、接口说明和数值结果。许可沿用 SeismicX GRTM 非商业科研许可。

## 下载与安装

从 [Release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev3) 下载与 CPython 版本、平台匹配的 wheel。
支持 CPython 3.10–3.14：Apple Silicon macOS 11+（CPU + Metal）和 Linux/WSL x86_64 / glibc 2.28+（CPU + CUDA）。
每种平台/Python 组合各有一个文件；核验 [SHA256](../downloads/SHA256SUMS-0.6.0.dev3) 后安装该文件。

```sh
python -m pip install /path/to/downloaded.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

PyPI 稳定版仍是 0.5.1；普通 `pip install seismicx-grtm` 不会获得此预览功能。
面波 NumPy CPU 接口不要求 PyTorch；CUDA 与 RF/HMC 接口需要 PyTorch。
CUDA 二进制针对计算能力 12.0，实机为 RTX 5090；框架实测环境为 Linux / Python 3.13 / Torch 2.8.0+cu128。
其他 Python 版本执行 CUDA 时仍须有兼容其 Python、GPU 和驱动的 PyTorch 发行版。
macOS 面波当前使用 CPU；包内 Metal 后端用于点源格林函数。

## 从速度模型计算频散和敏感核

```python
import numpy as np
from grtm import SurfaceWave

# 顶深 km、Vp/Vs km/s、密度 g/cm3；末层为半空间。
layers = np.array([[0., 3.2, 1.65, 2.1], [.8, 5.6, 3.15, 2.6],
                   [5., 6.7, 3.8, 2.9], [25., 8.1, 4.6, 3.3]])
sw = SurfaceWave(layers, device='cpu')  # Linux CUDA 改用 'cuda:0'
periods = [5., 10., 20.]
c = sw.phase_velocity(periods, wave='rayleigh')
u = sw.group_velocity(periods, wave='rayleigh')
k = sw.sensitivity(periods, kind='group', parameter='velocity_s')['kernel']
print(c, u, k.shape)
```

[面波使用与半解析推导](SURFACE_WAVES.zh-CN.md)说明批处理、单位、层参数、自动微分和适用范围。
[速度报告](SURFACE_WAVE_BENCHMARKS.zh-CN.md)将纯 CPU、纯频散 CUDA、完整敏感核分开：
完整 Vs 核纯 CPU 加速约 1.81× / 1.59×，CUDA 相对 disba CPU 的独立复测为 27.4× / 41.1×；
当前纯频散没有超过 disba 默认 CPU 正演。

本版还提供 `JointRFForward`、`ShallowCrustModel`、`JointRFProblem`、`hmc_sample`、`dram_sample` 和诊断接口。
0.6.0.dev2 的 [CPU RF 说明](DIFFERENTIABLE_RF_CPU.zh-CN.md)保留为历史文档；本版整合的 RF 接口同时支持 CPU 和原生 CUDA。
验证范围、二进制审计及实机检查见 [本版验证记录](../verification/preview-0.6.0.dev3.json)。
