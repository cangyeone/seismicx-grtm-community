# 0.6.0.dev16：完整 CPU/CUDA 噪声互相关

[English](PREVIEW_0.6.0.dev16.md) · [GitHub 编译包](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev16) · [PyPI](https://pypi.org/project/seismicx-grtm/0.6.0.dev16/)

新增 `grtm.noise_cross_correlation`：重叠分窗、去趋势、时间归一化、频带过滤/白化、
补零线性互相关、加权均值/求和或相位加权叠加、逐台站对缺测掩码与张量梯度。
各台站 FFT 在台站对间复用；线性叠加先归约互谱，最后才反 FFT。

```sh
python -m pip install --only-binary=:all: seismicx-grtm==0.6.0.dev16
```

本次提供 **CPython 3.13** 的 Apple Silicon macOS 11+（原生 CPU/Metal）和 Linux/WSL
x86_64、glibc 2.31+（原生 CPU/CUDA 12.8 SM120）两个编译包。原生求解器 CUDA 引擎要求
兼容驱动及计算能力 12.0，已测 RTX 5090；噪声处理使用 NumPy CPU 或已安装的 PyTorch CPU/CUDA。
NumPy 输入不要求 PyTorch，张量输入需另装适合设备的 PyTorch。噪声处理暂不支持 MPS、半精度；
本预览版不提供其他 Python/平台 ABI。

[完整接口与数值约定](NOISE_CORRELATION.zh-CN.md) · [示例](../examples/noise_cross_correlation.py) · [CPU/CUDA 性能](PERFORMANCE_0.6.0.dev16.zh-CN.md) · [验证](../verification/preview-0.6.0.dev16/verification.json) · [校验和](../downloads/SHA256SUMS-0.6.0.dev16)

35 个实现模块均为编译扩展，仅保留 3 个最小导入/命令行入口 Python 文件；不分发实现源码、
头文件或 sdist。科研/非商业许可不变。Mac host 库已重编译，仅复用已核验且源代码未变化的
Metal shader，记录源文件及 dev12 二进制哈希。包内原生格林函数 Metal 引擎与噪声处理的设备
支持不同；互相关仍限 CPU/CUDA。稳定 PyPI 0.5.1 不含该接口，请显式指定 dev16。
