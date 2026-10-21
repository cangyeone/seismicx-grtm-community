# 解析伴随与 CUDA 反向 AD 预览版 — 0.6.0.dev7

[Release / 二进制下载](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev7)

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev7-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

标签是占位符，需使用实际匹配文件名。支持 CPython 3.10–3.14；Apple Silicon macOS 11+（CPU/Metal），Linux/WSL x86_64 glibc≥2.28（CPU/CUDA）。CUDA 目标计算能力 12.0，实测 RTX 5090、Torch 2.8.0+cu128。安装不需编译器；张量接口及 CUDA 频散另需匹配 Python/GPU 的 Torch。

频散使用 GRTM 特征方程伴随，群速度核使用解析混合导数；RF 使用原生材料/矩阵伴随；Green 新增真正 CUDA 反向 AD，并将 Torch 频谱、物理合成、FFT 与源梯度保留在 GPU。低频病态频点仍使用主机扩展精度。CPU Levin 可联合输出物理波形与解析 JVP。没有将 RF FFT 或频散求根称作 Levin。

[Methods / 公式与接口](ANALYTIC_METHODS.zh-CN.md) · [Performance / 实测](PERFORMANCE_0.6.0.dev7.zh-CN.md) · [Devices](DEVICE_EXECUTION.zh-CN.md)

小任务并非全部更快；报告包含变慢情况。自适应 Levin 在 CPU；MPS RF/面波仍是 CPU FP64 桥接。许可保持研究/非商业限制，社区不分发求解器实现源码。仅发布 GitHub 预览，稳定 PyPI 不变。
