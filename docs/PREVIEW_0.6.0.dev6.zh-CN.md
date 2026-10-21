# 相速度、群速度与振荡积分预览版 — 0.6.0.dev6

[Release / 二进制下载](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev6)

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev6-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

上面标签是占位符，请使用对应实际文件名。支持 CPython 3.10–3.14；Apple Silicon macOS 11+（CPU/Metal），Linux/WSL x86_64 glibc≥2.28（CPU/CUDA）。CUDA 编译目标为计算能力 12.0，在 RTX 5090、Torch 2.8.0+cu128 验证。安装无需编译器；框架接口与 CUDA 频散另需适配 Python/GPU 的 Torch。

新增 CUDA 相/群速度寻根优化；保持 FP64、模态检查和根容差。另有 CPU 批量 Levin、正演/解析 JVP 共享积分、`method="auto"` Gauss/Levin 选择。自适应波数积分在 CPU；面波 CUDA 正演和微分仍在 GPU。其他设备边界与 dev5 相同，Green FFT 仍在主机，MPS RF/面波仍为显式 CPU FP64 桥接。

[Surface-wave report](PERFORMANCE_0.6.0.dev6.zh-CN.md) · [Quadrature / 积分报告](QUADRATURE_0.6.0.dev6.zh-CN.md) · [Device boundaries](DEVICE_EXECUTION.zh-CN.md)

精度范围和失败记录见报告。未宣称任意模型的严格无穷尾项保证，亦未证明新积分在同误差下全面快于 PTAM。许可保持研究/非商业限制，社区不分发求解器源码。预览只在 GitHub 发布；稳定 PyPI 未更新。
