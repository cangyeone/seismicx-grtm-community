# 计算设备与微分二进制预览 — 0.6.0.dev5

[English](PREVIEW_0.6.0.dev5.md) | 简体中文

下载[匹配的 wheel](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev5)，
安装实际文件名（下方标签只是占位符）：

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev5-<matching-tags>.whl
python -m pip install numpy scipy
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

支持 CPython 3.10–3.14，Apple Silicon macOS 11+（CPU/Metal），
Linux/WSL x86_64 glibc≥2.28（CPU/CUDA）。CUDA 目标为计算能力 12.0，已在 RTX 5090
验证。安装二进制不需要编译器/nvcc。Torch 适配器、RF 和 CUDA 面波需要另装匹配
Python/GPU 的 Torch；RTX 5090 本次使用 Torch 2.8.0+cu128。

新增共享 CPU RF 正演/VJP、CUDA Green 解析结构微分积分/收缩、批量 backward、
CUDA 缓冲区复用及 RF VJP 临时量压缩。半解析求导不扰动模型；面波 FP64 和默认
1e-10 km/s 容差不变。RF 和面波 CUDA 正演/微分在 GPU。Green 低频扩展精度、FFT
等仍有主机边界；MPS RF/面波是显式 CPU FP64 桥接，并非原生 GPU 加速。

见[简易接口与设备说明](DEVICE_EXECUTION.zh-CN.md)、[速度/精度测试](PERFORMANCE_0.6.0.dev5.zh-CN.md)。
CPU 为默认后端。CUDA 错误不静默回退。自定义 backward 只支持一阶微分。
CPU/Metal 与 CUDA 的 Green VJP 算法不同，后者目前按所选参数方向积分，不能保证
比 CPU 反向 AD 更快。

研究/非商业许可证保持不变，community 不包含求解器实现源码。稳定 PyPI 仍为
0.5.1；本次未上传 PyPI。下载资产附带 SHA-256、wheel 清单与验证记录。
