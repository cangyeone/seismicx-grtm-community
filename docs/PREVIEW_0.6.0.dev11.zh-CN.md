# 0.6.0.dev11 二进制预览

从[公开预览发布页](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev11)下载匹配 Python 和平台的 wheel：

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev11-*.whl
```

- CPython 3.12、Apple Silicon macOS 11+：CPU + Metal 包；频散能量图使用 CPU FP64，MPS 为明确的 CPU 桥接。
- CPython 3.13、manylinux_2_31 x86_64：CPU + CUDA 12.8，计算能力 12.0，在 RTX 5090 验证。CUDA 调用另需兼容的 PyTorch 和 NVIDIA 驱动。

新增[地表力频散能量图](DISPERSION_IMAGE.zh-CN.md)，返回垂直、径向或横向载荷下的复位移、原始和归一化幅值平方；支持 CPU/CUDA、融合模型 batch。[精度与速度](PERFORMANCE_0.6.0.dev11.zh-CN.md)。

新能量图接口本次仅正演；原有频散微分、Green/RF 接口及 dev10 CUDA Green 加速保留。适用弹性固体层，不包含地下震源或介质 Q 衰减。科研/非商业许可不变，本次 GitHub 预览不改变稳定 PyPI 版本。公开仓库只提供编译实现、文档和数值证据，不分发求解器实现源码。
