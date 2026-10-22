# 0.6.0.dev12 二进制预览

从[公开预发布](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev12)下载与 Python/平台匹配的 wheel：

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev12-*.whl
```

- CPython 3.12，Apple Silicon macOS 11+：CPU + Metal。频散图及其伴随使用 CPU FP64；MPS 明确桥接 CPU。
- CPython 3.13，manylinux_2_31 x86_64：CPU + CUDA 12.8，架构 120，实测 RTX 5090；另需兼容的 PyTorch 和 NVIDIA 驱动。
- 梯度和有限阵列处理需要 PyTorch，包括 NumPy 输入的 CPU 阵列调用；direct NumPy 正演仅需 NumPy。

新增[能量图解析伴随与有限阵列](DISPERSION_IMAGE.zh-CN.md)：direct、主动线源 FK、垂直点源 F–J、单独定义的各向同性随机源台站对互谱 F–J。支持模型、复震源、source PSD 的一阶导数和 batch；CUDA 物理计算、阵列变换及反传在同一设备完成。`image_sensitivity` 输出单目标 VJP/最小二乘梯度，不生成整个图像 Jacobian。[精度与耗时报告](PERFORMANCE_0.6.0.dev12.zh-CN.md)。

当前为各向同性弹性固体层、地表震源/接收点，不含 VTI、Q 衰减、流体、地下源、高阶导数，也不假定任意环境噪声都符合本次随机源模型。空间积分有限带宽，需要检查采样与端点收敛。用户提供的幻灯片用于图形组织参考，没有声称复现其 SPECFEM/SASEM 数值实验。

科研/非商业许可证保持不变。公开仓库仅发布编译后的实现、文档和数值证据，不包含求解器源码；本次 GitHub 预览不改变稳定 PyPI 版本。
