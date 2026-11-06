# 0.6.0.dev15：频谱白化编译预览版

[English](PREVIEW_0.6.0.dev15.md) · [编译包下载](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev15) · [校验和](../downloads/SHA256SUMS-0.6.0.dev15) · [验证记录](../verification/preview-0.6.0.dev15/verification.json)

新增波形接口 `grtm.processing.spectral_whiten` 和单边复频谱接口
`whiten_spectrum`，也可从 `grtm` 直接导入。支持平滑谱白化
（`smooth` / `rma`）和严格白化（`strict` / `brutal` / `phase_only`）。
详见[数据处理文档](DATA_PROCESSING.zh-CN.md)与[可运行示例](../examples/spectral_whitening.py)。
本包同时包含当前 array/DAS 和稳健面波求根实现。

## 安装匹配的编译包

从上方 Release 下载一个 `.whl`，在 CPython 3.13 环境中安装：

```sh
python -m pip install /path/to/seismicx_grtm-0.6.0.dev15-<matching-tags>.whl
python -c 'import grtm; print(grtm.__version__)'
```

将占位符替换为实际文件名。本次仅发布以下两个版本：

| CPython | 平台 | 包含的原生引擎 |
|---|---|---|
| 3.13 | Apple Silicon，macOS 11+ | CPU、Metal |
| 3.13 | Linux/WSL x86_64，glibc 2.31+ | CPU、CUDA 12.8，SM120 |

CUDA 求解器要求兼容的 NVIDIA 驱动及计算能力 12.0，已测 RTX 5090。
CPU 使用不需要 GPU。本次不提供其他 Python ABI、Intel Mac、Linux ARM 或原生 Windows 包。
张量输入额外需要适合目标 CPU/CUDA 设备的 PyTorch；NumPy 输入仅需 NumPy。

PyPI 上传尚待发布认证完成；在确认前请使用上述 GitHub wheel。
PyPI 稳定版 0.5.1 不包含白化功能。

## 设备与数值边界

- NumPy 路径在 CPU；PyTorch float32/float64 的白化及梯度保留在输入的 CPU/CUDA
  设备上。白化暂不支持 MPS 或半精度。
- 默认 `check_finite=True` 在 CUDA 上有一次输入校验的标量同步。输入已确认有限时，
  可设 `check_finite=False` 去除这次同步。
- 平滑白化除以居中的幅度谱滑动均值，端点重复扩展；严格白化将有效频带内通过阈值的
  非零频点幅度设为 1。`water_level=0` 对所有非零频点生效；零频点及 DC 保持为零。
  可选余弦过渡带位于平坦通带外。
- 这是白化算子，不是完整的 Bensen、NoisePy 或 MSNoise 处理流程；不会自动去趋势、
  填补缺测、归一化 RMS 或补零。

## 编译分发与验证

34 个实现模块（含数据处理和稳健求根辅助模块）均编译为扩展。
仅保留 3 个最小导入/命令行入口 `.py`；不发布 `.pyx`、C/C++/CUDA/Metal 源文件、
头文件或源码分发包。文档、使用示例、模型数据与许可证说明保留为可读文件。
编译不能阻止逆向工程，现有科研/非商业许可不变。

验证记录列出安装包哈希、源码泄漏检查、安装后的测试结果及设备/运行时版本。
Mac 的 host 库全部重编译；仅在核对 shader 源文件及 dev12 发布包哈希后复用未变化的
Metal 编译 shader。旧版性能数字仍属于历史测量，本次不声明统一加速比。
