# 跨模型 CUDA 格林函数融合 — 0.6.0.dev10

[二进制发布](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev10) · [接口与精度策略](CUDA_GREEN_BATCH.zh-CN.md) · [实测](PERFORMANCE_0.6.0.dev10.zh-CN.md)

| Python | 平台 | 后端 |
|---|---|---|
| CPython 3.12 | Apple Silicon macOS 11+ | CPU、Metal；显式格林函数微分在 CPU |
| CPython 3.13 | Linux/WSL x86_64，glibc ≥2.31 | CPU、CUDA sm120，RTX 5090 实测 |

下载对应 wheel，执行 `python -m pip install /path/to/wheel.whl`。Torch 需另行安装适配设备的版本。不要修改 wheel 标签；本次只发布上述两个 ABI，稳定 PyPI 版本不变。

CUDA `grtm.torch.Forward(..., vjp_method="wavefield")` 默认融合模型 batch。默认 `cuda_precision="strict"` 保留原低频精度保护；显式选择 `cuda_precision="fast"` 后，符合条件的低频正演和显式梯度留在 GPU。极低频模型仍保留 CPU 保护，特殊几何可能回退到原反向 AD。模型校验与拓扑规划仍在主机。查看 `execution_stats()`，融合计数覆盖整个 batch。

两份 wheel 的求解器、Python 扩展来自源码提交 `a130cc274d70239130a25f0c0bf8936682378c54`。Linux 经 auditwheel 修复为 glibc 2.31。Mac 主机库与 Python 扩展重新构建；仅复用与源码哈希一致的 dev7 编译 Metal shader，记录参考 wheel、shader 源码与二进制哈希，不附带 shader 源码。

CPU、MPS、RF 和频散算法没有新增修改或提速声明。面波文档新增训练/推理一致性、缺模掩码、固定归一化、跨模型族验证和分块存储约定。本次不发布训练集或网络。

许可仍为研究/非商业使用。完整源代码在项目源码仓库；community 只发布编译实现、文档与数值验证材料。哈希、测试日志和包装审计见 [验证目录](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/preview-0.6.0.dev10)。
