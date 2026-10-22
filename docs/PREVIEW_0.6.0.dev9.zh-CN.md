# 格林函数显式波场积分核预览 — 0.6.0.dev9

[二进制下载](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev9) · [理论与接口](GREEN_WAVEFIELD_KERNELS.zh-CN.md) · [速度与精度](PERFORMANCE_0.6.0.dev9.zh-CN.md)

| Python | 平台 | 后端 |
|---|---|---|
| CPython 3.12 | Apple Silicon macOS 11+ | CPU、Metal；新增显式格林函数核运行在 CPU |
| CPython 3.13 | Linux/WSL x86_64，glibc ≥ 2.31 | CPU、CUDA sm120；RTX 5090 实测 |

下载匹配 wheel，执行 `python -m pip install /path/to/wheel.whl`。张量接口另装匹配的 PyTorch。本次只发布以上两个 ABI，其余已有 ABI 保留 dev7，稳定版 PyPI 不变；不能修改 wheel 标签强行安装。

在 `grtm.torch.Forward` 或 `ForwardOperator` 中设置 `vjp_method='wavefield'` 即可使用精确层内波场积分。原有 `adjoint` 和默认选择保留。CPU/CUDA 使用 FP64；近静态频率仍采用原有扩展精度 AD，重合源接收位置、极薄子层或病态计算保留已有安全回退，执行统计会报告。本版本没有宣称 MPS 原生 FP64 波场核或跨模型融合的 CUDA batch。

两个 wheel 的源码版本均为 `f3d8106fc059ab75ba5580fbaa3fadcaeb908917`。Linux 通过 auditwheel 修复为 glibc 2.31。Mac 的全部本机 host 库及 Python 扩展重新编译，仅复用从 SHA256 校验过的 dev7 wheel 提取的未改变编译 Metal shader，同时核对 shader 源码及二进制哈希；不嵌入 shader 源码。这与 dev8 复用整个 Metal 动态库不同，来源写入构建和验证记录。

修复了已复现的 CUDA autograd 工作线程退出崩溃，并加入子进程回归测试。研究/非商业许可保持不变。community 发布二进制、文档及数值证据，完整求解器源码保留在完整仓库。
