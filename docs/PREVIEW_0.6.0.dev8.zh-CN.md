# 特征函数与受迫响应核预览版 — 0.6.0.dev8

[下载二进制](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev8) · [公式与接口](VARIATIONAL_KERNELS.zh-CN.md)

本次发布两个实际构建、安装并测试的包：

| Python | 平台 | 后端 |
|---|---|---|
| CPython 3.12 | Apple Silicon macOS 11+ | CPU、Metal；RF/面波 MPS 为 CPU FP64 桥接 |
| CPython 3.13 | Linux/WSL x86_64、glibc ≥ 2.31 | CPU、CUDA，计算能力 12.0，实测 RTX 5090 |

```sh
python -m pip install /path/to/<matching-wheel>.whl
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

张量接口及 CUDA 面波需另装与 Python/GPU 匹配的 PyTorch。其他 Python ABI 的 dev8 包尚未发布，不能更改文件名冒充兼容；需要旧 ABI 时保留 dev7。稳定 PyPI 不变。

本次 GitHub Actions 任务在构建步骤开始前结束，无法据此声称 CI 构建通过。使用本机和远程 CUDA 主机构建替代，Linux 包经 auditwheel 修复和审计。Mac 的 CPU/RF/面波及 Python 扩展重新编译；未改动的 Metal 后端复用已校验 SHA256 的 dev7 预编译组件，相关源文件无差异，组件来源记录在包内 `build-info.json`。不嵌入 Metal 源码。

新增 `differentiation='eigenfunction'` 选项和 RF/Green `sensitivity` 标量目标核接口。默认面波方法仍为 `adjoint`。解析核不代表任意模型的根完整性保证，也不证明反演一定更准确。FP64、同设备解析保护、低频扩展精度、MPS 桥接和一阶导数限制见方法文档。

许可保持研究/非商业限制；社区只发布编译实现、文档和数值验证记录，不分发求解器实现源码。
