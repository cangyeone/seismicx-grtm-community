# 安装与后端选择

[English](INSTALLATION.md) | 简体中文 | [首页](../README.zh-CN.md)

使用 Python 3.10–3.14 的虚拟环境：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install seismicx-grtm==0.5.1
python -m pip check
python -m grtm --info
```

发行名为 `seismicx-grtm`，导入名为 `grtm`。不要通过安装同名的其他 `grtm` 包获取本程序。
支持 Apple Silicon macOS 11+ 和 Linux/WSL x86_64、glibc 2.28+。
不提供原生 Windows、Intel Mac、Linux ARM、PyPy 或自由线程 CPython wheel。
例如 `cp312-cp312` 文件只对应 CPython 3.12；pip 会自动选择适配文件。

可选依赖：

```sh
python -m pip install 'seismicx-grtm[rf]==0.5.1'
python -m pip install 'seismicx-grtm[torch,jax]==0.5.1'
```

NumPy 是基本依赖，`rf` 添加反演所需 SciPy；合成与 H–κ 叠加不要求 SciPy。
PyTorch/JAX GPU 版另有自身的驱动和运行时要求。

```python
import grtm
print(grtm.available_backends())
solver = grtm.Solver(backend="cpu")
# solver = grtm.Solver(backend="cuda")
# solver = grtm.Solver(backend="mps")
```

默认 CPU，`c` 为别名。Apple wheel 同时包含 Metal，`metal`、`mps`、`apple` 是同一引擎。
Linux wheel 同时包含 CPU 和 CUDA，不需要替换两个不同的发行包。
`available_backends()` 只说明包内有该引擎，不保证当前设备可执行；失败会报错，不会静默回退。

CUDA wheel 包含静态链接的 CUDA 12.8 运行时，安装不需要 toolkit 或 nvcc。
执行 CUDA 要求兼容驱动和**计算能力 12.0**，已测 RTX 5090；旧架构请用 CPU。
WSL 需要 Windows 主机驱动提供 GPU 访问。CPU 使用不需要 NVIDIA 显卡或驱动。
Apple wheel 内含预编译 Metal 库，安装使用不需要 Xcode 或 shader 编译器；已测 M4 Max。

离线下载见[文件列表](../downloads/README.md)，与 PyPI 文件完全相同。
请核对 [SHA256SUMS](../downloads/SHA256SUMS)。离线机器还需 NumPy 和所选可选依赖；
可在相同系统/Python 的联网机器上用 `pip download --only-binary=:all: --dest wheelhouse` 准备依赖。

GitHub 自动的 “Source code” ZIP/TAR 只有此仓库的文档和数据；安装请选择 `.whl`。
没有源码自动构建回退。若之前安装过 `grtm-green`，先卸载旧发行包，避免同一导入目录冲突。

## 研究预览版

[0.6.0.dev1 安装与接口指南](RESEARCH_PREVIEW.zh-CN.md)。该 GitHub 预览版
与上面的 PyPI 0.5.1 稳定版分开安装。
