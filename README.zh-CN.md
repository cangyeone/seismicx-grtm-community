# SeismicX GRTM 用户社区

[English](README.md) | 简体中文

这里提供 **SeismicX GRTM 的详细使用说明、二进制下载和性能对比结果**。
程序用于层状介质格林函数计算，支持 CPU、CUDA 和 Apple Metal/MPS。
仓库中的代码片段是公开接口的使用示例，不包含求解器实现源码或开发历史。

## 0.6.0.dev1 研究预览版

新增方向 JVP、原生反向 VJP、JAX 无完整 Jacobian 的反向模式，以及独立的
Gauss/Levin/带保护 Shanks 波数积分。安装及详细示例见
[预览版指南](docs/RESEARCH_PREVIEW.zh-CN.md) · [English](docs/RESEARCH_PREVIEW.md)。
从 [GitHub 预览版](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev1)
下载与你的 Python/平台匹配的 wheel；**PyPI 稳定版仍为 0.5.1**。

结构导数和研究积分在 CPU 执行；自适应积分尚无框架 backward，也不认证完整原生积分误差。
新的功能与固定网格导数的边界在指南中单独说明。旧 FK 数据仍对应原冻结实验。
[预览版校验和](downloads/SHA256SUMS-0.6.0.dev1)与
[验证记录](verification/preview-0.6.0.dev1.md)单独保存。
发布包不包含求解器实现、论文源文件或未公开实验归档。

## 安装

```sh
python -m pip install seismicx-grtm
python -m pip install 'seismicx-grtm[rf]'         # 接收函数反演依赖
python -m pip install 'seismicx-grtm[torch,jax]'  # 可选框架接口
```

当前版本 **0.5.1**，导入方式为 `import grtm`，命令行入口为 `grtm`。
支持 Python 3.10–3.14：Apple Silicon macOS 11+ 的 wheel 包含 CPU + Metal，
Linux/WSL x86_64、glibc 2.28+ 的 wheel 包含 CPU + CUDA。
CUDA 仅针对计算能力 12.0，已验证 RTX 5090；CPU 使用不需要 NVIDIA 显卡。
安装不需要编译器或 CUDA toolkit。详见[安装说明](docs/INSTALLATION.zh-CN.md)。

## 快速开始

```python
import grtm

model = grtm.example_model(units="si")
model.update(log2_samples=6, dt=0.2, distances=[10000., 30000.], t0=[0., 0.])
solver = grtm.Solver(backend="cpu")  # 也可选 cuda、metal、mps
result = solver.forward(
    model, sdr=(20, 40, 60), scalar_moment=1e15, azimuth=45,
    fields=("displacement", "strain", "stress"), threads=4,
)
print(result["displacement"].shape)  # (2, 64, 3)
```

高层接口默认 **NED（北、东、下）坐标和 SI 单位**。
未指定震源时间历史时，输出是脉冲格林函数核，单位分别为 m/s、1/s、Pa/s；
加入无量纲震源历史并卷积后才是 m、应变和 Pa。
上例的短时间窗用于演示调用，科学计算需另行检查时间窗和积分收敛。

## 使用文档

- [安装、后端选择与离线下载](docs/INSTALLATION.zh-CN.md)
- [完整入门教程：模型、输出、命令行](docs/QUICKSTART.md)（英文）
- [矩张量/SDR、应力应变、批处理、结构导数、PyTorch/JAX](docs/PYTHON_FORWARD.zh-CN.md)
- [Apple Metal / MPS](docs/APPLE_METAL.zh-CN.md)
- [H–κ → 接收函数全波形拟合 → 结构反演](docs/RECEIVER_FUNCTIONS.zh-CN.md)
- [精度、物理约定与适用范围](docs/NUMERICAL_NOTES.zh-CN.md)
- [常见问题](docs/TROUBLESHOOTING.md)（英文）

点源结构导数使用 CPU 切线计算；选择 CUDA/Metal 正演不意味着结构反向计算在 GPU 上。
JAX 默认导数回调构造稠密 Jacobian，0.6 预览版另提供反向 VJP 模式；
PyTorch 使用 VJP，预览版可显式选择原生反向路径。
接收函数是独立的弹性平面波求解器，在 NumPy/CPU 上运行，结构反演使用差分导数。

## FK 对比和下载

[FK 对比说明](benchmarks/fk/README.zh-CN.md)包括四种设备、单线程基线、
我们添加的 FK OpenMP 多线程适配，以及层数、震源深度、震中距和接收点数的变化。
公开了完整计时表、750 次计时和 300 次预热数据；远程共享负载和慢于 FK 的案例均保留。
这些测量来自锁定的 0.5.0 科学计算版本，不能当作 0.5.1 wheel 的重新测速。

![FK 对比综合图](benchmarks/figures/performance-summary.png)

[二进制下载](downloads/README.md) ·
[v0.5.1 Release](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.5.1) ·
[PyPI](https://pypi.org/project/seismicx-grtm/) · [校验和](downloads/SHA256SUMS)

GitHub 自动显示的 “Source code” ZIP/TAR 只打包此文档和数据仓库。
安装程序请选择明确的 `.whl` 附件；求解器实现源码不在这个仓库内。

## 开发人员

- **Weiping Wang**
- **Xin Liu:** [xinliu_geo@outlook.com](mailto:xinliu_geo@outlook.com)
- **Yuqi Cai:** [caiyuqiming@foxmail.com](mailto:caiyuqiming@foxmail.com)
- **Ziye Yu:** [yuziye@hotmail.com](mailto:yuziye@hotmail.com)

使用问题与错误反馈请提交 [Issue](https://github.com/cangyeone/seismicx-grtm-community/issues)，
附版本、系统、Python、后端和最小模型。

## 许可

采用[闭源、仅非商业科研许可](LICENSE)，商业使用需要另行书面授权。
第三方条款见 [NOTICE.md](NOTICE.md)。公开文档不代表求解器开源。
FK 和外部 Fortran GRTM 分别来自 [Lupei Zhu 的 FK](https://github.com/rwalkerlewis/fk)
与 [Yunyi Qian 的 GRTM](https://github.com/YunyiQian/grtm)，不作为本项目作者的作品发布。
