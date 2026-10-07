# Apple Metal 与 MPS

[English](APPLE_METAL.md) | 简体中文 | [首页](../README.zh-CN.md)

Apple Silicon 上执行 `python -m pip install seismicx-grtm` 即包含 CPU 和预编译
Metal 库，使用时不需要 Xcode 或着色器编译器。已实测 M4 Max。

```python
import grtm
solver = grtm.Solver(backend="mps")  # metal、apple 为同义词
model = grtm.example_model(units="si")
model.update(log2_samples=6, dt=0.2, distances=[10000., 30000.], t0=[0., 0.])
result = solver.forward(model, sdr=(20, 40, 60), scalar_moment=1e15, threads=4)
print(solver.backend_stats())
```

这是 CPU/GPU 混合引擎：层间传播和敏感运算在 CPU 上使用 FP64，接收点积分在
Metal 上使用补偿浮点对运算，约 48 位有效尾数，不能等同于 IEEE FP64。
低频稳定化、PTAM、FFT 和结构导数仍在主机执行。
所有频率均走主机稳定化时，`dispatches=0` 可以是正常结果。

`backend_stats()` 中 GPU 时间不含 CPU 部分，性能应计整个接口调用。
多接收点可能更适合 Metal，较小任务可能 CPU 更快；详见[FK 对比](../benchmarks/fk/README.zh-CN.md)。
输出内存仍随样本、接收点和物理场数增长，大量模型优先使用迭代器。

PyTorch 的 `device="mps"` 与求解器的 `backend="mps"` 是两项独立设置。
MPS 张量使用 float32/complex64，内部主机物理计算仍为 FP64，结构反向为 CPU VJP。
完整可运行的 MPS 张量示例见[英文页面](APPLE_METAL.md#pytorch-mps-tensors)。
JAX 可通过主机回调调用 `backend="metal"`，不要求 JAX Metal 插件；
JAX 导数会构造稠密 Jacobian，不具备 PyTorch 流式 VJP 的内存特性。

只支持一阶导数。精度与极长周期限制见[数值说明](NUMERICAL_NOTES.zh-CN.md)。
