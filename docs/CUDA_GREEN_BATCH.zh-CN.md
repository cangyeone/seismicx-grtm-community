# 格林函数跨模型 CUDA 融合（0.6.0.dev10）

[English](CUDA_GREEN_BATCH.md) · [实测结果](PERFORMANCE_0.6.0.dev10.zh-CN.md)

`grtm.torch.Forward(..., vjp_method="wavefield")` 现在把模型、频率、波数放进
共同的 CUDA launch，层积分收缩再沿层并行。辐射模式、坐标转换、应力、震源时间函数、
FFT 及其 Torch 微分均对整个 batch 在调用方 GPU 上执行。
这里计算标量损失对全部所选参数的 VJP，不构造完整波形 Jacobian。

```python
import numpy as np
import torch, grtm
from grtm.torch import Forward

model = grtm.example_model()
model.update(log2_samples=7, dt=0.4, ptam=0)
layers = torch.tensor(np.repeat(np.asarray(model["layers"])[None], 16, axis=0),
                      dtype=torch.float64, device="cuda:0", requires_grad=True)
moment = torch.tensor(grtm.strike_dip_rake(20, 40, 60, 1.0),
                      dtype=layers.dtype, device=layers.device)
forward = Forward(model, units="native", backend="cuda", threads=4,
                  vjp_method="wavefield", cuda_batch=True,
                  cuda_precision="fast",
                  grid={"steps": [512] * 64, "period_length": 500.0})
waveforms = forward(layers, moment)
gradient, = torch.autograd.grad(waveforms.square().sum(), layers)
print(forward.execution_stats())
```

wavefield tensor 路径默认 `cuda_batch=True`。支持 `[L,6]` 或 `[B,L,6]` 模型，
以及共享 `[6]` 或逐模型 `[B,6]` 矩张量。batch 内原始层数相同，但层深、材料、
震源/接收点所在层及展开后的层数可以不同。采样、接收点、源深、输出物理量与单位等
配置共用。输入张量的设备和 dtype 必须一致。快速模式内部仍保持 FP64/complex128。
`cuda_batch=False` 可用于逐模型对照。本次不改变 CPU、MPS、接收函数和频散算法。

## 精度策略与设备边界

默认 `cuda_precision="strict"` 保留原来低频的 CPU 扩展精度保护，只开启融合。
`"fast"` 需显式选择，要求 analytic、PTAM=0、CUDA 输入与 wavefield 路径。
通过验收的低频也在 GPU 上使用同一套 FP64 显式波场积分，极端消减情形
保留原保护。几何特殊或显式积分出现非有限值时，回退到 GPU 原反向 AD。

令 `r_f = |omega_f| / (Vp_max * K_f * dk)`，其中复频率包含虚频阻尼。
strict 保护 `r_f < 0.005` 的频点。fast 模式下，只要某模型存在
`r_f < 0.0005` 的频点，就保留该模型的整套原保护策略；其余模型保护
DC 的 `r_f < 0.0005` 和非零频率的 `r_f < 0.002`。
新放行且低于 `0.005` 的频点使用同一套显式波场积分。这是经过案例验证的条件数保护，
**不是任意模型的误差保证**。验收采用波形相对误差 ≤1e-4、模型/震源梯度 ≤1e-3，
并单独检查每一列结构参数梯度；具体覆盖范围见实测报告。

极长时间窗继续保留保护，因为逐频误差小，合成后的 VJP 误差仍可能超标。
极薄子层、源/接收点重合以及显式积分条件不良时，按模型回退到原反向计算约定，
因此这些情形的速度收益会减小。

正常频率的数值数组、GPU 低频核、物理合成和 FFT 都在所选 GPU 上。
模型校验、拓扑、网格规划和少量统计信息在主机上，每次正演复制一次模型 batch
快照。如果仍有 CPU 保护频点，对应频谱和梯度需要跨设备传输，反传目前为它们
一次性下载整个 batch 的 cotangent。因此只有 `cpu_frequencies == 0` 才表示
数值计算全部在 GPU 上，并不表示没有主机 API 开销。

`execution_stats()` 在 autograd 工作线程反传后读取，融合路径的计数覆盖
**整个 batch**，包含模型数、CPU/CUDA 频点数、GPU 低频数、保留严格保护的模型数、
反向 AD 频点数和回退模型数。旧串行路径只报告最近一个模型。
plan 只持有主机元数据；GPU 工作区按工作线程/设备缓存，并限制各计算阶段的预算。
梯度按固定顺序归约，不使用跨模型原子加法。回归测试覆盖非默认 stream 和保留
计算图后重复反传，仍不支持二阶自动微分。
