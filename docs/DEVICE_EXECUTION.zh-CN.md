# 计算设备与批量接口 — 0.6.0.dev7

dev10 新增[格林函数跨模型 CUDA 融合与显式快速精度策略](CUDA_GREEN_BATCH.zh-CN.md)。下表保留 dev7 基线；RF、频散和 MPS 的设备边界未改变。

[English](DEVICE_EXECUTION.md) | 简体中文

正演和结构微分优先在选定的同一设备执行。RF、面波的 CUDA 路径将数值传播、寻根、
敏感核和反向计算保留在 CUDA；格林函数现在也有原生 CUDA 结构微分积分，只有下述
精度保护等环节仍需主机参与。CUDA 错误直接报告，不会悄悄切换成 CPU 求解器。

| 类型 | CPU | CUDA | Apple / MPS |
|---|---|---|---|
| 点源格林函数正演 | 原生 FP64，低频 long double | GPU FP64 积分；Torch CUDA 的 FFT/物理合成在 GPU，低频保护在主机 | 现有混合算法：主机 FP64 传播，Metal 补偿 float-pair 接收端积分 |
| 格林函数结构微分 | 原生切线 / 节点局部反向 AD | 原生解析切线与真正反向 AD；低频主机保护 | CPU 结构 AD |
| 联合 RF 正演与 VJP | 共享原生 complex128 准备、传播及材料伴随，Torch FFT | 原生 complex128 准备、传播、材料伴随及 Torch FFT 都在 CUDA | 显式 CPU FP64 桥接；实数输出 MPS float32，复频谱留在 CPU complex128 |
| 面波相/群速度与敏感核 | 原生 FP64 特征方程伴随及 dual/hyper-dual | CUDA 原生 FP64 特征方程伴随及 dual/hyper-dual、搜索和寻根 | 显式 CPU FP64 桥接；输出 MPS float32 |

MPS 的 RF/面波支持是兼容接口，**不代表原生 GPU 加速**。选用时会警告，并标注
`backend='mps-cpu-fp64'` 和 CPU 计算设备。MPS 输入的 float32 舍入和输出转换仍然存在；
内部 FP64 不能恢复输入已丢失的精度。需要保留双精度时应选择 CPU/CUDA float64。
所有原生自定义 backward 只支持一阶微分。

## 最简正演和梯度调用

安装 [dev7 二进制预览](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev7)、NumPy 和适配设备的 Torch。
RF/面波四列为层顶深度 km、Vp/Vs km/s、密度 g/cm³；Green `Forward` 使用 SI，
六列还包含 Qp/Qs。

```python
import numpy as np
import torch
import grtm
from grtm.torch import Forward

device = 'cuda:0'  # 'cpu' or 'mps'
dtype = torch.float32 if device == 'mps' else torch.float64
one = [[0.,3.6,2.,2.5], [2.,6.5,3.6,2.8], [35.,8.,4.5,3.3]]
earth = torch.tensor(one, dtype=dtype, device=device)[None].repeat(16,1,1)
earth.requires_grad_()
rf = grtm.JointRFForward([.06], device=device, kernel='native', threads=4)
pred = rf(earth)
g_rf = torch.autograd.grad(pred['rf'].square().mean(), earth)[0]
sw = grtm.SurfaceWave(earth, threads=1 if device.startswith('cuda') else 4)
c = sw.phase_velocity([5.,10.,20.])
g_sw = torch.autograd.grad(c.sum(), earth)[0]
k = sw.sensitivity([5.,10.,20.], kind='group', parameter='velocity_s')['kernel']

model = grtm.example_model(units='si')
model.update(log2_samples=6, dt=.4, distances=[10000.,30000.], t0=[0.,0.], ptam=0)
backend = 'cuda' if device.startswith('cuda') else 'metal' if device == 'mps' else 'cpu'
f = Forward(model, backend=backend, device=0, parameters=['vp','vs'],
            workers=4, threads=4, vjp_method='auto')
x = torch.tensor(np.asarray(model['layers']), dtype=dtype, device=device)
x = x[None].repeat(16,1,1).requires_grad_()
mt = torch.tensor(grtm.strike_dip_rake(20,40,60,1e15), dtype=dtype, device=device)
y = f(x, mt)
g_green = torch.autograd.grad(y.square().mean(), x)[0]
print(pred['rf'].shape, c.shape, k.shape, y.shape)
```

例子仅对 Green 的 Vp/Vs 求导，其余列返回零。RF/面波的输出包含模型批量和观测维度。
Green 构造器的 `device=0` 是原生 CUDA 卡号，应与张量设备一致。MPS 的 RF/面波
会按设计警告并使用主机计算。

## 同设备计算的具体边界

Green CUDA JVP 沿原 GRTM 方程传播解析导数。`vjp_method='adjoint'` 现在使用真正的 CUDA 反向 AD：每个波数/频率建立共享计算图，一次反向计算全部所选结构参数，避免逐参数重跑正演。`auto` 在支持时选择该路径。CPU 保留原生反向 AD。

CUDA 张量、解析导数、PTAM=0 时，`grtm.torch.Forward` 默认将频谱、FFT、辐射合成、应变/应力及源梯度保留在 GPU。模型拓扑/固定网格规划仍在主机。`device_pipeline=False` 保留旧的主机合成接口；NumPy/JAX 回调也仍有主机数组边界。PTAM JVP 在主机，原生 CUDA VJP 要求 PTAM=0。

统一的 `|omega| < 5e-3 * vmax * nk * dk` 准静态判据保留主机扩展精度，正演、JVP 与伴随使用相同条件；CUDA 反向仅回传这些受保护频点的余切。阈值由 dev6 的 3e-3 加严，避免低频转换区的导数舍入误差。`Forward.execution_stats()` 从自动微分工作线程捕获最近一个模型的 GPU/CPU 频点数、参数数及 tape 大小；`Solver.backend_stats()` 仍是调用线程的原生统计。固定算例通过验证不等于所有极端长周期模型都已验证。

RF 默认改为原生 CPU 核；`kernel='torch'` 保留独立 CPU/CUDA Torch 对照。
准备阶段使用原生解析材料伴随；`threads` 约束原生 CPU 工作线程，Torch FFT 等另由 `torch.set_num_threads()` 控制。
旧 NumPy `synthetic_rf` 是独立接口。CUDA RF 先在 warp 内归约频率余切再写梯度临时量，
`gi`/`gbc` 的频率行数约降至 1/32；这不意味着 forward tape 或总显存也下降 32 倍。

面波默认 `differentiation="adjoint"` 复用左右传播状态；`"forward"` 保留原解析参考。群速度核使用解析混合导数，临界坐标保留原解析路径。CPU 的基础量和敏感核按模型 × 周期并行，寻根仍保留受限的逐模型续接。
默认根容差保持 1e-10 km/s。CUDA 数值寻根和微分在 GPU，少量输入检查/搜索网格准备
在主机。生产路径不引入有限差分，也不依赖 disba。

## 批处理与内存

RF、面波具有原生 CUDA 模型批处理。Green CUDA 张量路径顺序派发模型并在 GPU 堆叠结果；旧回调路径使用有序主机模型工作线程。两者均未将模型维度融合成一个 GPU kernel。
`workers=4, threads=4` 约束原生主机部分为 16 线程预算，共享震源梯度按模型顺序归约。
`workers=1` 不启动主机线程池。Green CUDA 按主机工作线程复用缓冲区与独立 stream。
每个工作区的 tile 临时量目标为 512 MiB；已保留的峰值缓冲区、其他入口缓存和输出
另外占内存，不能理解成进程总显存上限。增大并发前应实测较小批量。

见[速度与精度报告](PERFORMANCE_0.6.0.dev5.zh-CN.md)，包含原始 disba/pyfk 对照，
以及[批量计时定义](BATCH_EXECUTION.zh-CN.md)。旧 dev4 报告保留为历史结果。

解析公式、CPU Levin 联合波形/JVP 及验证范围见[算法说明](ANALYTIC_METHODS.zh-CN.md)。
