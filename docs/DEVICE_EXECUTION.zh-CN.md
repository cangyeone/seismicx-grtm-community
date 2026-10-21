# 计算设备与批量接口 — 0.6.0.dev5

[English](DEVICE_EXECUTION.md) | 简体中文

正演和结构微分优先在选定的同一设备执行。RF、面波的 CUDA 路径将数值传播、寻根、
敏感核和反向计算保留在 CUDA；格林函数现在也有原生 CUDA 结构微分积分，只有下述
精度保护等环节仍需主机参与。CUDA 错误直接报告，不会悄悄切换成 CPU 求解器。

| 类型 | CPU | CUDA | Apple / MPS |
|---|---|---|---|
| 点源格林函数正演 | 原生 FP64，低频 long double | GPU FP64 积分；主机精度保护、FFT/物理合成 | 现有混合算法：主机 FP64 传播，Metal 补偿 float-pair 接收端积分 |
| 格林函数结构微分 | 原生切线 / 节点局部反向 AD | 原生解析切线积分和 GPU VJP 收缩；低频主机保护 | CPU 结构 AD |
| 联合 RF 正演与 VJP | 共享原生 complex128 核，Torch 准备/FFT | 原生 complex128 核及 Torch 准备/FFT 都在 CUDA | 显式 CPU FP64 桥接；实数输出 MPS float32，复频谱留在 CPU complex128 |
| 面波相/群速度与敏感核 | 原生 FP64 dual/hyper-dual | CUDA 原生 FP64 dual/hyper-dual、搜索和寻根 | 显式 CPU FP64 桥接；输出 MPS float32 |

MPS 的 RF/面波支持是兼容接口，**不代表原生 GPU 加速**。选用时会警告，并标注
`backend='mps-cpu-fp64'` 和 CPU 计算设备。MPS 输入的 float32 舍入和输出转换仍然存在；
内部 FP64 不能恢复输入已丢失的精度。需要保留双精度时应选择 CPU/CUDA float64。
所有原生自定义 backward 只支持一阶微分。

## 最简正演和梯度调用

安装 [dev5 二进制预览](PREVIEW_0.6.0.dev5.zh-CN.md)、NumPy 和适配设备的 Torch。
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

Green CUDA JVP 沿原 GRTM 方程传播链式导数。CUDA VJP 对所选参数方向做解析积分，
在 GPU 收缩频谱余切，每个方向仅回传一个标量，避免保存完整波形 Jacobian，也避免
逐参数回传导数频谱。它**不是 CUDA 反向 tape**，计算量仍随所选参数数量增长。
CPU 的 `vjp_method='adjoint'` 使用节点局部反向 AD；CUDA 的同名选项使用原生方向
收缩，支持时 `auto` 自动选择该路径。反演应只选需要求导的参数。

原准静态判据 `omega < 3e-3 * vmax * nk * dk` 触发的频点保留 CPU long double
消去误差保护。PTAM JVP 也保留在主机；原生 CUDA VJP 要求 PTAM=0。几何/网格规划、
结果数组、FFT、物理场和震源合成仍通过主机接口，所以 Green 还不是全流程 GPU 常驻。
在同一调用线程执行结构微分后，`Solver.backend_stats()` 会报告 CUDA/CPU 频点数和
所选方向数；工作线程的统计暂不汇总到主线程。极端长周期仍需单独检查梯度收敛性。

RF 默认改为原生 CPU 核；`kernel='torch'` 保留独立 CPU/CUDA Torch 对照。
`threads` 约束原生 CPU 工作线程，Torch 准备阶段另由 `torch.set_num_threads()` 控制。
旧 NumPy `synthetic_rf` 是独立接口。CUDA RF 先在 warp 内归约频率余切再写梯度临时量，
`gi`/`gbc` 的频率行数约降至 1/32；这不意味着 forward tape 或总显存也下降 32 倍。

面波 CPU 的基础量和敏感核按模型 × 周期并行，寻根仍保留受限的逐模型续接。
默认根容差保持 1e-10 km/s。CUDA 数值寻根和微分在 GPU，少量输入检查/搜索网格准备
在主机。生产路径不引入有限差分，也不依赖 disba。

## 批处理与内存

RF、面波具有原生 CUDA 模型批处理。Green 使用有序主机模型工作线程，现已覆盖
PyTorch 批量 backward，但没有把模型维度融合成一个 GPU kernel。
`workers=4, threads=4` 约束原生主机部分为 16 线程预算，共享震源梯度按模型顺序归约。
`workers=1` 不启动主机线程池。Green CUDA 按主机工作线程复用缓冲区与独立 stream。
每个工作区的 tile 临时量目标为 512 MiB；已保留的峰值缓冲区、其他入口缓存和输出
另外占内存，不能理解成进程总显存上限。增大并发前应实测较小批量。

见[速度与精度报告](PERFORMANCE_0.6.0.dev5.zh-CN.md)，包含原始 disba/pyfk 对照，
以及[批量计时定义](BATCH_EXECUTION.zh-CN.md)。旧 dev4 报告保留为历史结果。
