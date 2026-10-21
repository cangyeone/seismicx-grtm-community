# 模型批量、CUDA 执行方式与吞吐量

[English](BATCH_EXECUTION.md) | 简体中文

以下接口包含在 [0.6.0.dev4 编译预览版](PREVIEW_0.6.0.dev4.zh-CN.md)，稳定 PyPI 仍为 0.5.1。
[批量扩展测试](BATCH_SCALING_20261021.zh-CN.md)给出各模型数、CPU 预算、延迟、每秒模型数与数值核验。

| 接口 | 模型 batch 的执行方式 | 结构微分 |
|---|---|---|
| `JointRFForward` | 原生 CUDA 按模型×射线×频率并行 | 原生 CUDA VJP＋Torch CUDA 准备/FFT；另有 CPU Torch 参考 |
| `SurfaceWave` | 原生 CUDA 按模型、周期、搜索点、参数任务并行 | CUDA dual/hyper-dual 核；CPU 用 `threads` 按模型并行 |
| `Solver.forward_batch` | 主机线程池调度独立格林函数调用 | 每个模型仍有自己的正演分配和传输 |
| `ForwardOperator` 的批量输入 | 目前顺序调用各模型 | 格林函数 JVP/VJP 仍在 CPU，CUDA 正演也是混合流程 |

格林函数单个模型内部的频率、波数、接收点有 GPU 并行，但外层模型尚未融合进同一个 CUDA 启动。
`workers=4` 指 4 个主机调度线程，不代表 4 个融合 GPU 模型或 4 块显卡。

## 面波批量模型

```python
import numpy as np
import torch
from grtm import SurfaceWave

# 顶深 km、Vp/Vs km/s、密度 g/cm³；末行为半空间。
one = np.array([[0.,3.2,1.65,2.1], [.8,5.6,3.15,2.6],
                [5.,6.7,3.8,2.9], [25.,8.1,4.6,3.3]])
models = np.repeat(one[None], 64, axis=0)  # [B,L,4]
earth = torch.tensor(models, dtype=torch.float64, device='cuda:0')
sw = SurfaceWave(earth)  # CUDA 保持默认 threads=1
periods = np.geomspace(2,50,12)
c = sw.phase_velocity(periods)  # [64,12]
k = sw.sensitivity(periods, kind='group', parameter='velocity_s')['kernel']
print(k.shape)  # [64,12,4]

cpu = SurfaceWave(models, threads=16)  # 显式 CPU 模型工作线程
c_cpu = cpu.phase_velocity(periods)
```

`threads=16` 用于 CPU，不用于 CUDA。批量不改变 FP64/complex128 与默认 1e-10 km/s 根容差。
默认模型第一列为顶深，厚度数组请加 `parameterization='thickness'`。
CPU NumPy 输出与 CUDA Tensor 遵循相同形状约定。
Tensor 速度支持一阶模型自动微分；显式 `sensitivity()` 是导数数据，不是可继续求高阶导数的计算图。
详见[面波参数与限制](SURFACE_WAVES.zh-CN.md)。

## 接收函数与梯度

```python
import torch
from grtm import JointRFForward

# 此处是接口示例；批量测速实际使用 32 层。
one = torch.tensor([[0.,3.6,2.,2.5], [2.,6.5,3.6,2.8],
                    [35.,8.,4.5,3.3]], dtype=torch.float64, device='cuda:0')
earth = one[None].repeat(64,1,1).requires_grad_()
op = JointRFForward([.06], device='cuda:0')
pred = op(earth)
rf = pred['rf']              # [64,1,6,81]
vs_app = pred['vs_app']      # [64,1,6,41]
gradient = torch.autograd.grad(rf.square().mean(), earth)[0]  # [64,3,4]
```

列为顶深、Vp、Vs、密度，native 单位 km、km/s、g/cm³；首层顶深固定，末层为半空间。
CUDA 传播/反向与 Torch 准备/FFT 均在 GPU。
`device='cpu'` 选择 CPU Torch 参考；测速时显式指定其全局 Torch 线程数。
接收函数 4 个 Torch 线程与其他接口的 16 个模型工作线程是不同资源配置。
旧的 NumPy `synthetic_rf` 是独立的 CPU 接口。

## 格林函数的主机批量调度

```python
import numpy as np
import grtm

model = grtm.example_model(units='si')
model.update(log2_samples=6, dt=.4, distances=[10000.,30000.], t0=[0.,0.])
base = np.asarray(model['layers'])
models = []
for factor in np.linspace(.99,1.01,16):
    layers = base.copy()
    layers[:,2] *= factor
    models.append(dict(model, layers=layers))
moment = grtm.strike_dip_rake(20,40,60,1e15)
solver = grtm.Solver(backend='cuda')
out = solver.forward_batch(models, moment_tensor=moment,
                           fields=('displacement',), workers=4, threads=4, device=0)
print(out['displacement'].shape)  # [16,2,64,3]

# 正演在 CUDA，下面的结构 VJP 在 CPU。
op = solver.operator(model, parameters=['vp','vs'], threads=16,
                     device=0, vjp_method='adjoint')
linear = op.linearize(base, moment)
gradient, _ = linear.vjp(np.ones_like(linear.value), moment_gradient=False)
```

这里 4 个模型调度线程×每模型 4 个原生 CPU 线程，总 CPU 预算为 16。
模型读写、GPU 分配与传输仍逐模型进行。
`op(earth_batch, moment)` 虽接受 batch，但当前顺序遍历，不会自动使用 `forward_batch` 的线程池。
两种接口都不会自动把结构 VJP 变成 GPU 模型批量核，详见[正演与微分指南](PYTHON_FORWARD.zh-CN.md)。

## 延迟、吞吐量与公平计时

以独立的面波示例对象为例，先预热，再在每次计时前后同步 CUDA：

```python
import statistics
import time

def seconds(function, device, repeat=3):
    function()
    torch.cuda.synchronize(device)
    trials = []
    for _ in range(repeat):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        trials.append(time.perf_counter()-start)
    return statistics.median(trials)

t = seconds(lambda: sw.phase_velocity(periods), 'cuda:0')
batch = 64  # 对应上方面波示例
print(batch/t, 'models/s;', 1000*t/batch, 'ms/model')
```

各段代码为独立示例，计时应保留对应的模型、设备对象。批量 B、整批耗时 T 秒时：

- 整批延迟为 T；平均每模型耗时为 `1000*T/B` 毫秒。
- 吞吐量为 `B/T` 模型/秒。
- 同批量 CUDA 相对 CPU 加速比为 `T_CPU(B)/T_CUDA(B)`。
- GPU 相对单模型的吞吐提升为 `B*T_CUDA(1)/T_CUDA(B)`。

设备驻留、复用对象的计时不含初次构造/上传。实际反演每步改变模型时，还应测完整步骤，包含模型生成、
需要的搜索网格刷新与搬运。小 batch 中，接口校验与主机调度仍可能占较大比例。
RF 反向内存随模型数、层数、FFT 频率数增长；应结合实测吞吐和显存选 batch，不假定越大越快。
batch 不改变接口声明的精度，也不改变当前 CPU/GPU 微分边界。

以上四段 Python 示例已于 2026-10-21 在安装后的 dev4 wheel 与 RTX 5090 执行通过，
中英文 Python 语法树一致，见[执行记录](../verification/batch-scaling-20261021/documentation-examples-20261021.json)。
