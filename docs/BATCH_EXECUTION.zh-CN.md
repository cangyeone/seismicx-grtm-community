# 模型批量、CUDA 执行与吞吐量

[English](BATCH_EXECUTION.md) | 简体中文

本指南更新到 [0.6.0.dev5](PREVIEW_0.6.0.dev5.zh-CN.md)。三类接口的完整可运行例子、
单位和设备边界见[设备与批量接口](DEVICE_EXECUTION.zh-CN.md)；
[当前实测](PERFORMANCE_0.6.0.dev5.zh-CN.md)包含单模型与批量、纯正演与微分。
[dev4 批量报告](BATCH_SCALING_20261021.zh-CN.md)保留为历史数据。

| 接口 | 批量含义 | 结构微分 |
|---|---|---|
| `JointRFForward` | 原生 CUDA 模型×射线×频率任务 | 原生 CPU/CUDA VJP；Torch 准备和 FFT 与计算设备一致 |
| `SurfaceWave` | CUDA 模型/周期/搜索/参数任务 | CUDA dual/hyper-dual；CPU 通过 `threads` 并行模型与周期 |
| `Solver.forward_batch` | 主机线程调度独立模型调用 | 每个模型保留主机 I/O 边界；CUDA 临时缓冲按线程复用 |
| `ForwardOperator` / PyTorch `Forward` | `workers` 控制有序模型工作线程，包括 backward | CUDA 解析积分/收缩及低频主机保护；CPU/Metal 主机 AD |

Green 单模型内部并行频率、波数和接收点，模型维度目前由主机调度。
`workers=4` 不是把4个模型融合成一个 GPU kernel，也不是使用4张GPU。
`workers=4, threads=4` 对原生主机部分提供16线程预算；批量共享震源梯度按模型顺序求和。

RF/面波输入可为 `[B,L,4]` 的模型张量。CPU/CUDA 推荐 float64，最终一层为半空间。
RF 原生 CPU 线程数由 `threads` 控制，Torch 准备还受 `torch.set_num_threads()` 控制；
`kernel='torch'` 保留独立参考实现。面波 `threads=16` 用于 CPU；CUDA 保持默认1。
面波默认层顶深度参数化，也可用 `parameterization='thickness'`。
相/群速度支持一阶模型 autograd；显式 `sensitivity()` 返回导数数据，不构成高阶图。

MPS 的 RF/面波是带警告的 CPU FP64 兼容路径，实数输出转为 MPS float32，
复 RF 频谱仍在 CPU complex128。Green Metal 是主机 FP64 传播与补偿 float-pair
GPU 接收端积分的混合后端，结构梯度在CPU。不能把这些统称为全流程 MPS 加速。

## 统一计时口径

```python
import statistics
import time
import torch

def seconds(function, device='cuda:0', repeat=3):
    function()
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        elif device == 'mps':
            torch.mps.synchronize()
    sync()
    trials = []
    for _ in range(repeat):
        sync()
        start = time.perf_counter()
        function()
        sync()
        trials.append(time.perf_counter() - start)
    return statistics.median(trials)
```

批量 B 耗时 T 秒：

- 整批延迟为 T，平均每模型 `1000*T/B` 毫秒。
- 吞吐量为 `B/T` 模型/秒。
- 同批量 CUDA 相对 CPU 加速比为 `T_CPU(B)/T_CUDA(B)`。
- GPU 相对自身 B1 的吞吐提升为 `B*T_CUDA(1)/T_CUDA(B)`。

构造与初次上传之外的热身计时，不代表每步更改模型的完整反演耗时；后者还应包含
模型生成、必要网格更新和传输。增大批量可能提高吞吐，也会增加内存和整批延迟。
要同时报告层数、接收点/周期/FFT长度、CPU线程预算、精度与微分算法；
不能用不同主机、不同误差设置或旧版较慢CPU基线相除后称为新版普遍加速比。
