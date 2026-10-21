# GRTM 的振荡积分与解析伴随

0.6.0.dev7 在同一套 GRTM 物理方程上优化三类正演与导数。生产导数不扰动模型、不重新寻根做差分；有限差分只用于独立验证。参数为层界面或厚度、Vp、Vs、密度，以及受衰减模型支持的 Qp/Qs。

## 三类问题怎样统一

频散求解特征方程 `F(c,T,m)=0`，一阶灵敏度为 `c_m=-F_m/F_c`。新实现保存各层的右传播状态与左伴随，复用它们求全部层参数的 `F_m`。通常全部核的层数复杂度从 O(L²) 降为 O(L)；临界模态坐标附近保留原解析路径作为精度保护，故不能把这个复杂度变化直接当作任何任务的加速比。群速度 `U=c²/(c+T c_T)` 的核通过根路径 `(dc/dT,dT)=(c_T,1)` 上的解析混合导数计算，没有周期差分。

接收函数与格林函数是受迫响应，不能直接套用表面波特征值的能量积分公式。它们采用相同的方程微分原则：对 `AX=B`，有 `dX=A⁻¹(dB-dA X)`；反向求解 `AᴴQ=bar_X`，得到 `bar_B=Q`、`bar_A=-QXᴴ`。接收函数原生代码直接使用这个矩阵伴随并解析求材料导数；格林函数对共享 GRTM 源码中的复数算子应用解析反向链式法则。复数损失采用 `dL=Re(sum(conj(bar_x)*dx))`。频散复解析特征方程的双线性转置约定与它相对应，不能漏掉两种表示之间的共轭关系。

[QEDispInv](https://github.com/pan3rock/QEDispInv) 的特征函数核提供了参考思路；其接口调用 CPS 的 `sregn96/slegn96`。这里独立实现 FP64 GRTM 特征方程伴随，没有复制该库的 CPS 内核，也不声称与其全部实现细节相同。[CPS 群速度核教程](https://rbherrmann.github.io/ComputerProgramsSeismology/TUTORIAL/SWKERNELS/index.html) 说明其群速度核还涉及数值周期导数；本实现使用解析混合导数。

## Levin 用在哪里

格林函数的波数积分包含 `J0(kr)` 和 `J1(kr)`。Levin 法利用贝塞尔函数的微分方程，求振荡积分的辅助微分方程，再由端点值计算积分。实现批量采样、缓存只依赖几何的实 SVD 分解，并把正演与解析 JVP 放在同一组自适应面板上。近场可用 Gauss；`method='auto'` 根据振荡范围选择。这里实现的是 Levin 类方法，没有额外声称实现 Filon 法。

频散求根和当前平面波接收函数没有这个波数积分。它们分别优化特征方程与散射方程；RF 的频率到时间变换使用 FFT。Levin 并不代替 FFT 或全部寻根步骤。

参考：[Levin 的振荡积分误差分析](https://www.tau.ac.il/~levin/OscIntAnal.pdf)、[自适应 Levin 方法](https://arxiv.org/abs/2211.13400)。有限截止、采样残差和嵌入误差估计并不构成无穷尾项及底层浮点误差的严格上界。

## 简单接口

```python
import numpy as np
import torch
import grtm
from grtm.torch import Forward

# 面波：同一接口支持相速度、群速度及全部参数核。
layers = [[0., 3.6, 2., 2.5], [2., 6.5, 3.6, 2.8], [35., 8., 4.5, 3.3]]
sw = grtm.SurfaceWave(layers, device='cuda:0', differentiation='adjoint')
phase = sw.sensitivity([3., 8., 18.], kind='phase', parameter='all')
group = sw.sensitivity([3., 8., 18.], kind='group', parameter='all')
# differentiation='forward' 保留原解析参考；不是有限差分。

# RF：层间准备、传播与反向材料导数均使用原生解析代码。
x = torch.as_tensor(layers, dtype=torch.float64, device='cuda:0').requires_grad_()
rf = grtm.JointRFForward([.04, .06], device='cuda:0', kernel='native')
rf_gradient, = torch.autograd.grad(rf(x)['rf'].square().sum(), x)

# 格林函数：模型使用 SI 单位；固定网格可由 operator.freeze_grid 获取。
model_si = grtm.example_model(units='si')
model_si.update(log2_samples=5, dt=.4, distances=[10000.], t0=[0.], ptam=0)
green = Forward(model_si, backend='cuda', parameters=['vp', 'vs', 'rho'],
                vjp_method='adjoint')
x_si = torch.as_tensor(model_si['layers'], dtype=torch.float64,
                       device='cuda:0').requires_grad_()
mt = torch.as_tensor(grtm.strike_dip_rake(20, 40, 60, 1e15),
                     dtype=torch.float64, device='cuda:0')
waveform = green(x_si, mt)
gradient, = torch.autograd.grad(waveform.square().sum(), x_si)
print(green.execution_stats())

# CPU 自适应 Levin：方向的单位/形状与 model_si['layers'] 相同。
direction = np.zeros_like(np.asarray(model_si['layers'], dtype=float))
direction[:, 2] = 1.  # Vs 方向，单位 m/s
joint = grtm.Solver().forward_quadrature(
    model_si, cutoff=4., method='levin', direction=direction,
    moment_tensor=mt.detach().cpu().numpy(), fields=('displacement',),
    atol=1e-10, rtol=0.)
y = joint['displacement']
dy = joint['jvp']['displacement']
```

上例中的 `layers` 是原生单位的层顶深度、Vp、Vs、密度，可附加 Q 列；`model_si` 是完整格林函数 SI 配置。波数截止始终使用 km⁻¹。Levin 的容差针对原始积分通道，不能直接解释成位移或应力的相对误差保证。`direction` 使用解析被积函数导数，不对面板选择的离散分支求导。等深直接场恢复需显式传入 `subtract_direct=True`，此时积分目标变为解析 Abel 直接场加有限截止散射场。

## 设备边界与验证

- 面波 CPU/CUDA 的寻根和核、RF CPU/CUDA 的层间准备与传播都在所选设备上。默认采用 FP64；原生框架反向接口仅保证一阶梯度。
- CUDA `Forward` 在 `PTAM=0`、解析导数且输入为 CUDA 张量时，频谱、FFT、辐射合成、应变/应力、源参数梯度和结构反向 AD 均在 GPU。只在主机规划模型拓扑和固定网格。`device_pipeline=False` 可复查旧的主机合成路径。NumPy 接口仍返回主机数组。
- 动态 P/S 模态在低频大波数下近乎线性相关；统一的 `|omega|/(vmax*kmax)<5e-3` 保护让这些频点保留主机扩展精度。CUDA 反向只传输受保护频点的余切量。`execution_stats()` 从自动微分工作线程捕获最近一个模型的实际设备计数；批次不冒充融合为一次原生求解。
- 自适应 Levin/Gauss 与联合 JVP 接口目前在 CPU。MPS 的 RF/面波仍是明确提示的 CPU FP64 桥接，输出实张量为 float32；格林函数 Metal 保留既有混合实现。
- 独立验证覆盖表面波高阶模重新求根差分、物理缩放恒等式、RF 独立张量图、格林函数 JVP/VJP 对偶、GPU 与 CPU 的全部物理场、非默认流、共享源批处理、掩码及低频保护。高阶模 disba 灵敏度不作为正确性基准。

性能必须分别报告已知根后的核、完整 API 调用、单模型与批处理。测量协议和实际结果另见版本性能报告；算法复杂度及单点原型结果不能替代端到端实测。
