# 格林函数的显式层内波场积分核（dev9）

`vjp_method='wavefield'` 为格林函数新增 CPU/CUDA 显式解析核。它与原有 GRTM 使用相同的 P-SV/SH 模态、本构、震源归一化、固定波数积分、FFT 和物理合成；常规频率的结构梯度不再建立传播 AD tape。原有 `vjp_method='adjoint'` 保留为参考，默认策略暂不改变。

## 使用

```python
import numpy as np
import torch
import grtm
from grtm.torch import Forward

device = 'cuda'  # CPU 使用 'cpu'
model = grtm.example_model(units='si')
model.update(source_depth=8000., receiver_depth=0., ptam=0,
             log2_samples=7, dt=.1, distances=[10000.], t0=[0.])
op = Forward(model, backend=device, vjp_method='wavefield', threads=4,
             parameters=['depth', 'vp', 'vs', 'rho', 'qp', 'qs'])
layers = torch.tensor(np.asarray(model['layers']), dtype=torch.float64,
                      device=device, requires_grad=True)
source = torch.tensor(grtm.strike_dip_rake(20., 40., 60., 1e15),
                      dtype=layers.dtype, device=device)
y = op(layers, source)
gradient, = torch.autograd.grad(y.square().sum(), layers)
print(op.execution_stats())
```

支持 `[B,L,6]` 模型 batch、单一或逐模型震源、位移/位移梯度/应变/应力、时间域和频域、Vp/Vs/密度/Qp/Qs/界面深度。`Forward.sensitivity(...)` 保留调用者选择的 `wavefield` 方法。NumPy 使用 `ForwardOperator(..., vjp_method='wavefield')`；底层 `Solver.spectrum_vjp(..., method='wavefield')` 返回原始模型的六列梯度。

这是标量目标或任意输出余切的 VJP，不是逐样点完整 Jacobian。所有材料参数由一次正向/伴随波场计算收缩得到，计算量仍随层数、频率和波数节点数增加；“不逐参数重做传播”不等于与层数无关。

## 与现有 GRTM 一致的本构

令复速度为 `a=Vp(1+l/Qp)`、`b=Vs(1+l/Qs)`，其中 `l=log(ω/2π)/π+i/2`。现有求解器采用

\[
\mu=\rho V_s^2,\qquad C=\mu a^2/b^2,\qquad D=\mu\omega^2/b^2,\qquad \lambda=C-2\mu.
\]

因此 D/ω² 是这一传播约定中的有效惯性系数。不能在有衰减时直接把它替换为实密度，也不能直接套用无衰减面波的能量归一化公式。

P-SV 的状态为 `y=(u_x,u_z,t_x,t_z)`，满足 `y'=Ay`：

\[
A=\begin{pmatrix}
0&k&1/\mu&0\\
-(1-2\mu/C)k&0&0&1/C\\
4\mu(1-\mu/C)k^2-D&0&0&(1-2\mu/C)k\\
0&-D&-k&0
\end{pmatrix}.
\]

SH 为 `u'=t/μ, t'=(μk²-D)u`。采用 GRTM 的下降/上升模态，分别在层顶/层底锚定，全部传播指数有界；半空间仅保留衰减模态。通过模态位移/牵引消元得到每层的精确动态刚度，再用小块三对角系统求解所有震源分量及伴随分量。这是同一层状弹性边值问题的精确表示，没有有限元深度离散。

## 闭式积分与完整梯度

对公共约定 `δJ=Re(gᴴ δu)`，将共轭余切作为双线性伴随源。若伴随场为 v，其共轭状态为 `p=(-t_v,u_v)`，则材料项为

\[
\delta J_{\rm volume}=-\Re\sum_\ell\int_{z_\ell}^{z_{\ell+1}}p^T\delta A\,y\,dz.
\]

正向和伴随场均是四个（SH 两个）指数模态的线性组合。每一模态对直接计算

\[
\int_0^h e^{\alpha z+\beta}\,dz=e^\beta h\,\operatorname{exprel}(\alpha h),\quad
\operatorname{exprel}(x)=(e^x-1)/x.
\]

小参数使用展开以避免消减，大正实部改从另一端评价；半空间使用收敛尾积分 `-1/α`。积分不增加深度节点，也不扰动材料参数。首先收缩出 μ、C、D 的三个核，再解析映射到 Vp、Vs、ρ、Qp、Qs。

完整实现还包含：

- 力源和矩张量源的位移/牵引跳跃，以及源归一化的材料导数。
- 接收端位移及垂向导数余切，对应伴随力/偶极源；接收算子的显式材料项。
- 界面移动的边界 Hamiltonian 项，以及虚拟源界面到原始深度参数的映射。
- Hankel/Bessel 权重的精确转置。权重代码由原始 `distance_integrand` 生成，不另换 Bessel 近似。
- 现有应力本构、震源、FFT、窗函数和坐标旋转的直接导数。

## 精度保护与设备执行

以下情形保留已有反向 AD，并通过诊断计数注明：

1. 原有 `GRTM_QUASISTATIC_RATIO=5e-3` 判定的低频/高波数病态频率，沿用原有 CPU 扩展精度路径，阈值不放宽。
2. 源与接收点同深度、源在自由表面的分布式迹约定。
3. 源/接收点切分后的有限子层薄于 `1e-4 km`，或层求解出现奇异/非有限结果。

常规 CUDA 路径在同一 GPU 上完成层模态、正向/伴随块求解、解析积分、参数映射、确定性规约、FFT 和物理合成。第 2/3 类常规频率回退仍在 GPU；只有第 1 类使用原有 CPU 精度保护。CPU 没有降低到 float32。非默认 CUDA stream 和 PyTorch 惰性负号/共轭余切包含在测试中。

CUDA 分为按 `(频率,波数,层)` 并行的模态准备、按 `(频率,波数)` 求解、按层积分、参数映射和规约；避免让少量线程串行完成所有层。当前 batch API 逐模型调度原生调用，尚未把不同模型融合到一次跨模型 CUDA launch，不能声称已经实现跨模型融合。

极低频保护仍可能主导大时间窗的端到端耗时。`threads` 也控制这部分 CPU 工作，比较时必须同时披露 CPU 线程数。无保护频率的任务与有保护频率的任务应分别报告。固定积分网格是导数定义的一部分；这里没有通过减少波数节点、改变 PTAM 或降低精度换取速度。

## QEDispInv 对照的范围

QEDispInv 的 `SwEgn96::kernel(freq,c)` 接收已知相速度并调用 CPS 特征函数/能量积分，不在该函数中寻根。其模型入口为 float32，但 Fortran 内部主要数组为 float64；速度差不能简单归因于“全部使用单精度”。我们的对照固定相同相速度、周期和模型，对照的是已知根的相速度核，不把这个耗时直接当作完整格林函数基准。

dev9 去掉了面波相速度能量核不使用的 `Fc/Ft` 预计算；群速度需要的周期导数仍保留。QEDispInv 的专用实数公式仍比当前通用复数 GRTM 能量实现更快，具体差距和测试条件见性能报告。其源码未复制进入求解器。

来源：[QEDispInv 固定版本的核入口](https://github.com/pan3rock/QEDispInv/blob/ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa/src/swegn96.cc)、[Fortran 计算实现](https://github.com/pan3rock/QEDispInv/blob/ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa/fortran/sregn96.f90)。
