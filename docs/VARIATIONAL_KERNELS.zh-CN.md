# 从特征函数核到 RF、格林函数伴随核

> 本页描述 dev8。dev9 已新增格林函数显式连续深度波场积分后端，详见[当前方法说明](GREEN_WAVEFIELD_KERNELS.zh-CN.md)。下文关于格林函数继续使用反向 AD 的表述是 dev8 的历史状态。

dev8 增加 `SurfaceWave(differentiation='eigenfunction')`，使用 **GRTM 模态波场、FP64 闭式能量积分和解析周期导数**。RF、格林函数增加 `sensitivity` 接口，将多个观测量合成一个标量目标后，一次反向计算全部已选参数梯度。

这里区分两种实现：面波新路径显式计算特征函数能量积分；RF 使用散射矩阵伴随与材料核收缩，格林函数继续使用经验证的 GRTM 复数算子解析反向链式法则。**后两者实现受迫方程的离散伴随恒等式，没有另行实现连续深度体积积分后端，也没有把旧反向 AD 改名为特征函数积分。** 三者共享方程微扰原理，不共用一个本征值公式。

## 面波：本征问题

对弹性、各向同性固体，令 `D(k,ω,m)=K(k,m)−ω²M(m)`，`D u=0`。左乘共轭本征函数消去 `δu` 后，固定频率有

\[
\partial_m c=\frac{c}{k}\frac{\langle u,D_m u\rangle}{\langle u,D_k u\rangle},\qquad c=\omega/k.
\]

分母满足 `⟨u,D_k u⟩=2ω U∫ρ|u|² dz`，因此这是通量/能量归一化，与模态任意幅值无关。

实现由 GRTM 稳定反射/透射递推重建模态。下行波锚定层顶，上行波锚定层底，避免厚层中增长指数溢出。所有指数对的重叠积分均闭式求值；接近零的指数和使用 `exprel` 的解析级数，不用深度离散或差分。

令 `Idiv=∫|∇·u|² dz`、`Istrain=∫2 ε(u):ε(u)* dz`、`Iu=∫|u|² dz`，各层分子为

\[
N_{V_p}=2\rho V_p I_{div},\quad
N_{V_s}=2\rho V_s(I_{strain}-2I_{div}),\quad
N_\rho=(V_p^2-2V_s^2)I_{div}+V_s^2 I_{strain}-\omega^2I_u.
\]

层厚核使用界面形状导数：层间连续位移、牵引下，对拉格朗日密度的 Hamiltonian 跳跃求值，再对下方界面作后缀求和。半空间厚度和 Love 的 Vp 核严格为零。

群速度 `U=c²/(c+T c_T)`。沿根路径 `(dc/dT,dT)=(c_T,1)` 对波场、积分、能量归一化作一阶双数传播，同时得到 `c_m` 和 `d(c_m)/dT`，代入

\[
U_m=\frac{2cc_m(c+Tc_T)-c^2[c_m+T\,d(c_m)/dT]}{(c+Tc_T)^2}.
\]

没有逐参数全栈传播，也没有周期差分。非简根、模态交叉附近仍可能病态；未找到根返回既有缺失状态。`kernel_path` 为 0 表示能量积分，1 表示同设备解析参考保护，2 表示缺根。返回 `energy_residual` 和 `boundary_residual`（群速度时同时检查沿根路径周期导数的无量纲残差）；残差通过只能验证当前模态的方程一致性，不能证明模态编号或没有漏根。默认方法仍是 `adjoint`，不假定能量路径在所有负载上都更快。

## 受迫波场：RF 与格林函数

对于 `A(m)u=f(m)` 和实标量观测量 `J(u,m)`，以 `δJ=Re(gᴴδu)+J_m δm` 定义复数余切。解 `Aᴴv=g` 得到

\[
\partial_m J=\operatorname{Re}\{v^H[f_m-A_m u]\}+J_m.
\]

这就是特征函数微扰思想在受迫问题中的对应形式。弹性连续弱式的体积项可写成

\[
K_\lambda=-\operatorname{Re}\int (\nabla\cdot v)^*(\nabla\cdot u),\quad
K_\mu=-\operatorname{Re}\int 2\epsilon(v)^*: \epsilon(u),\quad
K_\rho=\operatorname{Re}\int\omega^2v^*\cdot u.
\]

离散 GRTM 对每个连续性/边界解 `X=A⁻¹B` 使用 `Q=A⁻ᴴ bar_X`、`bar_B=Q`、`bar_A=−QXᴴ`，实现相同的变分恒等式。RF 新材料回传先收缩到 Lamé 参数和密度，再解析映射到 Vp/Vs/Q，去掉原来的局部参数方向循环。格林函数用原生解析复数反向算子实现该离散恒等式，保留源项、应力的直接本构项和源接收深度链式项。

RF 的余切必须经过 **水平与垂直两分量**。实际反褶积 `H Z*/max(|Z|²,η max|Z|²)` 的分母、水位参考、归一化、Gaussian 滤波、FFT 和质点运动均参与导数。只微分 H 或冻结 Z 是错误的。在 `max` 恰好切换或出现并列极值处，使用框架约定的子梯度，不宣称经典可微。

无衰减特征函数能量公式不能直接移植到衰减/流体/各向异性。新面波路径只处理既有各向同性弹性固体。RF 与格林函数保留各自已支持的复速度/Q 模型，通过复算子导数处理衰减；这不等于已推导新的衰减面波能量核。

## 接口和复杂度

```python
import torch
import grtm
from grtm.torch import Forward

layers = [[0., 3.6, 2., 2.5], [2., 6.5, 3.6, 2.8], [35., 8., 4.5, 3.3]]
sw = grtm.SurfaceWave(layers, device='cuda:0', differentiation='eigenfunction')
k = sw.sensitivity([3., 8., 18.], kind='group', parameter='all')
print(k['kernel_path'], k['energy_residual'])

rf = grtm.JointRFForward([.04, .06], device='cuda:0')
y = rf(layers)
# 一个目标，可联合 RF 与质点运动；这里零数据仅用于演示。
r = rf.sensitivity(layers, observed={'rf': torch.zeros_like(y['rf']),
                                   'vs_app': torch.zeros_like(y['vs_app'])},
                   weights={'rf': 1., 'vs_app': .1})
gradient = r['kernel']  # 与输入 layers 同形状；depth,Vp,Vs,rho

model = grtm.example_model(units='si')
model.update(ptam=0, log2_samples=5, dt=.4, distances=[10000.], t0=[0.])
green = Forward(model, backend='cuda', parameters=['vp','vs','rho'])
x = torch.tensor(model['layers'], dtype=torch.float64, device='cuda:0')
mt = torch.tensor(grtm.strike_dip_rake(20,40,60,1e15),
                  dtype=x.dtype, device=x.device)
value = green(x,mt)
g = green.sensitivity(x,mt,cotangent=torch.ones_like(value))
print(g['kernel'], g['source_kernel'], g['execution'])
```

`observed` 返回 `J=½Σ weight·|prediction−observed|²` 的梯度；`cotangent` 返回 `ReΣ conj(cotangent)·prediction` 的梯度。二选一。权重可广播，观测和余切必须与对应输出形状完全相同。结果均 detach，不修改输入 `.grad`，只保证一阶。Green 新接口强制解析伴随而不改变原对象的配置；PTAM 必须为 0。固定网格下比较差分。

参数数目不再决定完整波场求解次数；所有层的核生成与输出至少 O(L)。`reverse_passes=1` 指一次目标函数反向遍历，底层仍按模型、频率、波数及必要的精度分区计算。它**不是完整的“每个样点×每个参数”Jacobian**；若要每个样点各自的核，就需要更多伴随右端或更大输出，不能承诺总消耗与参数、观测数量无关。速度控制点可用 `m=Bθ` 的 `Bᵀg_m` 投影，不需要每个控制点再跑波场。

CPU/CUDA 使用双精度，CUDA 物理和梯度尽量留在设备上。Green 保留既有低频扩展精度保护；MPS 仍受 FP64 能力限制，RF/面波明确使用 CPU 桥接。新 Green 核接口要求 CPU/CUDA 张量，不能把 MPS 桥接称为原生 FP64 GPU。

参考：[QEDispInv 接口](https://github.com/pan3rock/QEDispInv/blob/main/src/swegn96.cc)、[CPS 核教程](https://rbherrmann.github.io/ComputerProgramsSeismology/TUTORIAL/SWKERNELS/index.html)、[de Jong 等 RF 伴随核](https://doi.org/10.1093/gji/ggac098)、[Liu & Tromp 波场伴随核](https://doi.org/10.1111/j.1365-246X.2008.03798.x)。实现独立推导，不复制 CPS/QEDispInv 内核。
