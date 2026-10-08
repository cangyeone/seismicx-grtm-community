# 可微接口与积分方法预览版 — 0.6.0.dev1

[English](RESEARCH_PREVIEW.md) | 简体中文

**0.6.0.dev1 为 GitHub 预览版**，新增方向 JVP、原生反向 VJP 和独立波数积分接口。
PyPI 稳定版 **0.5.1 不包含这些新增接口**。求解器继续只提供二进制，原生积分并未
获得适用于任意地球模型的完整误差认证。

## 安装

从 [v0.6.0.dev1 Releases](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev1)
选择与你的 CPython 和操作系统匹配的 wheel，使用实际文件名安装：

```sh
python -m pip install ./seismicx_grtm-0.6.0.dev1-<matching-tags>.whl
python -m pip install scipy python-flint
python -c 'import grtm; print(grtm.__version__, grtm.available_backends())'
```

`<matching-tags>` 是占位符。框架接口另外安装 torch 或 jax。Mac wheel 包含 CPU + Metal，
Linux/WSL x86_64 包含 CPU + CUDA，具体 Python 版本以发布资产为准。
此次预览版通过 GitHub 下载；`pip install seismicx-grtm` 仍安装 PyPI 版本。
结构导数及研究积分在 CPU 执行，GPU 仍可用于原有正演。

## 方向导数与反向梯度

```python
import numpy as np
import grtm

model = grtm.example_model(units="si")
model.update(log2_samples=5, dt=0.4, distances=[10000., 30000.],
             t0=[0., 0.], ptam=0)
layers = np.asarray(model["layers"], dtype=float)
moment = grtm.strike_dip_rake(20, 40, 60, scalar_moment=1e15)
solver = grtm.Solver(backend="cpu")
probe = solver.operator(model, parameters=["vp", "vs"], threads=2)
grid = probe.freeze_grid(layers)
op = solver.operator(model, parameters=["vp", "vs"], grid=grid,
                     vjp_method="adjoint", threads=2)
linear = op.linearize(layers, moment)
direction = np.zeros_like(layers)
direction[:, 2] = 10.  # 每层 Vs 的 +10 m/s 方向
waveform_tangent = linear.jvp(direction)
layer_gradient, moment_gradient = linear.vjp(linear.value)
```

最后一行对应示例目标函数 `0.5 * sum(waveform**2)`；实际拟合应传入实际加权损失的
波形余切。高层默认 NED/SI，方向与模型使用相同单位。

- `jvp()`：沿整个模型方向进行一次原生积分，采用解析链式法则，不依赖模型差分。
- `jacobian()`：逐参数切线积分，返回完整稠密导数，可用于单层敏感性波形。
- `vjp_method="tangent"`：逐列积分后收缩，导数存储低，但计算量随参数数目增加。
- `vjp_method="adjoint"`：每个积分节点进行反向计算，一并累加结构梯度；要求 PTAM=0。
- `auto`：支持且 PTAM=0、选中参数至少 8 个时使用反向路径，否则用切线；只是启发式。
- `linearize()`：保存输入、网格与基准输出快照，JVP/VJP 仍重新计算传播。

`parameters` 以外的列为零表示未求导，不表示波形不敏感。第一层顶深固定。
优化阶段应通过构造器 `grid=` 固定离散正演；未传入时，导数只冻结当前模型选出的
局部网格，不包含网格选择的导数。层数改变、震源/接收点跨越界面、改变拓扑不在此
局部导数契约内。当前支持一阶结构导数，显式差分模式仍可作为验证或后备。

## PyTorch / JAX

PyTorch `grtm.torch.Forward(..., grid=grid, vjp_method="adjoint")` 提供一阶 backward；
GPU 正演或 GPU 张量也使用 CPU 结构反向。暂不支持高阶导数、torch.compile、torch.func/vmap。

JAX `make_forward(..., grid=grid, derivative_mode="vjp", vjp_method="adjoint")`
提供不构建完整结构 Jacobian 的一阶反向 `grad`、`jit` 和顺序 callback `vmap`，
backward 会重新计算基准场。该模式不支持 JAX `jvp`/`jacfwd` 和高阶导数。
默认 `derivative_mode="jacobian"` 保留原来的稠密 Jacobian 路径。
完整例子见[英文指南](RESEARCH_PREVIEW.md#pytorch-and-jax)。

## 波数积分

```python
native_model = grtm.example_model()  # km、km/s、g/cm³
native_model.update(log2_samples=4, dt=0.4, distances=[10., 30.],
                    t0=[0., 0.], ptam=0)
with solver.wavenumber_kernel(native_model, frequency_index=4) as kernel:
    samples = kernel([0.03, 0.1, 0.3])  # [节点, 接收点, 30]
    result = kernel.integrate(2., method="gauss", atol=1e-9, rtol=0)
    assert result["certified"] is False
```

- `gauss`：自适应成对 Gauss 规则，返回有限区间误差估计。
- `levin`：Bessel–Levin 配点，零点附近用 Gauss，返回采样残差估计。
- `shanks`：相邻面板批量采样、补偿求和、带分母保护的迭代三点变换。

Gauss/Levin 超出面板预算会报错。Shanks 用 cutoff/panel_width 定义工作预算，
其 atol/rtol 不是停止容差；`panels_per_batch=32` 控制批量，`sequence_levels`
记录降阶情况，`transformed_input_error_estimate` 仍是估计，不能认证无限尾项。
`coefficients(k)` 要求 k>0，接近零时系数有强抵消，应优先直接采样。

`forward_quadrature()` 可进一步合成位移等物理输出，高层仍默认 SI。
**cutoff 始终以 km⁻¹ 为单位**，可传入标量或每个非负频率的截断值。
atol 控制原始频域通道，不能解释为米或帕的误差容差。
原生 Levin 包装器按绝对容差分配两个子区间，rtol 不放宽该预算。

原生积分均返回 `certified=False`；`require_certified=True` 会拒绝执行。
同深度直接场可能需要 Abel 解释和解析扣除，专用扣除路径尚未实现。
任意有限 Q 模型可能不满足所需材料条件，有限波数极点仍需检查。
**自适应积分尚无 PyTorch/JAX backward**，不可与固定网格 JVP/VJP 混用。

## 有限范围的认证

```python
from grtm.quadrature import certified_hankel_reference, psv_high_k_bounds
reference = certified_hankel_reference("1", "2", power=1, atol="1e-12")
certificate = psv_high_k_bounds(native_model, 4, relative_radius=1e-4)
```

前者仅认证解析族 `k**power * exp(-a*k) * J_order(r*k)`，a>0，power/order 为非负整数。
后者认证材料盒和整个高波数区间上的归一化 P–SV 矩阵与分组递推逆矩阵，深度固定；
不认证原生采样值、有限波数极点或完整积分。不符合材料类的输入会被拒绝。
不同接口的 certified 标记必须结合各自的范围理解。

请记录版本、模型、单位、网格/截断、后端和容差，并检查收敛性。
极长周期结构梯度仍未充分验证；旧 FK 基准不代表本次预览版的新测速结果。
