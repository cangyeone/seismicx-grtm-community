# Python forward、批处理和导数接口（0.4）

[English](PYTHON_FORWARD.md) | 简体中文

当前发行名为 `seismicx-grtm`（SeismicX GRTM），导入及命令行用 `grtm`。旧包迁移与安装见 [README](INSTALLATION.zh-CN.md)。

## 安装和后端

```bash
pip install .                       # C / CPU；macOS 默认同时编入 Metal
pip install '.[torch,jax]'           # 可选框架依赖
pip install . -Ccmake.define.GRTM_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120
```

CUDA 构建需要 NVIDIA CUDA toolkit。`Solver(backend="cuda")` 显式选择 GPU；
CPU wheel 在无 GPU 的机器上独立使用。CUDA wheel 同时含 CPU 后端，GPU 错误会直接抛出。
NumPy 是基本依赖，导入 `grtm` 不会导入 Torch/JAX。框架自身支持的 Python 版本由其发行版决定。
远程 RTX 5090 对应 architecture 120；其他 GPU 应选择其实际架构。
macOS 可选 `Solver(backend="metal")` 或 `"mps"`，二者选择同一个自定义混合 Metal 引擎。
主机 FP64 层间传播与 GPU 浮点对积分的分工、PyTorch MPS 张量及实测见 [APPLE_METAL.md](APPLE_METAL.zh-CN.md)。

## 模型、震源、坐标与单位

高层 `forward` 和 `ForwardOperator` 默认 `units="si"`：

| 输入 | SI | `units="native"` |
|---|---|---|
| layers 每行 | `[top_depth_m, Vp_m_s, Vs_m_s, density_kg_m3, Qp, Qs]` | `[km, km/s, km/s, g/cm³, Qp, Qs]` |
| source_depth / receiver_depth / distances | m | km |
| dt / t0 | s | s |
| moment_tensor | N m | `10¹⁸ N m` 为 1 |
| force | N | `10¹⁵ N` 为 1 |

`Solver.compute()`、`write_input()` 和原数值文件保留既有 native 单位约定；
`example_model()` 返回 native 模型，`example_model(units="si")` 返回 SI 模型。
物理 forward 和梯度要求 0.3 原生库（含本版矩张量物理修正）；外部旧库仍可调用 compute()。
顶界面深度必须从 0 严格递增，最后一层为半空间，Vp > Vs > 0，密度和 Q 必须为正。
暂不支持流体层、零距离或可变层数的可微接口。

坐标为 **NED**（北、东、下），方位角从北顺时针，角度全部为度。
六分量矩张量顺序为 `(Mnn, Mee, Mdd, Mne, Mnd, Med)`；也接受对称 3×3 矩阵。
正 rake 表示逆断层滑动。`strike_dip_rake` 生成双力偶，矩张量也支持任意对称源。
必须提供 moment_tensor、sdr 或 force 三者之一。

```python
import grtm
model = grtm.example_model(units="si")
solver = grtm.Solver(backend="cpu")       # "c" 是同义词；也可以 "cuda"
mt = grtm.strike_dip_rake(20, 40, 60, scalar_moment=1e15)
result = solver.forward(
    model, moment_tensor=mt, azimuth=[0, 45, 90],
    fields=("displacement", "gradient", "strain", "stress"), threads=8,
)
# 或 solver.forward(model, sdr=(20,40,60), scalar_moment=1e15, ...)
```

返回位移 `[distance, sample, 3]`，gradient/strain/stress `[distance, sample, 3, 3]`。
`result["spectrum"][field]` 是同一物理场的复频谱，`dt/t0/distances` 提供采样信息。
`coordinates="rtd"` 使用径向、方位向、向下；默认 `"ned"` 同时旋转矢量和张量。
梯度定义为 `gradient[..., i, j] = ∂u_i / ∂x_j`；应变为对称梯度，剪切分量为张量剪应变，
不乘工程剪应变的 2。应力采用拉伸为正的约定。

## 时间函数与应力

底层是脉冲格林函数。给定 N m 的源幅度而**不提供 stf**，返回的是脉冲响应核，
位移标度为 m/s、应变为 1/s、应力为 Pa/s，`response="impulse_green"`。
不能将这些核直接标成普通位移和静态应力。

`stf` 是以 dt 采样、从 t=0 开始的**无量纲矩张量/力时间历史** h(t)；
实际源历史为 M h(t) 或 F h(t)，不是归一化的矩率。程序做因果线性卷积并乘 dt，
保留前 n_samples 点。这时位移为 m、应变无量纲、应力为 Pa，`response="history"`。
要使用矩率历史，先按 dt 积分成矩历史。`stf=[1/dt]` 可复现脉冲核的数值。
卷积要求所有 t0=0；已有偏移窗口会缺失因果起始段，程序拒绝这种组合。
有限时间窗、截频和周期积分仍需按研究问题检查收敛。

```python
import numpy as np
n = 1 << model["log2_samples"]
t = np.arange(n)*model["dt"]
history = np.minimum(t/0.5, 1.0)       # 最终矩 M0 的线性上升历史
result = solver.forward(model, moment_tensor=mt, stf=history)
print(result["field_units"])          # m、1、Pa
```

应力默认 `constitutive="kernel"`，与现有核的材料定义一致：
μ = ρ Vs0² 保持实数，复速度 a(ω)、b(ω) 按 Q 色散模型计算，
λ(ω) = μ[(a(ω)/b(ω))² − 2]，σ = λ tr(ε) I + 2με。
程序在底层使用的复频率上施加本构关系，再按相同阻尼 FFT 转回时间域。
`constitutive="elastic"` 使用参考 λ=ρ(Vp0²−2Vs0²)，是有限 Q 情况下的明确近似。
本构参数取接收点所在层；接收点恰在材料界面时取上侧，与原内核一致。

## 大量 Earth models

```python
# 模型可以由生成器不断提供；最多 workers 个计算任务在途，保持输入顺序。
for result in solver.iter_forward_batch(
    models, workers=4, threads=4, moment_tensor=mt,
    fields=("displacement", "stress"),
):
    save_one(result)

all_results = solver.forward_batch(models, moment_tensor=mt, workers=4, threads=4)
bases = solver.compute_batch(native_models, workers=4, stack=False)
```

`stack=True` 堆叠为 `[model, distance, sample, ...]`，要求形状和元数据一致；
异构模型可以 `stack=False` 或使用流式接口。失败抛出 `BatchError`，包含模型 index。
默认 CPU 每个模型按 workers 分配线程预算，显式 threads 不改写。
GPU 默认 workers=1，可指定 device；批处理当前逐模型调用 GPU 内核。
这不是在一个 GPU kernel 中融合全部 Earth models。完整结果的内存随模型数增长，
大量数据建议使用 `iter_*` 接口逐个消费。

统一张量接口还支持一次输入 `[B,L,6]`，以及共享 `[6]` 或逐模型 `[B,6]` 的震源：

```python
op = solver.operator(model, field="displacement", azimuth=45)
y = op(earth_layers_batch, moment_tensor_batch)  # [B,D,N,3]
```

## 结构参数 Jacobian / sensitivity

```python
layers = np.asarray(model["layers"], dtype=np.float64)
op = solver.operator(
    model, field="stress", domain="signal", azimuth=45,
    parameters=["vp", "vs", "rho", "qp", "qs", "depth"], threads=8,
)
j = op.jacobian(layers, mt, check_step=True)
# j["layers"]: output_shape + [L,6]; j["moment_tensor"]: output_shape + [6]
# j["parameters"]: 实际求导的 (layer_index, name)
# j["relative_change"]: 与独立中心差分的相对差异诊断
layer_gradient, source_gradient = op.vjp(layers, mt, cotangent)
```

默认 **`method="analytic"`**：从共享 C 数学核自动生成一阶切线代码，以链式法则传播
模态平方根、复 Q 速度、Lamé 参数、界面反射透射、层间指数、震源项及积分的导数。
这是数值波数积分 + 解析链式导数的半解析方法，求导过程不扰动模型、不依赖差分步长。
`jacobian()` 中每个选中的结构参数运行一次切线积分；矩张量的线性梯度复用基函数。
常规频率采用 double，准静态病态频率采用 long double；线程内复用工作区。
应力梯度同时包含位移导数和接收层本构参数的直接导数。

0.6.0.dev1 开发版还提供一次方向积分的 `jvp()`、`linearize()` 及 PTAM=0 时的
原生反向 `vjp_method="adjoint"`，详见[预览版接口指南](RESEARCH_PREVIEW.zh-CN.md)。
PyPI 0.5.1 仍使用原有逐参数切线 VJP；新版 `auto` 在支持时对至少 8 个所选参数使用反向算法。

`method="finite_difference"` 是显式可选的中心差分方法，每个参数两次求解。
`relative_step` 默认 1e-3，`absolute_step` 默认 0，h=max(|p| relative_step, absolute_step)。
`check_step=True` 对所选方法额外运行 h/2 中心差分作独立校验；它有额外计算开销，
`relative_change` 是一致性诊断，不是物理积分误差界限。
解析方法的 `steps` 全为 0。未选参数和固定的第一个界面深度返回 0，明确表示未求导。

两种方法都冻结基础模型的积分周期 L 和每个频率的**整数波数步数**，
避免模型变化导致离散积分截断跳变。`op.freeze_grid(layers)` 可用于外部验证：

```python
grid = op.freeze_grid(layers)
y_plus = op(layers_plus, mt, grid=grid)
y_minus = op(layers_minus, mt, grid=grid)
```

导数保持层数、源/接收层和数值分支的拓扑固定，不对步数取整或 PTAM 极值选择求导。
界面深度与源/接收点重合时，相关深度导数拒绝计算；差分跨界也会拒绝。
PTAM 的导数沿基础模型选定的极值/收敛分支传播，切换点不光滑，应做一致性检查。
极端长周期的结构梯度尚未通过一致性验证，具体算例和诊断见
[验证报告](FORWARD_VERIFICATION.zh-CN.md)，不能仅靠启用扩展精度判断可靠性。
只提供一阶导数。Jacobian 是稠密输出；切线 VJP 逐参数收缩，避免保存完整结构 Jacobian。
解析结构切线与新版反向梯度都运行在 **CPU**，包括选择 CUDA/Metal 的 forward。
矩张量导数是 NumPy 线性合成。不得将结构梯度计时称为 GPU 原生反向传播。

## PyTorch

```python
import torch
from grtm.torch import Forward
f = Forward(model, backend="cpu", field="displacement", azimuth=45,
            parameters=["vp", "vs", "rho"], threads=8)
earth = torch.tensor(layers, dtype=torch.float64, requires_grad=True)
source = torch.tensor(mt, dtype=torch.float64, requires_grad=True)
y = f(earth, source)
loss = (y - observation).square().mean()
loss.backward()
# earth.grad 和 source.grad；可用 f.forward_sdr(earth, strike, dip, rake, M0)
```

`grtm.torch.strike_dip_rake` 保留框架原生角度梯度。
支持 `[B,L,6]`、共享/逐模型震源，以及复频谱输出。
输入需要相同实浮点 dtype/device，输出留在输入 device；CPU/CUDA 建议 float64。
`backend="cuda"` 是显式选择原 CUDA 求解器，张量与原生库之间存在 host 拷贝。
`backend="mps"` 选择混合 Metal 求解器；PyTorch 的 MPS 张量使用 float32，复输出为 complex64。
张量的 `device="mps"` 与求解器 `backend="mps"` 分别控制张量位置和原生计算引擎，需要分别设置。
结构 backward 使用上述 CPU VJP；只对 `parameters` 中指定的列计算梯度。
0.6 开发版可指定 `vjp_method="adjoint"`。
本适配器支持普通 autograd 的一阶 backward，不支持高阶梯度、torch.compile 或 torch.func/vmap。

## JAX

```python
import jax
import jax.numpy as jnp
from grtm.jax import make_forward
jax.config.update("jax_enable_x64", True)  # 应用自行决定精度，包不修改全局设置
f = make_forward(model, backend="cpu", parameters=["vp", "vs", "rho"], threads=8)
y = jax.jit(f)(jnp.asarray(layers), jnp.asarray(mt))
grad_earth, grad_source = jax.grad(
    lambda earth, source: jnp.sum(f(earth, source)**2), argnums=(0,1)
)(jnp.asarray(layers), jnp.asarray(mt))
```

通过 `pure_callback` + `custom_jvp` 支持 jit、一阶 grad/JVP/jacfwd、顺序 callback 的 vmap
和显式 Earth batch。`grtm.jax.strike_dip_rake` 保留角度梯度。
这不是 XLA 原生求解器，也不支持高阶导数或对几何、dt、STF 配置自动求导。
JAX 默认导数 callback 构造稠密 Jacobian，内存为 O(output_cells × L × 6)，未选列仍占据数组位置。
0.6 开发版增加 `derivative_mode="vjp"`，提供无完整 Jacobian 的一阶反向模式；
该模式不支持 `jvp`/`jacfwd`。固定优化网格的构造参数 `grid=` 见[新指南](RESEARCH_PREVIEW.zh-CN.md)。
PyPI 0.5.1 中，很大的输出建议 NumPy/PyTorch 的 VJP。不要假设 JAX vmap 将模型融合成 GPU kernel。

框架机制参考：[PyTorch 自定义 autograd](https://docs.pytorch.org/docs/stable/notes/extending.html)、
[JAX callbacks](https://docs.jax.dev/en/latest/notebooks/external_callbacks.html)、
[JAX 自定义 JVP](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html)。
NED 约定参考：[Pyrocko moment tensor](https://pyrocko.org/docs/current/library/examples/moment_tensor.html)。
