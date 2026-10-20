# GRTM 面波频散与半解析敏感核

[English](SURFACE_WAVES.md) | 简体中文

此接口从本项目的层状介质 GRTM 数学核扩展得到，计算 Rayleigh / Love 波的相速度、群速度和逐层敏感核。CPU 与原生 CUDA 共用 FP64 / complex128 数学实现。**生产计算不调用 disba，不扰动每个模型参数后重新寻根。** disba 只用于独立对照和测速。

## 安装与最小调用

[0.6.0.dev4 二进制预览版](https://github.com/cangyeone/seismicx-grtm-community/releases/tag/v0.6.0.dev4)
提供预编译 wheel；稳定 PyPI 0.5.1 不含本功能。下载与 Python/平台匹配的文件后，
执行 `python -m pip install /path/to/downloaded.whl`。
CUDA 接口另需 `python -m pip install 'torch>=2.4'` 及兼容的 NVIDIA 驱动。
Linux 预览包目标计算能力为 12.0，已在 RTX 5090 验证。
macOS 面波接口当前使用 CPU，包内 Metal 后端用于点源格林函数。

有完整源码访问权的维护者也可按下面的方式构建。

这是本工作区新增的源码功能，已有 PyPI 0.5.1 不包含它。在源码根目录安装：

```sh
python -m pip install '.[surface]' \
  -Ccmake.define.GRTM_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120
```

`120` 是验证使用的 RTX 5090 架构；其他 GPU 应选择相应架构。编译需要支持该架构的 CUDA Toolkit / nvcc。非标准工具链可设置 `CMAKE_CUDA_COMPILER` 和 `CUDAToolkit_ROOT`。CPU 源码安装用 `python -m pip install .`，macOS 可加 `-Ccmake.define.GRTM_ENABLE_METAL=OFF`；CPU 的 NumPy 接口不要求 PyTorch。

```python
import numpy as np
from grtm import SurfaceWave

# 与 GRTM 一致：顶深、Vp、Vs、密度。最后一层是半空间。
# native 单位：km、km/s、g/cm3；周期为秒。
layers = np.array([
    [0.0, 3.2, 1.65, 2.1],
    [0.8, 5.6, 3.15, 2.6],
    [5.0, 6.7, 3.80, 2.9],
    [25., 8.1, 4.60, 3.3],
])
periods = np.geomspace(2, 40, 16)
sw = SurfaceWave(layers, device="cuda:0")

phase = sw.dispersion(periods, wave="rayleigh", mode=0, kind="phase")
k = sw.sensitivity(periods, wave="rayleigh", kind="phase",
                   parameter="velocity_s")
print(phase["velocity"])  # [period]
print(k["kernel"])       # [period, layer]，dc/dVs

group = sw.sensitivity(periods, wave="love", kind="group", parameter="all")
print(group["kernel"].shape)  # [period, layer, 4]
```

也可直接调用 `grtm.surface_wave_dispersion(layers, periods, device="cuda:0")` 和 `grtm.surface_wave_kernels(...)`。完整可运行例子：

```sh
python examples/surface_wave_cuda.py --device cuda:0
python examples/surface_wave_cuda.py --device cpu
```

## 输入、输出和参数含义

- `layers` 支持 `[L,4]`、`[L,6]` 或 `[B,L,4/6]`；也接受含 `layers` 的 GRTM 模型字典。默认第一列为顶深，首层顶深固定为零，后续顶深严格递增。Q 列不参与此弹性计算。
- 使用 SI 模型时设置 `units="si"`：顶深 m、速度 m/s、密度 kg/m3。周期始终为 s，输出速度和深度遵循输入单位。
- `parameterization="thickness"` 接受 `[h,Vp,Vs,rho]`；各有限层厚度必须为正，半空间厚度被忽略。可以用 `SurfaceWave.from_thickness(h, vp, vs, rho, device=...)` 直接承接 disba 风格数组。
- `wave="rayleigh"` / `"love"`，`kind="phase"` / `"group"`，`mode=0` 是基阶，`mode=1,2,...` 是按相速度排序的高阶模态。输入周期可不排序，输出保留输入顺序。
- 单模型速度为 `[P]`，单参数敏感核为 `[P,L]`；批量模型对应 `[B,P]` 和 `[B,P,L]`。即使只输入一个周期，也保留周期维。`parameter="all"` 增加末维，顺序为 `thickness, velocity_p, velocity_s, density`；也可传这些名称组成的 tuple/list。
- CPU NumPy 输入得到 NumPy 数组；CUDA 输出为设备上的 PyTorch Tensor。`device="cuda:1"` 和当前 CUDA stream 均受支持。选择 CUDA 后出错会报告，不会悄悄换到 CPU。
- `velocity`、`phase_velocity`、`kernel`、`depth`、`valid`、`residual` 和参数元数据随结果返回。`normalized=True` 返回无量纲对数敏感核 $m_j/v\,\partial v/\partial m_j$，默认返回有量纲导数。

| 参数 | 计算的导数 | 约定 |
|---|---|---|
| `thickness` | 改变一个有限层厚度 | 下部所有界面随之平移；半空间厚度核为零 |
| `top` | 移动一个 GRTM 界面顶深 | 顶深零固定；内界面核为相邻厚度核之差 |
| `velocity_p` | 单层 Vp 独立变化 | Love 波严格为零 |
| `velocity_s` | 单层 Vs 独立变化 | Vp、密度保持独立固定 |
| `density` | 单层密度独立变化 | 不自动施加速度—密度经验关系 |

如果参数间有经验关系或控制点基函数，可通过 PyTorch 在外部建立映射，反演梯度会沿这些映射继续传播。
返回的是逐层参数核，不能直接当成连续深度核；有限层速度核除以层厚才对应层平均的每单位深度核，半空间不作这种换算。

## 与原 GRTM 的关系

数学入口是 `grtm_all/grtm_c/src/popm/func/eigmatrix.c` 的 `mat_PSV_E` 和 `grtc.c` 中的层间连续条件、衰减传播与半空间边界。新增共享文件 `src/surface_wave.hpp` 直接使用相同的 P–SV 本征矩阵、SH 应力关系和模态方向符号。

面波是该层状体系在自由表面条件下的齐次解，也对应格林函数的极点。直接用表面阻抗行列式做实轴寻根会遇到位移极点，两个符号变化还可能被同一网格区间抵消。这里把**同一 GRTM 层间映射**写成衰减半空间子空间的外积传播：Rayleigh 波传播六个二阶子式，Love 波传播位移—应力状态；每层做正实数归一化。传播指数的实部不增长，避免厚层/短周期的指数溢出，且频散函数不含位移极点。

GRTM 本征矩阵的奇偶结构使界面映射可分解为两个 2×2 解算，而不逐点做一般 4×4 求逆。原有点源格林函数、接收函数、C/CUDA/Metal 接口没有被替换。

这不是移植 disba 的 Dunkin / fast-delta / CPS 程序。disba 不属于本接口运行依赖。

## 半解析微分

设频散条件为 $F(c,T,m)=0$，且目标为孤立简单根 $F_c\ne0$。先求一次相速度根，再计算

\[
c_{m_j}=-F_{m_j}/F_c,\qquad c_T=-F_T/F_c,
\]

\[
U=\frac{c^2}{c+T c_T}.
\]

群速度核需要混合二阶导数：

\[
c_{Tm_j}=-\frac{F_{Tm_j}+F_{Tc}c_{m_j}}{F_c}
 +\frac{F_T(F_{cm_j}+F_{cc}c_{m_j})}{F_c^2},
\]

\[
U_{m_j}=\frac{2c c_{m_j}(c+T c_T)-c^2(c_{m_j}+T c_{Tm_j})}
{(c+T c_T)^2}.
\]

所有 $F$ 的偏导都通过本征矩阵、界面解算和传播递推的解析链式法则计算。实现使用 dual / hyper-dual 数：$\epsilon_a^2=\epsilon_b^2=0$，混合项系数直接给出 $F_{ab}$。它没有参数扰动步长，也不通过两次相速度之差估计群速度。有限差分只出现在测试与验证脚本中。

CUDA 将模型、周期、搜索点及层参数任务并行分配。当前是逐参数的解析切线传播；它避免了逐参数重新寻根，但不宣称全 Jacobian 的层数复杂度已经降为线性。

## 自动微分和反演

```python
import torch
from grtm import SurfaceWave

m = torch.tensor(layers, dtype=torch.float64, device="cuda:0",
                 requires_grad=True)
sw = SurfaceWave(m)
c = sw.phase_velocity(periods)
loss = ((c - observed_phase_velocity)**2).mean()
loss.backward()
print(m.grad)  # [layer, top/Vp/Vs/rho]
```

`group_velocity()` 同样支持模型一阶梯度，内部群速度梯度使用上面的混合二阶频散导数。可以多次调用同一算子并分别 backward。原生自定义 backward 不支持再次自动微分；显式 `sensitivity()` 结果也是一阶敏感核数据，不带二阶自动微分图。周期作为固定输入。

## 寻根、缺失模态及适用范围

- 仅支持水平、各向同性、弹性固体，要求正密度、Vs>0、Vp>$\sqrt{4/3}$ Vs；不包含水层、复速度衰减或泄漏模态。
- 搜索截断在半空间 Vs 以下。半空间上的均匀固体有 Rayleigh 基阶，没有 Love 模态；缺失模态是正常物理结果。
- `dc` 是速度搜索最大基础间距，默认原生单位为 0.005 km/s，SI 下对应 5 m/s。它不控制敏感核差分。附加网格按每层垂直慢度加密，处理紧贴层内 Vs 的模态；小规模网格规划和排序在主机端完成，频散函数、寻根细化和导数都在所选后端计算。
- `root_tolerance` 默认对应 $10^{-10}$ km/s；还检查归一化频散残差，不能仅凭区间缩小把极点当根。
- 默认 `missing="raise"`；高阶模态会随周期截止，可设 `missing="nan"`，检查 `valid` 并保留周期对齐。不得把缺失值替换成零。
- 近截止、模态极密或根近简并时，应减半 `dc` 检查模态识别，并检查残差与模型扰动后的模态连续性。临界速度处的奇异本征基使用极小确定性偏移作极限评估，不承诺在精确简并点存在普通导数。
- `max_grid_points=20001` 防止无界分配；层很厚、周期很短或层数很多时可明确提高限制。有限网格不是完整模态计数的数学证明。

## 复核与基准

```sh
python -m pip install disba==0.7.0 pytest  # 仅验证/测速需要
python -m pytest tests/test_surface_wave.py -q
python examples/validate_surface_wave.py --device cuda:0 --output validation.json
python examples/benchmark_surface_wave.py --device cuda:0 --output benchmark.json
```

测试包括均匀半空间 Rayleigh 解析解、两层 Love 独立频散方程、厚层短周期漏模态案例、相/群速度所有参数的独立重寻根差分审计、密度整体缩放不变性、层厚—周期缩放关系、SI/原生单位一致性、批量模型、顶深梯度、多 GPU/非默认 stream 及 autograd。

初次本地验证使用 16 个六层模型（半数含低速层），与 disba 0.7.0 默认 Dunkin 比较 394 个基阶/第一高阶速度，最大绝对差约 $2.44\times10^{-6}$ km/s。一次全部参数的独立重寻根差分审计中，最大绝对核差约 $2.51\times10^{-8}$。这些数值仅描述该测试集合，不是对所有层状模型的误差界；disba 本身也有寻根误差。

测速脚本比较完整的 **Vs 敏感核计算**，包含本方法的寻根与求导，并同步 GPU 计时。CPU 对照使用 disba 默认单边 2.5% 参数扰动；Numba 首次编译和算子构建与常态调用时间分开。单模型延迟、批量吞吐、相速度核与群速度核分别报告；结果保存在本工作区 `optimization/surface-wave-20261020/`，不将正演吞吐直接当成反演加速比。

### 最终 CMake 构建实测

RTX 5090、Threadripper PRO 3955WX，32 层、12 周期、16 模型、Rayleigh 波 **Vs 核**。各项是完整批次的同步主机耗时，已预热，重复两次取中位数；构建/模型搬运另记。硬件共享，未停止其他任务，因此这些是本次观测值。

| CPU 对照 | 核类型 | CPU (s) | CUDA (s) | 比值 |
|---|---|---:|---:|---:|
| disba 默认差分 | 相速度 | 6.804 | 0.248 | 27.4× |
| disba 默认差分 | 群速度 | 13.646 | 0.332 | 41.1× |
| 同一 GRTM 半解析数学核，单线程 | 相速度 | 3.969 | 0.248 | 16.0× |
| 同一 GRTM 半解析数学核，单线程 | 群速度 | 9.153 | 0.332 | 27.6× |

前两行同时体现算法和设备的差别，后两行固定数学实现后比较 CPU/CUDA。原始重复值分别在 `final-benchmark.json` 和 `final-same-math-benchmark.json`。8/32 层、1/16/64 模型的初始扫描结果另存于 `benchmark.json`，不把批量比值用于单模型延迟。

同一个固定种子模型的全部层参数核，以独立重寻根中心差分为审计参照，得到以下最大绝对差（原生单位）：

| 波型 / 核 | GRTM 半解析 | disba 默认差分 |
|---|---:|---:|
| Rayleigh / 相速度 | 2.51e-8 | 2.81e-2 |
| Rayleigh / 群速度 | 1.86e-8 | 3.31e-2 |
| Love / 相速度 | 8.39e-9 | 2.91e-2 |
| Love / 群速度 | 1.27e-8 | 6.32e-3 |

这张表只比较固定审计模型中的导数近似误差。disba 使用默认 2.5% 单边参数扰动，群速度还含默认频率差分；不把该结果当成所有模型的误差界，也不推断地质反演精度。数据见 `cuda-validation.json`。

纯 CPU 和纯频散 CUDA 补测见[完整速度报告](SURFACE_WAVE_BENCHMARKS.zh-CN.md)，其中同时报告未提速的情况。

## 0.6.0.dev4 寻根优化

最新结果见[三类正演与微分的 CPU/CUDA 测试报告](PERFORMANCE_0.6.0.dev4.zh-CN.md)。
上方 0.6.0.dev3 数据属于优化前的历史结果。

`search="auto"` 从同一 GRTM 特征矩阵推导五分量实数试探算子，使用有括区保护的插值和原复数方程验证，保持原根容差。Rayleigh 变分下界用于跳过无根速度范围。CPU 续接仅用于温和、正常递增结构与密集周期；低速层、强突变、高阶模态和稀疏周期保留完整搜索。`search="scan"` 保留原复数全扫描，便于独立复核。

群速度使用实数 GRTM 递推的解析 log-c/log-T 导数，不使用频率差分；病态状态回退到原复数实现。模型敏感核保留原复数双数/超双数实现。CPU 默认 `threads=1`，`threads=16` 并行不同模型及其核；CUDA 计算始终在 GPU 执行。
