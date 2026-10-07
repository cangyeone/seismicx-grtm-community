# 接收函数：H–κ 初始化到全波形结构反演

[English](RECEIVER_FUNCTIONS.md) | 简体中文

安装反演依赖：`python -m pip install 'seismicx-grtm[rf]==0.5.1'`。合成与 H–κ 叠加使用 NumPy，反演时才加载 SciPy。
```python
import numpy as np
import grtm

model = {"layers": [[0, 6300, 6300/1.782, 2800],
                    [34450, 8000, 4500, 3300]]}
p = np.linspace(0.04, 0.08, 12)/1000  # 水平慢度，s/m
observed = grtm.synthetic_rf(model, p, dt=0.05, n_samples=768, t0=-5)
result = grtm.receiver_function_workflow(
    observed, h=np.arange(28000, 42001, 1000),
    kappa=np.arange(1.6, 1.901, 0.02), vp=6300,
    hk_options={"bootstrap": 100, "seed": 7},
    inversion_options={"fit_window": (1, 26)},
)
print(result["model"]["layers"])
print(result["fit"]["success"], result["fit"]["rms"])
```

流程先用 Ps、PpPs、PpSs+PsPs 做 H–κ 叠加，建立地壳/地幔两层初始模型，
再进行有界最小二乘全波形拟合。默认仅反演地壳厚度和 Vp/Vs；平均 Vp、
密度、地幔性质固定。默认密度为 2800 kg/m³，地幔 Vp/Vs/密度为
8000 m/s、4500 m/s、3300 kg/m³。H–κ 网格端点作为优化边界。
该流程不意味着接收函数能独立确定所有物性。

上例为合成数据演示；可用 `numpy.savez` 保存数组。合成恢复不等同于实际台站验证。

模型行沿用 `[层顶深度, Vp, Vs, 密度]`，也接受附带 Qp/Qs 的六列形式。
层顶从 0 严格递增，末层为半空间。本实现是**弹性**正演，忽略可选 Q 列并
通过返回的 `attenuation="elastic"` 标明。支持水平各向同性固体层、自由表面、
底部上行平面 P 波；所有层须满足 p*Vp<1。尚不支持流体、水层、倾斜、
各向异性、衰减或 S 接收函数。

默认 SI：m、m/s、kg/m³、**s/m**。`units="native"` 为 km、km/s、g/cm³、
s/km。s/degree 必须另行转换；不自动计算球形地球的到时与射线参数。
R 方向**背离震源**，Z 向上；NED 中的 D 要取负转换为 Z。若 beta 为台站指向
震源的反方位角（弧度），R=-N*cos(beta)-E*sin(beta)。
输出 RF 为 `[事件, 时间采样]`，时间零点为直达 P，t0 为输出起始时间。

观测数据可用 `receiver_function(radial, vertical, dt=..., t0=..., slowness=p)`
进行水准稳定反褶积。输入须事先去仪器响应、去趋势、加窗、旋转并截取 P 波窗；
不自动拾取 P 波。共同输入到时在 R/Z 中抵消，t0 控制反褶积输出窗。
Gaussian 采用 exp[-omega²/(4*a²)]，`gaussian=a` 的单位是 rad/s，非 Hz。
归一化使滤波后的垂直自反褶积在零时延处振幅为 1。带限震源和噪声会影响
water-level 正则化，因此实测 RF 与理想脉冲正演可能仍有差异。

新平面波引擎在 NumPy CPU 中运行，按慢度和频率批量解稳定散射递推，
包含界面转换与自由表面多次波，无需点源波数积分。现有 C/CUDA/Metal 点源
引擎和框架接口保持原用途；此 RF 接口尚无 GPU 或 PyTorch/JAX 梯度桥接。
FFT 默认补零到至少 4 倍记录长度；长尾、高反射模型应检查 8 倍补零或更长记录。

H–κ 会将任何事件中任何一种震相超出时间窗的网格设为 NaN；若全无有效
网格则报错。输出 `boundary_maximum` 标明最大值是否贴边，支持事件 bootstrap。
bootstrap 区间仅条件于所用 Vp、权重与震相识别，不包含模型误差。

多层模型可直接调用 `invert_receiver_functions(observed, initial_model, ...)`，
指定 `parameters=((0,"thickness"),(1,"thickness"),(1,"kappa"))` 等。
参数包括有限层 thickness、各层 vp、kappa、density；层号从 0 起。
改变厚度将移动所有更深层顶。Vp 与 kappa 是独立坐标，Vs=Vp/kappa；
固定 kappa 改 Vp 会同步改变 Vs。Q 和层数固定。`bounds` 用参数二元组作键，
边界须含初始值；`prior_std` 可设置围绕初始值的高斯先验。

结构 Jacobian 使用**中心差分**，近边界采用单侧三点差分，与已有点源原生 AD
分开。返回 `success/message`、拟合前后 RMS、`active_bounds`、`data_rank`、
缩放奇异值等。只有满秩、无先验、线性损失的拟合返回局部条件协方差；
该近似忽略 RF 噪声相关、固定 Vp/地幔误差及模型不匹配。数值收敛不代表解唯一。

物理验证包括独立弹性 ODE/矩阵指数边界解对照、均匀半空间、同质分层不变性、
转换波/多次波到时极性、单位、补零、震源抵消、独立到时脉冲 H–κ 和两层/多层
合成恢复。方法出处和完整参数说明见 [英文文档](RECEIVER_FUNCTIONS.md)。
