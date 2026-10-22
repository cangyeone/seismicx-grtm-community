# 地表力激发的频散能量图

**0.6.0.dev12** 在 dev11 正演基础上增加解析伴随、PSD 和有限阵列；接口包括 `SurfaceWave.dispersion_image` 和一步调用接口 `surface_wave_image`。在频率—相速度网格上施加地表谐波力（应力边界条件），用 GRTM 求解受迫波场，输出复位移及其幅值平方。CPU、CUDA 共用 FP64 公式，CUDA 将不同模型和网格点融合到一次原生 kernel 启动中。

```python
import numpy as np
from grtm import SurfaceWave

# 地层顶深度 km，Vp/Vs km/s，密度 g/cm³；末层为半空间。
layers = [[0., 3.6, 2., 2.5], [2., 5.5, 3.1, 2.7], [10., 8., 4.5, 3.3]]
sw = SurfaceWave(layers, device='cpu', missing='nan')  # CUDA 改为 'cuda:0'
f = np.linspace(.025, .5, 121)       # Hz，严格递增
c = np.linspace(1.7, 4.49, 701)      # km/s，严格递增
image = sw.dispersion_image(f, c, wave='rayleigh')
# response: [频率, 速度, 2]，complex128，最后一维为 [径向, 垂直]
# power:    原始 |u_vertical|²
# energy:   每个频率分别归一化后的 power
# valid:    响应是否有限且数值求解成功，不是“模态是否存在”的掩码
phase = sw.phase_velocity(1/f, wave='rayleigh', mode=0)
```

`examples/dispersion_energy_image.py` 可输出叠加频散曲线的 PNG/PDF、原始复响应/幅值平方/掩码 NPZ，以及参数 JSON。画图时使用 `energy.T`，横轴频率、纵轴相速度。对数显示的截断下限只用于画图，不修改原始数据。

接上面的示例直接保存图片（需要 matplotlib）：

```python
import matplotlib.pyplot as plt
# For CUDA outputs, move only the finished image to the host for plotting.
e = image['energy']
if hasattr(e, 'detach'):
    e = e.detach().cpu().numpy()
plt.pcolormesh(f, c, e.T, shading='auto', cmap='magma')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase velocity (km/s)')
plt.colorbar(label='Normalized displacement power')
plt.savefig('dispersion-image.png', dpi=180)
```

## 震源、分量与单位

- Rayleigh 默认 `source='vertical'`，也可指定 `'radial'` 或复数向量 `[径向, 垂直]`；Love 使用 `wave='love'`，默认横向力，也可传 `[复振幅]`。
- `component` 可选 Rayleigh 的 `'vertical'` / `'radial'`、Love 的 `'transverse'`，或 `'total'`（位移各分量幅值平方之和）。无论选择哪个，完整复位移向量都保留。
- 默认振幅为一个单位的**谐波地表牵引应力**：native 为 GPa，SI 为 Pa。位移分别为 km、m；`power` 分别为 km²、m²。跨单位比较必须同时换算震源：1 GPa = 10⁹ Pa。单位应力的响应也就是对应地表柔度列。这是谱域平面波载荷，不是给定距离处、单位为牛顿的点力。
- 模型支持 `[B,L,4/6]`，输出增加 B 维；震源支持共享 `[2/1]` 或逐模型 `[B,2/1]`。本次震源、接收点都位于地表。
- `normalization='frequency'` 按每模型、每频率的速度方向最大值归一化；`'global'` 按每个模型整幅图最大值；`'none'` 不归一化。零震源保持零；失败点保留 NaN 和 `valid=False`，不填补缺模。

## 方程与图的含义

令实波数 `k=2πf/c`、复频率 `ω̃=2πf(1−iη)`，其中 `η=damping>0`。利用现有 GRTM 特征矩阵和界面连续条件，从满足衰减/辐射条件的半空间向上递推，只使用有界的 `exp(−qh)` 传播因子。在地表得到位移矩阵 U、应力矩阵 T；按表面外法向处理载荷后求解 `T a = f`，再算 `u = U a`。自由表面对应 `det(T)=0`；受迫解保留了震源激发强弱和接收分量权重。没有把行列式倒数当作波场能量。`boundary_residual` 是地表小系统的相对残差，不是完整的条件数或误差估计。

默认 `damping=0.002` 用来展宽极点，**不是介质 Q 衰减**。更小的值需要更密的速度网格；有限展宽和网格采样会影响峰宽、峰高及峰值位置。这里“能量”指位移幅值平方，不是介质体积分总能量或能流。某阶模态可能因震源/分量选择而很弱；没有被俘获模态的位置，也可以有平滑的受迫响应。它是谱域理论响应，阵列 MASW/F–J 图还会受距离积分、孔径、震源谱和处理方式影响。

阻抗与自由表面条件的关系可参见 [Jiang 等（2025），公式 3–5](https://link.springer.com/article/10.1186/s40623-025-02164-2)；代码沿用本项目 GRTM 特征矩阵和界面映射。

## 支持范围与验收

当前适用于各向同性、弹性固体层和辐射半空间；可选 Q 列与现有频散接口一样被忽略。不包含流体、介质衰减、地下震源/接收点。CPU/CUDA 支持下文描述的一阶图像伴随；原有频散曲线求导和敏感核接口不受影响。

CPU 返回 NumPy，输入 Torch 模型时返回 Torch。CUDA 的数值求解、响应和归一化均留在指定设备/当前流上；输入合法性检查可能同步少量标量。MPS 明确走 CPU FP64 桥接：复响应/复震源保留在 CPU，实数图和掩码返回 MPS（float32），元数据注明计算设备。

测试包含独立组装的完整 P/SV 波动方程、SH 半空间/单层解析解、同层拆分不变性、震源线性、密度缩放、SI 换算、多阶频散峰位置，以及 CPU/CUDA batch 和非默认流。发布附精度、耗时记录；物理正确性不只依赖 CPU/CUDA 相互对比。

## 有限阵列和 source PSD（dev12）

`source_psd` 支持非负标量、`[F]`、`[B,F]`，计算 `PSD×|参考震源响应|²`，复数 `response` 不乘 PSD。提供 PSD 时，定义它为参考震源无量纲时间函数的谱密度（1/Hz），`power` 单位为位移²/Hz；实际力的 PSD 为 `|source|²×source_psd`。默认 `None` 为单位权重。**逐频率归一化会消除正的逐频率 PSD 和公共震源幅值；反演震源功率应使用原始 `power`。** 最大值归一化、绝对值和有效性掩码只在分段意义下可微。

```python
from grtm import DispersionArray

line = DispersionArray(method='fk', field='coherent', length=1.07,
                       n_receivers=108, offset=.05, window='hann',
                       integration_velocities=np.linspace(.3, 10., 2001), max_cache_mb=2048)
line_image = sw.dispersion_image(f, c, array=line, source_psd=np.ones(len(f)))

active = DispersionArray(method='fj', field='coherent',
                         radii=np.linspace(.05, 1.12, 108),
                         integration_velocities=np.linspace(.3, 10., 2001), max_cache_mb=2048)
active_image = sw.dispersion_image(f, c, array=active)

noise = DispersionArray(method='fj', field='noise',
                        positions=np.linspace(0., 1.07, 108),
                        integration_velocities=np.linspace(.3, 10., 2001), max_cache_mb=2048)
noise_image = sw.dispersion_image(f, c, array=noise)
```

主动点源和随机源分别调用、分别输出；同一个 `DispersionArray` 可以在多次反演迭代中复用。上面的积分网格只是接口示例，必须结合模型、频率、展宽和孔径检查**采样间距与积分端点**的收敛。`array=None` 保持 direct response。阵列结果保留 `response/direct_power/direct_valid`，另外返回 `array_response`、适用时的 `receiver_response`、处理后的 `power/energy/valid`、距离、窗函数、积分网格及 `integration_phase_step_max=max(Δk)×max(|r|)`。较大相位步长提示空间振荡采样不足；较小相位步长也不保证解析了窄极点。

积分严格限于所给正波数区间，默认使用显示速度网格；没有补出区间外的波场，也不自动保证完整点源波形收敛。`band_limited=True` 明确记录这一点。

| 输出类型 | 震源单位 native / SI | 接收记录及阵列响应 | power |
|---|---|---|---|
| direct 谱域响应 | GPa / Pa | km / m | 位移幅值平方 |
| 主动线阵 FK | GPa·km / N/m | 复位移 | 波束位移幅值平方 |
| 主动点源 F–J | GPa·km² / N | 复垂直位移 | Hankel 波束位移幅值平方 |
| 随机源 FK/F–J | GPa / Pa，参考谱域载荷 | 期望互谱 | FK 期望功率；F–J 有符号变换的**绝对值** |

主动阵列中保留的 direct `response` 是空间谱位移：FK 单位长度²，F–J 单位长度³；`array_response/receiver_response` 是长度。PSD 乘在功率上，不乘在主动源复振幅上。随机源互谱已包含 PSD。各类单位由元数据记录，direct 功率另有 `direct_power_units`。

采用 `exp(i k x+i ωt)`，主动 FK 先计算
`u(r)=∫[G(k)exp(i k r)+G(−k)exp(−i k r)]dk/(2π)`，再做 `Σw u exp(i k_out r)/Σw`。Rayleigh 负波数支使用物理对称性 `C(−k)=D C(k)D`，`D=diag(−1,1)`；Love 为偶函数。这是二维**线力**实验，不能把绝对振幅直接当作三维点力实验。

主动 F–J 先计算 `u_z(r)=∫J0(kr)G_zz(k)k dk/(2π)`，再计算 `∫w(r)r J0(k_out r)u_z(r)dr / ∫w(r)r dr`。当前支持**地表垂直点力与 Rayleigh 垂直记录**，这里“径向记录”指按震源—台站径向距离排列，不是径向位移分量。`receiver_response[...,0]` 保存这些复数垂直记录。尚不支持水平点力/SH 的轴对称 F–J、VTI 或地下震源。有限孔径和离散采样产生展宽、旁瓣和干涉；不使用行列式倒数替代这些观测过程。

随机源 F–J 明确定义为**各向同性、有限波数带、互不相关的空间 Fourier 载荷系数**。震源极化定义在随波数方向旋转的径向/垂直或横向基底。台站对互谱为
`C(r)=∫J0(kr)P(k)k dk / ∫k dk`，其中 `P=source_psd×|G_selected|²`，或相应向量迹。再使用同一个按径向面积归一化的 J0 变换。`array_response` 保留正负号，`power=abs(array_response)` 保持互谱单位，**没有再次平方**。归一化固定指定波数带内的总载荷功率，改变积分端点也改变该随机源模型。

这个随机源模型不是任意环境噪声的格林函数提取公式，不假定能量均分或 `Im(G)` 关系，也不是某个有限地表区域内独立点源的积分。主动源复记录没有通过丢弃相位变成噪声互谱。Rayleigh 垂直选择对应 ZZ；Rayleigh 径向/Love 选择对应水平向量迹，`total` 对应相应总迹，不是按台站对方向投影的 TT/RR。

F–J 支持 `[N,2]` 台站坐标。等价台站对距离合并，按距离做梯形积分，**不按重复台站对数量加权**；`station_pair_count` 保留原始台站对数。也可通过 `radii=` 输入已整理的观测距离。随机 FK 使用一维单向非相干平面波及阵列响应平方，与二维各向同性随机 F–J 明确区分。

几何、频率/速度网格和 damping 固定。Fourier/Bessel 基函数、阵列处理及其反传在所选 CPU/CUDA 设备完成，主机只处理固定网格缓存键和少量校验信息。`max_cache_mb` 限制估算的基函数/工作空间，不是整个求解器显存上限。NumPy 的 CPU 阵列调用也需要安装可选 PyTorch 依赖。几何单位与模型长度一致；线阵支持 positions，或 length/n_receivers/spacing 中两个一致的参数；窗可选 rectangular/hann/hamming。

用户提供的参考幻灯片用于组织对比图。没有原始输入和独立输出，不能声称复现其中 SPECFEM2D/SASEM/VTI 数值结果。观测方法参考：[阵列响应与多模态干涉](https://academic.oup.com/gji/article/218/1/295/5420372)、[F–J、几何和各向同性源假设](https://academic.oup.com/gji/article/237/1/250/7589741)。

## 解析伴随与能量图反演

```python
import torch
model = torch.tensor(layers, dtype=torch.float64, device='cpu', requires_grad=True)
operator = SurfaceWave(model)  # CUDA 使用 CUDA 张量
prediction = operator.dispersion_image(f, c, array=active, normalization='none')
observed = torch.zeros_like(prediction['power'])  # 替换为相同预处理、单位的观测
valid = prediction['valid'] & torch.isfinite(observed)
loss = .5 * (prediction['power'][valid] - observed[valid]).square().sum()
loss.backward()  # model.grad 保留原模型形状、参数化和单位

grad = sw.image_sensitivity(f, c, observed=observed.detach().cpu().numpy(),
                           field='power', array=active, normalization='none')
# model_gradient、objective、prediction、valid
# 或 cotangent=...，计算 Re(sum(conj(cotangent)*prediction)) 的梯度。
```

地表伴随求解 `Tᴴλ=Uᴴg`，将波场振幅和伴随场与 `dU/dT` 收缩；随后沿有界反射递推反传，收缩每层 Vp、Vs、密度、厚度的解析特征矩阵导数。一次向上传播、一次向下伴随得到全部层参数梯度，不逐参数扰动模型，不构造“全部像素×全部参数”Jacobian。工作量和状态量随层数 O(L) 增长，**不能宣称与层数无关**。

CUDA 跨模型/频率/速度节点计算并在同一 stream 归约梯度；伴随状态及部分梯度工作区按 128 MiB 分块，不意味着整个阵列图显存只有 128 MiB。CPU 使用线程私有梯度后归约。阵列算子、功率和归一化通过 Torch 的解析反向算子连接。只支持**一阶导数**；频率/速度网格、阵列几何和 damping 不求导。

模型、复震源和非负 PSD 可用可训练 Tensor；便捷接口在提供数值 source/source_psd 时返回对应梯度。模型梯度与输入深度/厚度参数化、单位一致；固定顶深度、未使用的半空间厚度，以及弹性模型忽略的 Q 列梯度为零。有效性边界变化不求导。无效观测/预测像素从目标函数排除；阵列积分任一点失败，则该频率的阵列行无效。不存在的模态不填补到训练标签中。
