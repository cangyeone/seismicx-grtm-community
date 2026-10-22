# 地表力激发的频散能量图

**0.6.0.dev11** 增加 `SurfaceWave.dispersion_image` 和一步调用接口 `surface_wave_image`。在频率—相速度网格上施加地表谐波力（应力边界条件），用 GRTM 求解受迫波场，输出复位移及其幅值平方。CPU、CUDA 共用 FP64 公式，CUDA 将不同模型和网格点融合到一次原生 kernel 启动中。

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

当前适用于各向同性、弹性固体层和辐射半空间；可选 Q 列与现有频散接口一样被忽略。不包含流体、介质衰减、地下震源/接收点，以及能量图的梯度。对需梯度的张量显式报错，可在 `torch.no_grad()` 下计算图；原有频散曲线求导和敏感核接口不受影响。

CPU 返回 NumPy，输入 Torch 模型时返回 Torch。CUDA 的数值求解、响应和归一化均留在指定设备/当前流上；输入合法性检查可能同步少量标量。MPS 明确走 CPU FP64 桥接：复响应/复震源保留在 CPU，实数图和掩码返回 MPS（float32），元数据注明计算设备。

测试包含独立组装的完整 P/SV 波动方程、SH 半空间/单层解析解、同层拆分不变性、震源线性、密度缩放、SI 换算、多阶频散峰位置，以及 CPU/CUDA batch 和非默认流。发布附精度、耗时记录；物理正确性不只依赖 CPU/CUDA 相互对比。
