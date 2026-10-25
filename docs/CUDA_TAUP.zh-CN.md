# CUDA TauP：物理走时、全震相与射线路径

本组件直接求解球对称射线方程，不使用神经网络替代计算，也不包含训练权重。

## ObsPy 接口

源码导入 `from cuda_taup import TauPyModel`；编译包导入 `from seismicx_cuda_taup import TauPyModel`。

距离参数沿用 ObsPy 的**度**，不是 km；地表弧长可转换为 `distance_km / model.model.radius_of_planet * 180 / pi`。深度单位 km，走时单位秒，离源角 0° 向下、90° 水平、180° 向上。

```python
from seismicx_cuda_taup import TauPyModel

model = TauPyModel("iasp91", device="cuda:0")
arrivals = model.get_travel_times(
    source_depth_in_km=10., distance_in_degree=35.,
    phase_list=["p", "P", "Pn", "S", "SKS"],
    receiver_depth_in_km=0.,
)
for a in arrivals:
    print(a.name, a.time, a.ray_param_sec_degree,
          a.takeoff_angle, a.incident_angle)

paths = model.get_ray_paths(10., 120., ["Pdiff", "PKIKP"])
paths.plot_rays(show=False)

pierce = model.get_pierce_points(10., 35., ["P", "S"], add_depth=[100., 250.])
```

默认 `phase_list=("ttall",)`，返回各震相的全部到达并按走时排序。支持 P/S、上行 p/s、Pn/Sn、PP/SS、多次反射、转换波、核内震相、绕射/折射、`kmps` 表面传播，以及原始解析器支持的数字深度震相及震相组。重复射线参数按阴影间隙处理，0°/360° 的绕行到达保留 ObsPy 的重复记录约定。

继承的 `get_travel_times_geo`、`get_ray_paths_geo`、`get_pierce_points_geo` 使用相同的 CUDA 数值求解；地理坐标转换、可选重采样及绘图沿用 ObsPy 的 CPU 实现。路径经纬度需要另装 `geographiclib`，与 ObsPy 相同。

不再有此前首至原型的深度 200 km / 距离 3000 km 限制。输入仍需符合所选球对称模型及 ObsPy 震相语法的有效范围。支持内置模型及自定义 ObsPy `.npz` 模型。`.nd/.tvel` 转 `.npz` 可使用原始 ObsPy `build_taup_model`，模型创建尚未 GPU 化。

## GPU 批处理

```python
import torch
distances = torch.linspace(.1, 179.9, 1024, device="cuda:0", dtype=torch.float64)
model.prepare(10., phase_list=["ttall"])
r = model.get_travel_times_batch(10., distances, phase_list=["ttall"])
print(r.time, r.takeoff_angle, r.incident_angle)
print(r.phase_names, r.phase_index, r.query_index)
print(r.gradient)  # [源深度导数, 地表弧长距离导数]，s/km

rp = model.get_travel_times_batch(10., distances, ["P", "S"], output="path")
print(rp.path_values, rp.path_offsets)  # GPU 上的紧凑路径数组
arrivals_per_station = rp.to_obspy()   # 显式转回 CPU Arrival 对象
```

一个 batch 共用震源深度、接收深度和震相列表，适合一个地震对多个台站计算。改变深度或震相集时需要重新预处理，`prepare` 提供 LRU 缓存。对大量不同震源深度，冷启动成本应单独计入，不能套用同深度批量加速比。`output="pierce"` 可批量返回穿透点；路径批处理的输出本身可能很大，应据显存选 batch 大小。

## CPU 与 CUDA 的边界

| 环节 | 执行设备 |
|---|---|
| 载入模型、深度分层、震相路径解析与缓存 | ObsPy / CPU，一次性预处理 |
| P/S 分层 Bullen 积分、震相曲线构造 | CUDA FP64 |
| 距离搜索、射线寻根、到达筛选、离源角/入射角 | CUDA FP64 |
| 详细路径、pierce 累加与紧凑数组 | CUDA FP64 |
| `TensorArrivals` 数值输出 | 所选 CUDA 设备 |
| `Arrival/Arrivals` 兼容对象、地理坐标转换、绘图 | CPU，按接口需要转换 |

热路径没有调用 CPU `calc_time`、`shoot_ray`、`refine_arrival`、路径或 pierce 求解函数；自动测试将这些 CPU 函数替换成异常后验证 CUDA 三类查询仍成功。CUDA 不可用或内核出错时会报错，不静默退回 CPU 求解。

## 精度与兼容性

默认 `subdivisions=1, search_hidden_roots=False` 保留 ObsPy 的震相采样/到达约定。可选 `subdivisions=8, search_hidden_roots=True` 加密采样并检查区间内部极值，可能返回原始 API 漏掉的有效射线；此选项不保证到达数量与原始 API 相同，也不是任意模型不漏根的证明。

数值积分与寻根采用 FP64。射线参数停止容差为 `min(ray_param_tol, 1e-11) s`，亦可因距离残差达标或浮点分辨率停止。这是停止准则，不是误差上界。`ray_param_tol` 必须为正。

详细路径使用 Bullen 慢度插值。pierce 为兼容 ObsPy，保留其粗曲线插值和**线性速度转向深度**约定，两者不应混为同一数值对象。速度界面处的深度导数具有单侧约定，首至或震相切换处梯度可跳变；没有声称整个震相选择和深度预处理可被自动微分。

对原始 **ObsPy 1.4.2** 的回归验收覆盖 iasp91、ak135、PREM：342 组查询、8447 条配对到达、36 组详细路径/pierce，涵盖 0–600 km 震源、0/50 km 接收点、近距至全球及绕行距离。到达数量全部一致；最大走时差 `7.33e-11 s`，最大离源角/入射角差 `2.51e-4°`。8 项接口测试通过，包括非默认 CUDA stream、自定义序列化模型、地理接口、绘图和禁止 CPU 数值回退。它们是已测范围的回归结果，不是任意自定义模型的全域正确性证明。

另外检查了 1066a、1066b、ak135f_no_mud、herrin、jb、pwdk、sp6 七个内置模型的 21 组查询、629 条到达，数量一致，最大走时差小于 `6.4e-12 s`。

**低速层转换波的原始 API 异常：**将 iasp91 的 77.5/120/165 km 处改成 Vp=7.2、Vs=4.0 km/s 后，原始 ObsPy 在部分 PS/SP 查询中返回的射线参数，交回原始 `shoot_ray` 会偏离目标约 17–61°。CUDA 不将距离不连续处当成根；这些不闭合候选会被过滤。兼容对象的 `cuda_diagnostics`、GPU 结果的 `rejected_candidates` 和运行时提示会明确报告过滤。数值证据保留原始数量差异及每条原始射线的闭合复核，不把该组自定义模型称为“与原始 API 数量完全一致”。

## 本机速度实测

iasp91、震源深度 10 km、RTX 5090、FP64。GPU 输入已在设备，预热后 5 次同步墙钟中位数，排除模型和震相预处理；CPU 使用 8 个预热进程，同一组点，每配置测一次。“预解析 CPU”也缓存震相对象。下表为源码接口计时，安装包另做数值回归。

| 震相 / 输出 | Batch | CUDA (ms) | 原始 CPU 8 进程 (s) | 对原始 CPU 加速 | 对预解析 CPU 加速 |
|---|---:|---:|---:|---:|---:|
| P/p/Pn 走时、角度 | 1024 | 81.9 | 2.961 | 36.2× | 32.6× |
| ttall 走时、角度 | 1024 | 3734.7 | 77.745 | 20.8× | 20.2× |
| ttall 详细路径 | 32 | 299.7 | 7.622 | 25.4× | 25.0× |
| ttall pierce | 32 | 277.4 | 4.750 | 17.1× | 16.7× |

**共享负载条件：**另一项目当时使用 24 个 CPU worker，结束时两张 GPU 的利用率约 81%/78%，CPU load average 约 29。没有停止其他任务。这些结果描述当时的共享服务器，不是独占性能保证。单点 P/p/Pn：CUDA 31.9 ms，原始 CPU 13.4 ms，CPU 更快；全震相单点：CUDA 191.9 ms，原始 CPU 511.8 ms。

## 构建与安装

测试平台为 RTX 5090 / CUDA 12.8 / Torch 2.8。源码可设置其他 CUDA 架构重新编译：

```sh
TAUP_NVCC=/path/to/nvcc TAUP_CUDA_ARCH=sm_120 sh cuda_taup/build.sh
python -m unittest discover -s cuda_taup/tests -v
python cuda_taup/scripts/validate.py --device cuda:0
python cuda_taup/scripts/benchmark.py --device cuda:0 --n 1024
```

编译包依赖 NumPy、CUDA 版 PyTorch 和 `obspy==1.4.2`；地理路径坐标可额外安装 `geographiclib`。研究/非商业许可保持不变，ObsPy 为单独安装的未修改 LGPL 依赖。此组件不是 ObsPy/TauP 官方版本。
