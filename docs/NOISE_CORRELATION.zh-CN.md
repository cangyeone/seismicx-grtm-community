# CPU/CUDA 噪声互相关

[English](NOISE_CORRELATION.md) · [两种谱白化](DATA_PROCESSING.zh-CN.md)

`grtm.noise_cross_correlation` 接收真实记录或合成记录，形状为
`[..., 台站, 采样点]`，前导维度可代表模型、日期或记录 batch。
支持 NumPy CPU 和 PyTorch CPU/CUDA 的 float32/float64，NumPy 整数提升为 float64。
暂不支持 MPS 或半精度；不静默回退 CPU。本接口从 **0.6.0.dev16** 起提供，dev15 只有白化。

```python
import torch
from grtm import noise_cross_correlation

records = torch.randn(32, 65536, device="cuda", dtype=torch.float32)
result = noise_cross_correlation(
    records, dt=0.02, window_samples=4096, step_samples=2048,
    pairs=None, max_lag=5.0,  # 全部 i<j 台站对，最大时延单位为秒
    detrend="constant", taper_fraction=0.05,
    whitening="smooth", smooth_bins=11, freqmin=0.2, freqmax=10.,
    normalization="coeff", stack="mean", nan_policy="omit",
    window_batch_size=8, pair_batch_size=256,
    check_finite=False,       # 输入与参数已在上游校验
)
ccf = result["correlation"]   # [496, 501]，仍在 CUDA
lags = result["lags"]         # 秒，仍在 CUDA
valid = result["valid"]       # 无有效窗口的台站对为 False，CCF 为 NaN
```

## 时延方向及线性相关

台站对 `(i,j)` 定义为 `C_ij[l] = sum_t x_i[t] * x_j[t+l]`。
**正时延表示 j 比 i 晚到**；等于预处理后调用
`numpy.correlate(x_j, x_i, "full")`，也与
[SciPy 的直接相关定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)一致。
交换台站对会反转 CCF。默认生成 `i<j`；`include_auto=True` 加入自相关。
显式 `pairs=[[0,1],[1,0],...]` 保留顺序、重复及反向台站对。

互谱为 `conj(FFT(x_i))*FFT(x_j)`，FFT 补零长度至少为 `2*N-1`，默认取下一个 2 的幂。
过小的 `nfft` 会报错，避免循环相关绕回。`max_lag` 选取指定秒数以内的整数采样时延，
包含零；默认完整输出 `[-(N-1), N-1]`。相关和不额外乘 `dt`。

## 处理顺序

1. 提取完整窗口，按台站/窗口排除缺测，并使用质量权重。
2. `detrend="constant"` 去均值（默认）；`"linear"` 去最小二乘线性趋势；`None` 关闭。
3. 可选 `time_normalization="onebit"` 取符号，或 `"rma"` 除以时域幅度滑动均值。
   `time_smooth_samples` 是奇数总窗口宽度，端点重复扩展。
4. 两端半余弦渐消，`taper_fraction` 是每一端占比，默认 0.05，允许 0–0.5，0 关闭。
5. 可选频带过滤和谱白化。平滑/严格白化沿用[原接口定义](DATA_PROCESSING.zh-CN.md)。
   `whitening=None` 时，显式设置 `freqmin/freqmax` 仅做频带过滤；两者均省略则不处理频域。
   启用频带过滤会去除 DC，`taper_hz` 是通带外的余弦过渡带。
6. 相关归一化、补零 FFT、台站对互谱和叠加。

白化使用原窗口长度 N 的 FFT，再对处理后的有限记录补零计算线性相关；频谱平滑宽度
对应 N 点 FFT 的频率间隔。每个台站的相关用 FFT 在各台站对间复用。

`window_samples=None` 表示整段；步长默认半个窗口，只使用完整窗口。
`window_starts` 和 `dropped_samples` 给出覆盖范围。不会自动插值缺测、重采样、去仪器响应、
旋转分量或校正时钟；这些需在上游完成。观测和预测必须使用相同预处理，白化/one-bit 会改变
幅度信息，输出并不自动等同于具有绝对幅值标定的格林函数。

## 归一化、权重与叠加

| `normalization` | 每窗口的定义 |
|---|---|
| `"none"` | 原始样本乘积和 |
| `"biased"` | 除以 N |
| `"unbiased"` | 除以 `N-abs(lag)`，只校正矩形窗重叠数，不校正渐消/滤波偏差 |
| `"coeff"`（默认） | 除以两个处理后完整窗口的 L2 范数，不是逐时延局部 Pearson 系数 |

`window_weights` 广播到 `[..., 窗口, 台站]`，须有限且非负；0 排除该台站窗口。
台站对权重为两个台站权重的乘积。窗口掩码在分窗后应用，避免影响相邻重叠窗口。

| `stack` | 返回与计算 |
|---|---|
| `"mean"`（默认） | 加权均值，`[..., 台站对, 时延]` |
| `"sum"` | 加权和，同上 |
| `None` | 各窗口独立相关，`[..., 窗口, 台站对, 时延]`；返回值尚未乘权重 |
| `"pws"` | 加权均值乘以加权相位相干度的 `pws_power` 次方（默认 2） |

线性叠加先累积互谱，最后每个输出台站对只做一次反 FFT。PWS 需要逐窗 CCF，在选定时延区间
通过离散 Hilbert 变换求解析信号相位。因此改变 `max_lag` 会影响边界相位估计；
`pws_power=0` 回到加权均值，解析幅值为零时相位贡献设为零。

## 缺测与返回值

默认 `nan_policy="raise"` 拒绝 NaN/Inf；`"omit"` 在当前设备检查每个台站窗口，
排除包含非有限采样点的窗口，其他台站/窗口仍可参与。处理后能量为零的窗口也排除。
无有效台站对权重时返回 **NaN 和 `valid=False`**，不能将其填 0 后作为有效训练标签。

叠加结果的 `window_count` 是贡献窗口数，`weight_sum` 是台站对权重和，形状均为
`[..., 台站对]`。`stack=None` 时它们为 `[..., 窗口, 台站对]`，计数为 0/1，
权重和就是当前窗口的台站对权重。`pairs`、`lag_samples`、`lags`（秒）、`window_starts`
（采样点）均留在输入设备。FFT 长度、窗长/步长、丢弃尾部样本数等为普通元数据。

## 设备、显存及梯度

去趋势、时间归一化、白化、缺测掩码、FFT、互谱、叠加、PWS 和微分均留在输入 CPU/CUDA。
台站对/权重若传入主机列表或 NumPy，会在入口复制一次；张量参数必须与记录位于同一设备。
默认校验有限值、台站对范围和权重可能触发标量同步。上游已校验时使用
`check_finite=False` 可去除这些同步；`nan_policy="omit"` 仍在设备上处理缺测。

窗口和台站对分块限制前向临时显存，密集全台站对输出本身仍为台站数平方量级。
可指定所需台站对减少成本；`stack=None` 要保存全部输出窗口。
线性叠加还需保留 `batch * 台站对数 * (nfft//2+1)` 个复数的频谱累积量，
这部分不随台站对分块大小减少；显存不足时需指定更少的台站对。
autograd 会保留反向所需张量，分块不保证训练计算图总显存固定。
one-bit 的 sign 几乎处处梯度为零；掩码切换、阈值、零幅值与并列极值处有不可微边界。
生产微分使用 PyTorch autograd，没有差分；采样、几何、处理选项是固定元数据。

运行示例：`python examples/noise_cross_correlation.py --device cuda`，也可选 `cpu`/`numpy`。
独立时域相关、脉冲时延、台站对互易性、缺测、权重、分块、PWS、CPU/CUDA 精度、梯度及
非默认 CUDA stream 均有测试。测速脚本单独记录 NumPy、单/多线程 CPU、驻留 CUDA、
包含波形上传和 CCF 下载的 CUDA 耗时；排除预热与磁盘读写，不宣称普遍加速比。
