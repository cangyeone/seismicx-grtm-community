# 数据处理：频谱白化

[English](DATA_PROCESSING.md) | 简体中文

新增 `grtm.processing` 数据处理模块，提供平滑谱白化和严格白化。
也可直接从 `grtm` 导入，功能包含于 `0.6.0.dev15` 预览版。
见[安装说明](PREVIEW_0.6.0.dev15.zh-CN.md)；稳定 PyPI 0.5.1 尚未包含。

```python
from grtm.processing import spectral_whiten, whiten_spectrum

# traces 可以是单道 [sample]、多道 [station, sample] 或更高维批量。
smooth = spectral_whiten(traces, dt=0.05, freqmin=0.1, freqmax=2.0,
                         method="smooth", smooth_bins=11, taper_hz=0.05)
strict = spectral_whiten(traces, dt=0.05, freqmin=0.1, freqmax=2.0,
                         method="strict", taper_hz=0.05)
```

## 两种方法

设复频谱为 `X`，幅度为 `A=abs(X)`，频带窗为 `W`。

| 方法 | 计算方式 | 可用名称 |
|---|---|---|
| 平滑白化 | `W * X / max(滑动平均(A), floor)` | `smooth`、`rma` |
| 严格白化 | 有效频点使用 `W * X / A`，其余为零 | `strict`、`brutal`、`phase_only` |

名称不区分大小写。平滑白化参考 [Bensen 等（2007）第 2.2 节](https://ciei.colorado.edu/ambient_noise/pubs/GJI_Bensen_etal2007_final.pdf)；
[NoisePy 入门示例](https://noisepy.github.io/NoisePy/get_started.html)采用 `rma`。
严格白化对应 [MSNoise 工作流](https://msnoise.org/doc/workflow/005_compute_cc.html)
中的 `Brutal` 定义。别名表示算法类型，不表示与外部软件整条处理流程、
平滑边界或默认窗参数逐点相同。
[MSNoise 1.6 配置源码](https://github.com/ROBelgium/MSNoise/blob/1.6/msnoise/default.py#L50)
也明确记录默认类型为 `Brutal`（幅度设为 1.0）。

- `smooth_bins=11` 表示**总共 11 个频率点**，不是左右各 11 点。
  使用中心算术滑动平均，两端复制端点；先对完整 rFFT 幅度谱平滑，再施加频带窗。
  窗宽必须为正奇数且不超过频谱长度，名义带宽为
  `smooth_bins / (n_samples * dt)` Hz；严格模式忽略该参数。
- `floor = water_level * 通带内最大幅度`，各道独立计算。
  默认 `water_level=1e-12`。平滑模式限制最小分母；严格模式将幅度不高于该阈值的频点置零，
  其余通带频点幅度为 1。设为 `0` 时，每个非零通带频点都严格归一为 1。
- 精确零幅度的频点保持为零，全零记录仍为零；不为无信号频点制造相位。
  两种方法保留有效频点的相位，并去除直流分量。
- 平滑窗宽为 1 且 `water_level=0` 时，两种方法等价。

## 参数和返回值

`dt` 使用秒；`freqmin`、`freqmax`、`taper_hz` 使用 Hz。
要求 `0 <= freqmin < freqmax <= Nyquist`，且通带内至少有一个正频率 FFT 点。
默认频带覆盖零频至 Nyquist，直流分量始终置零。

`taper_hz=0` 为矩形频带；正值在指定通带**外侧**加半余弦过渡，
并截于零频和 Nyquist。通带内保持平坦，过渡带幅度从 1 降至 0。

`axis=-1` 表示最后一维为采样轴，其余维度均为独立批量。
例如 `[batch, sample, component]` 使用 `axis=1`。
输出保留形状、精度和奇偶长度，输入不会被修改；NumPy 整数波形提升为 float64。
不隐式补零、去趋势、加时间窗或重新调整 RMS。
应先处理缺测、趋势和记录边缘，再进行白化。

已有 FFT 时可直接处理复频谱，减少重复计算：

```python
import numpy as np

n = traces.shape[-1]
X = np.fft.rfft(traces, axis=-1)
Xw = whiten_spectrum(X, dt=0.05, n_samples=n,
                    freqmin=0.1, freqmax=2.0, method="strict")
whitened = np.fft.irfft(Xw, n=n, axis=-1)
```

`n_samples` 必须是原 FFT 长度，频谱长度必须为 `n_samples // 2 + 1`。
采用 NumPy/PyTorch 默认 FFT 归一化；传入有效单边 rFFT 频谱，
即 DC 和偶数长度的 Nyquist 分量为实数。
白化会移除绝对振幅信息；用于反演时，观测和预测应使用同一套处理参数。

## CPU、CUDA 和微分

NumPy 使用 CPU；PyTorch float32/64、complex64/128 使用输入张量所在的 CPU/CUDA。
FFT、滑动平均、归一化、逆 FFT 和梯度都在该设备完成，不自动回退 CPU。
NumPy 路径不依赖 PyTorch；张量路径需要安装 PyTorch。
暂不支持 float16、bfloat16 和 MPS，遇到这些输入会明确报错。

```python
import torch
from grtm.processing import spectral_whiten

x = torch.randn(32, 4096, dtype=torch.float64, device="cuda", requires_grad=True)
y = spectral_whiten(x, dt=0.05, freqmin=0.1, freqmax=2.0,
                    method="smooth", smooth_bins=11, check_finite=False)
gradient, = torch.autograd.grad(y.square().mean(), x)
assert y.device == x.device == gradient.device
```

默认 `check_finite=True` 检查 NaN/Inf，在 CUDA 上会有一次标量同步。
已经完成数据质量检查时，可设为 `False`，消除这一步同步；关闭检查后，非有限输入不受支持。
梯度通过 PyTorch 的 FFT、幅度、平滑和归一化自动微分计算，不使用生产差分。
零幅度、阈值切换和最大值并列处存在分段可微的限制；采样和处理参数是固定配置。

完整示例：`python examples/spectral_whitening.py --device numpy`，
也可使用 `--device cpu` 或 `--device cuda`。
公式、相位、频带、全零、batch、精度、梯度和设备驻留测试在私有实现仓库执行。
公开验证结果见[dev15 验证记录](../verification/preview-0.6.0.dev15/verification.json)。
