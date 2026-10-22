# 频散能量图伴随：精度与效率（0.6.0.dev12）

实测 RTX 5090（CUDA:1）对 Threadripper PRO 3955WX，CPU 原生 4 线程、Torch 算子 1 线程；FP64，64 频率×256 速度，damping=0.002。下表为**完整正演＋功率目标＋全部层参数梯度**的同步墙钟时间，预热 1 次、测量 3 次取中位数，排除构造模型/最终拷回主机。阵列基函数使用热缓存；不含首次基函数构造。

| 层数 | Batch | 波型/观测 | CPU (ms) | CUDA (ms) | 加速比 |
|---:|---:|---|---:|---:|---:|
| 8 | 1 | rayleigh direct | 169.196 | 4.051 | 41.8× |
| 8 | 1 | love direct | 13.481 | 3.088 | 4.4× |
| 8 | 16 | rayleigh direct | 1844.034 | 22.341 | 82.5× |
| 8 | 16 | love direct | 196.287 | 5.089 | 38.6× |
| 32 | 1 | rayleigh direct | 442.244 | 8.656 | 51.1× |
| 32 | 1 | love direct | 52.282 | 3.638 | 14.4× |
| 32 | 16 | rayleigh direct | 7153.278 | 79.276 | 90.2× |
| 32 | 16 | love direct | 762.933 | 14.398 | 53.0× |
| 32 | 1 | rayleigh fk coherent | 1082.719 | 17.407 | 62.2× |
| 32 | 1 | rayleigh fj coherent | 532.646 | 11.286 | 47.2× |
| 32 | 1 | rayleigh fj noise | 445.406 | 10.172 | 43.8× |
| 32 | 16 | rayleigh fk coherent | 15038.916 | 158.710 | 94.8× |
| 32 | 16 | rayleigh fj coherent | 7077.971 | 81.906 | 86.4× |
| 32 | 16 | rayleigh fj noise | 7208.355 | 81.023 | 89.0× |

阵列测速使用 Rayleigh、32 个接收距离、4 km 长度、0.2 km 偏移、矩形窗，积分与显示共用网格。此配置用于比较计算量，不是空间采样收敛证明。CUDA/CPU 梯度相对 L2 最大差异 **3.46e-12**.

独立 PyTorch GRTM 通用矩阵求解反向 AD 对照：16×64 网格，CPU 两者均使用 1 线程；CUDA 两者均在同一 GPU。下表为“PyTorch AD 耗时 / 显式伴随耗时”，不是对历史版本或所有 AD 实现的普遍加速承诺。

| Layers | Batch | Wave | CPU ratio | CUDA ratio |
|---:|---:|---|---:|---:|
| 8 | 1 | rayleigh | 1.14× | 14.26× |
| 8 | 1 | love | 2.85× | 8.21× |
| 8 | 16 | rayleigh | 0.57× | 7.98× |
| 8 | 16 | love | 0.67× | 3.53× |
| 32 | 1 | rayleigh | 1.12× | 29.35× |
| 32 | 1 | love | 3.10× | 26.03× |
| 32 | 16 | rayleigh | 0.57× | 22.04× |
| 32 | 16 | love | 0.64× | 12.71× |

CPU 并非全面提速：相同单线程条件下，batch=16 的显式伴随为该 Torch AD 参考的 0.57–0.67×，即耗时约多 1.5–1.8 倍；CUDA 收益更明显。报告保留这些较慢案例。

与独立 AD 的梯度相对 L2 最大差异：**3.97e-12**.

独立全局 P/SV 波动方程校验覆盖 1/3/8/32 层；方向差分仅用于验收。步长 10⁻⁵ 时，与解析伴随的方向导数相对差异在 10⁻¹¹–10⁻⁸ 量级。还检查了复数震源梯度、PSD 线性及归一化退化、正负波数物理对称性、独立角向平均得到的 J0 互谱、SI 换算、网格加密、NaN 观测掩码和 CUDA 非默认 stream。

主动与随机 F–J 的双参数无噪声局部反演均能恢复表层 Vs 和首界面深度；这是固定其它参数及震源的功能验证，不是全局收敛或统计优越性的证明。

最终二进制完整测试：macos 136 passed / 33 platform skips (169 total), linux 159 passed / 10 platform skips (169 total).

详细数值、每次计时样本与比较图：[verification](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/dispersion-adjoint-0.6.0.dev12). [接口与物理定义](DISPERSION_IMAGE.zh-CN.md)。
