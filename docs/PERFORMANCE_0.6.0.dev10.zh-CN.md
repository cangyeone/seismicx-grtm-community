# dev10 格林函数跨模型融合：精度与速度

比较未改动的 dev9 安装包与 dev10 安装包。旧包分别使用反向 AD、显式 wavefield；新包分别使用融合 strict、融合 fast。strict 保留原低频保护，fast 在经过验收的频点使用 GPU 显式积分，极低频仍保护。CPU、MPS、RF 和频散算法本次未改。

RTX 5090 / Threadripper PRO 3955WX；Torch 1 线程，原生 CPU 保护 4 线程。FP64、PTAM=0、N=128、每频率 K=512 波数步、dt=0.4 s、积分周期长度 500 km、1 个接收点。8/32 层，B=1/4/16，batch 中速度逐模型增加 0.1%。全部六列结构参数与震源梯度，固定平方波形损失。

每种方法预热 1 次、同步墙钟计时 3 次，以下为中位数，原始 JSON 保留全部重复。包括模型校验、主机元数据传输、传播、合成、FFT 与反传；不包括编译/首次安装。相同模型和固定积分网格，无减少节点。strict 每模型有 3/64 个 CPU 保护频点；本表 fast 的 CPU 频点数为 0。

## 正演 + 全参数 VJP

| L | B | dev9 AD ms | dev9 wavefield ms | dev10 strict ms | dev10 fast ms | wavefield/strict | wavefield/fast | AD/fast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 441.865 | 215.039 | 212.027 | 32.611 | 1.01× | 6.59× | 13.55× |
| 8 | 4 | 1861.536 | 932.469 | 894.016 | 100.818 | 1.04× | 9.25× | 18.46× |
| 8 | 16 | 7307.377 | 4126.933 | 3516.751 | 365.433 | 1.17× | 11.29× | 20.00× |
| 32 | 1 | 3539.108 | 969.595 | 856.143 | 79.257 | 1.13× | 12.23× | 44.65× |
| 32 | 4 | 14462.191 | 3433.517 | 3505.059 | 272.950 | 0.98× | 12.58× | 52.98× |
| 32 | 16 | 57365.357 | 14337.333 | 13215.841 | 1028.001 | 1.08× | 13.95× | 55.80× |

## 仅正演

| L | B | dev9 AD ms | dev9 wavefield ms | dev10 strict ms | dev10 fast ms | wavefield/strict | wavefield/fast | AD/fast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 52.841 | 52.619 | 53.446 | 7.215 | 0.98× | 7.29× | 7.32× |
| 8 | 4 | 208.739 | 236.031 | 196.078 | 15.365 | 1.20× | 15.36× | 13.59× |
| 8 | 16 | 838.900 | 942.335 | 766.149 | 47.696 | 1.23× | 19.76× | 17.59× |
| 32 | 1 | 190.632 | 200.500 | 193.479 | 19.365 | 1.04× | 10.35× | 9.84× |
| 32 | 4 | 769.218 | 768.468 | 742.741 | 52.070 | 1.03× | 14.76× | 14.77× |
| 32 | 16 | 3391.038 | 3373.196 | 2952.635 | 167.781 | 1.14× | 20.10× | 20.21× |

## 单独衡量跨模型融合

| L | B | serial fast ms | fused fast ms | serial/fused |
|---:|---:|---:|---:|---:|
| 8 | 16 | 498.039 | 365.433 | 1.36× |
| 32 | 16 | 1251.065 | 1028.001 | 1.22× |

两列均使用 dev10 fast，仅改变 `cuda_batch`。这是融合自身的收益；上表 wavefield/fast 同时包含低频路径改变。单模型已经有大量频率/波数并行，因此 batch 融合不保证再得到 B 倍加速。

## 同机 CPU 参考

| L | B | CPU wavefield ms | CUDA fast ms | CPU/CUDA |
|---:|---:|---:|---:|---:|
| 8 | 1 | 823.015 | 32.611 | 25.24× |
| 32 | 1 | 2851.363 | 79.257 | 35.98× |

CPU 参考仍为未修改的 dev9，4 原生线程；本次没有宣称纯 CPU 新增提速。CPU/CUDA 同时包含设备和精度保护策略差异。

## 精度与覆盖范围

全部记录的最大相对范数误差: waveform **1.263e-08**, model gradient **1.921e-08**, source gradient **1.183e-08**, individual parameter column **1.639e-07**.

额外 48 条精度记录覆盖 8/32 层、B=4、普通梯度、LVL、强梯度、极薄层、浅半空间和低 Q；分别使用普通及长时间窗，并采用固定随机线性波形 cotangent。strict/fast 都与同一模型的原反向 AD 比较。长时间窗保留保护，不能将正常时间窗的加速比推广过去。薄层等特殊几何可能使用 GPU AD 回退。所有记录均达到波形 1e-4、模型/震源梯度及各参数列 1e-3 的目标。

这些是有限案例的数值验证，不是任意模型的误差保证、连续波数积分收敛证明或反演精度优越性的证明。严格模式仍为默认。微小速度差可能受三次采样和系统噪声影响。模型/网格规划仍在主机，只有 `cpu_frequencies==0` 时数值传播和梯度才全部在 GPU。

两平台安装包各运行 149 项测试：Apple 118 通过/31 平台跳过，Linux/CUDA 140 通过/9 平台跳过。测试包含不同模型层位、物理场、单位、掩码、非默认 stream、保留图重复反传和退出时资源释放。

[API](CUDA_GREEN_BATCH.zh-CN.md) · [Install](PREVIEW_0.6.0.dev10.zh-CN.md) · [Verification](https://github.com/cangyeone/seismicx-grtm-community/tree/main/verification/preview-0.6.0.dev10)

Raw records: [legacy-cuda.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/legacy-cuda.json) · [fused-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/fused-final.json) · [serial-fast-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/serial-fast-final.json) · [validation-final.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/validation-final.json) · [cpu-reference.json](https://github.com/cangyeone/seismicx-grtm-community/blob/main/verification/green-batch-0.6.0.dev10/cpu-reference.json)
