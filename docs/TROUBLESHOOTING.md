# Troubleshooting

[Home](../README.md) · [Installation](INSTALLATION.md)

| Symptom | Check or action |
|---|---|
| `No matching distribution found` | Use CPython 3.10–3.14 and a supported OS/architecture; upgrade pip. Intel Macs, native Windows, Linux ARM, PyPy and free-threaded Python are not covered. |
| `ModuleNotFoundError: grtm` | Run installation and execution with the same interpreter: `python -m pip`; check the active virtual environment. The install name is `seismicx-grtm`. |
| Wrong module imported | Avoid a local file named `grtm.py`; print `grtm.__file__`. Remove the old `grtm-green` distribution if installed. |
| CUDA listed but fails to initialize | A bundled library does not guarantee a usable GPU. Verify NVIDIA driver/WSL access and compute capability 12.0; use `backend="cpu"` on unsupported hardware. |
| CUDA reports no compatible kernel image | The 0.5.1 binary targets compute capability 12.0, not older NVIDIA architectures. |
| MPS dtype error | PyTorch MPS tensors need float32/complex64. Set both tensor device and solver backend; use CPU float64 when needed. |
| GPU is slower | Small jobs may not amortize setup/transfers. Time the complete API call after warmup; compare the same model, grid and outputs. |
| High memory consumption | Use model iterators, fewer concurrent workers, selected fields and VJP instead of dense Jacobians. JAX derivative callbacks retain dense Jacobians. |
| Invalid distance/layer error | Require positive distances, increasing tops beginning at zero, Vp > Vs > 0 and positive density/Q. The last layer is a half-space. |
| Unexpected amplitudes | Distinguish SI `forward()` from native-unit `compute()`; check moment units, source history versus rate, impulse-kernel units and coordinate signs. |
| Source history rejected with nonzero `t0` | Start every receiver window at zero for causal convolution, then crop the returned history. |
| Gradient discontinuity | Check layer crossings, interfaces at source/receiver, adaptive grid or PTAM branch switches; compare on a frozen local grid. |
| RF inversion converges to a bound | Inspect grid range, fit window, fixed Vp, filter/sign consistency, parameter identifiability and residuals. Extend bounds only when physically justified. |

## Diagnostic information

```sh
python --version
python -m pip show seismicx-grtm numpy
python -m pip check
grtm --info
```

```python
import grtm
print(grtm.__file__)
print(grtm.__version__)
print(grtm.build_info())
```

For a bug report include a minimal model, source, command/API call, expected and
observed behavior, traceback and backend. Remove private paths or confidential
data as appropriate. Use [Issues](https://github.com/cangyeone/seismicx-grtm-community/issues)
or the developer contacts on the [home page](../README.md#developers).
