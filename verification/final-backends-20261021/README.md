# CPU/CUDA/Apple measurements — 0.6.0.dev5

See the [bilingual report](../../docs/PERFORMANCE_0.6.0.dev5.md) and [execution boundaries](../../docs/DEVICE_EXECUTION.md).
Raw synchronized repetitions, model counts, CPU budgets and numerical discrepancies are retained in JSON; provenance records hardware and runtimes. MPS RF/surface entries explicitly compute on CPU. Green CUDA derivatives retain host precision exceptions and field synthesis.

Reproduce with [benchmark_backends.py](../../benchmarks/forward-derivatives/benchmark_backends.py) and [benchmark_pyfk.py](../../benchmarks/forward-derivatives/benchmark_pyfk.py), using installed GRTM and original external dependencies. Output files must not already exist. These drivers use public APIs and contain no solver implementation.

`pyfk-refined-*` uses finer quadrature than `pyfk-*`; do not mix their timings. CUDA pyfk is unchanged upstream Numba/CuPy. Original disba uses default finite-difference kernels; they are not derivative ground truth. Some short GPU timings vary substantially; all trials are provided. No cross-host ratios or general inverse-accuracy claim is made.
