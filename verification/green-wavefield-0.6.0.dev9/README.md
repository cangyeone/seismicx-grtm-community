# dev9 installed-wheel benchmark evidence

Read the [English report](../../docs/PERFORMANCE_0.6.0.dev9.md) or [中文报告](../../docs/PERFORMANCE_0.6.0.dev9.zh-CN.md).

144 synchronized median timing records compare `wavefield` against `adjoint` in the same installed dev9 binary. `apple` is M4 Max CPU; `cpu` is Threadripper PRO 3955WX; `cuda` is RTX 5090 on the same Linux host. Native CPU threads=4, Torch threads=1, FP64, two warmups and three recorded samples. `regular-final` has no protected frequency bins; the other large workload retains 3/64 protected bins. Different workload rows are not interchangeable.

`qe-known-root-apple.json` measures eight fundamental Rayleigh periods at identical known roots, one CPU thread, excluding root search and upstream C++ map allocations. It includes the original reference and isolated FP64 promotion experiments. `cuda-profile.json` contains diagnostic GPU activity, not end-to-end speed measurements. `summary.json` contains derived ratios and retains slowdowns.

Release wheels, hashes, packaging audits and installed test logs are in [preview verification](../preview-0.6.0.dev9/). Solver source and benchmark drivers are maintained in the full repository; this public folder contains only documentation and numerical evidence. The research/noncommercial licence remains unchanged.
