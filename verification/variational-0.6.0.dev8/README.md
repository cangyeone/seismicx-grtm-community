# Variational-kernel measurements, 2026-10-22

`measure.py` measures installed compiled wheels, not a source import. CPU uses
four native threads and one Torch thread; CUDA uses one host/native control
thread and the caller's GPU stream. Two warmups, three timed repetitions,
synchronized wall-clock medians, FP64, no overlap between timed workloads.

Surface models use layer tops linearly spaced from 0 to 45 km, Vs=3.2+0.027z,
Vp=1.8Vs, rho=2.6+0.01z, and periods 2,3,5,8,12,20,35,50 s. All four layer
parameters are requested, fundamental Rayleigh mode, root tolerance 1e-11 km/s.
The 8/32-layer models use B=1/32; 96 layers uses B=1. `end_to_end` includes
roots and all kernels; `known_roots` excludes roots but includes the base/root
period derivative. `forward` is the analytical per-parameter reference,
`adjoint` is the existing characteristic adjoint, `eigenfunction` is the new
energy-integral path. Guarded fallbacks and maximum native-unit absolute
kernel differences are recorded. The complete cost is not parameter-count
independent: all layers must still be traversed and kernels written.

RF uses the same Earth profiles, two rays, two Gaussian filters, nfft=256 and
81 samples. Timings include forward and a combined RF/polarization objective
adjoint. Green uses 16 samples, 32 fixed wavenumber steps, B=1/4 and either one
Vs parameter or all Vp/Vs/rho values. Green's retained adjoint is compared with
parameter-wise analytic tangents. CUDA tangent mode includes its legacy host
synthesis path; CUDA adjoint mode also benefits from the existing GPU FFT and
synthesis. This is an end-to-end comparison, not an isolated primitive speedup
or a new dev8 Green adjoint algorithm.

Raw JSON requires `complete: true`. Keep all cases, including slowdowns. Compare
CPU and CUDA on the same Threadripper/RTX host separately from Apple CPU. dev7
and dev8 wheel comparisons include compiler/build-environment differences;
within-dev8 method comparisons isolate those differences. No disba sensitivity
kernel is treated as ground truth. Recomputed-root FD, independent RF tensor
physics, Green tangents, invariants, missing modes and CPU/CUDA parity are
covered by `tests/test_energy_kernels.py` and the full installed-wheel suite.

The benchmark driver and test implementations are maintained in the full source repository. This public directory distributes numerical evidence and protocol only. Absolute host paths were redacted; numerical measurements are unchanged.
