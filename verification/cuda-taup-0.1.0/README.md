# Physical CUDA TauP 0.1.0 verification

[Chinese report](../../docs/CUDA_TAUP.zh-CN.md) | [English report](../../docs/CUDA_TAUP.md)

Compiled physical solver only. No learned model, weights, training data or solver
implementation source is distributed here. The full source is in seismicx-grtm.

The installed binary passes 8 API/geometry/device tests and the 342-query,
8447-arrival original ObsPy1.4.2 regression, plus 36 path/pierce cases.
Seven more built-in models match 629 arrivals over 21 queries.

extra-compatibility.json retains raw custom low-velocity-model count differences
and original shoot_ray audits of non-closing PS/SP candidates. CUDA rejects
these discontinuity artifacts, warns on host conversion and exposes diagnostics.
There are no unexplained differences in that audit; this is not a claim of exact
original API counts on every custom model. The modified input model is included.

performance.json records source-interface timings before diagnostic-only changes,
shared CPU/GPU host load, and both original-API and pre-parsed 8-process baselines.
It is not a dedicated-machine guarantee or a benchmark of the final wheel.
Final wheel numerical verification is separate.

verification.json binds source hashes, the audited wheel and numerical evidence.
Installed test logs redact home-directory prefixes only. Research/noncommercial
license; ObsPy is a separately installed unmodified LGPL dependency.
