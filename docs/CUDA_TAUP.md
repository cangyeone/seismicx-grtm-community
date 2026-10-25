# Physical CUDA TauP

This folder contains the physical CUDA solver. No learned model or training
weights are used or distributed in this component.

Distances follow ObsPy's **degrees**, not km. Depths are km and times seconds.
Convert arc km with `distance_km / model.model.radius_of_planet * 180 / pi`.

```python
from cuda_taup import TauPyModel  # source checkout
# from seismicx_cuda_taup import TauPyModel  # compiled distribution

model = TauPyModel("iasp91", device="cuda:0")
arrivals = model.get_travel_times(10., 35., phase_list=["p", "P", "Pn", "S", "SKS"])
for a in arrivals:
    print(a.name, a.time, a.takeoff_angle, a.incident_angle)

paths = model.get_ray_paths(10., 120., phase_list=["Pdiff", "PKIKP"])
pierce = model.get_pierce_points(10., 35., phase_list=["P", "S"], add_depth=[100., 250.])
paths.plot_rays(show=False)  # ObsPy arrival containers/plotting are preserved
```

The class subclasses ObsPy's `TauPyModel` and replaces its travel-time,
ray-path and pierce methods. Geographic methods call these CUDA replacements;
ObsPy retains the geodesic coordinate conversion and geographic resampling.
`geographiclib` is needed for geographic path coordinates, as in ObsPy.

## Device-resident batches

```python
import torch
distances = torch.linspace(.1, 179.9, 1024, dtype=torch.float64, device="cuda:0")
model.prepare(10., phase_list=["ttall"])
r = model.get_travel_times_batch(10., distances, phase_list=["ttall"])
print(r.time, r.takeoff_angle, r.incident_angle, r.ray_param_sec_degree)
print(r.phase_names, r.phase_index, r.query_index)
print(r.gradient)  # source-depth and arc-distance partials, s/km

r = model.get_travel_times_batch(10., distances, ["P", "S"], output="path")
print(r.path_values, r.path_offsets)  # packed CUDA [p, time, distance_rad, depth]
lists_of_obspy_arrivals = r.to_obspy()  # explicit device-to-host conversion
```

Each batch shares a source depth, receiver depth and phase list, matching the
common event-to-many-stations use case. Depths are not restricted to 200 km,
nor distances to 3000 km. Built-in or custom ObsPy `.npz` models, P/S, converted,
reflected, core, head and diffracted phases, phase groups and `kmps` surface phases
use the same parser and model constraints as ObsPy. Angular winding is retained.

## Execution and precision

CPU preprocessing loads the model, inserts source/receiver/additional depths and
parses phase itineraries. `prepare` caches this work. New unique depths or phase
sets have a cold CPU cost; this backend does not claim a fully GPU model builder.
GPU FP64 evaluates branch integrals and curves, searches all phase intervals,
refines roots, calculates angles and accumulates paths/pierce points. The hot path
does not call CPU travel-time, shooting, root, ray-path or pierce solvers. Tensor
results stay on the chosen GPU; compatibility `Arrival` objects are materialized
on the CPU by request. Geographic conversion and plotting retain ObsPy CPU code.

Default `subdivisions=1, search_hidden_roots=False` preserves the original phase
sampling/arrival convention, while filtering non-closing discontinuity artifacts.
`TensorArrivals.rejected_candidates` and `Arrivals.cuda_diagnostics` expose these
rejections; host conversion also emits a warning. Repeated ray parameters split
shadow gaps. Optional
`subdivisions=8, search_hidden_roots=True` searches finer samples and internal
extrema, and can return additional valid rays absent from the standard ObsPy API;
it is not an exact-count compatibility mode or a universal no-missed-root proof.

`ray_param_tol` must be positive; the ray-parameter stopping tolerance is tightened
to min(ray_param_tol, 1e-11) s. Distance convergence or FP64 resolution may also
terminate refinement; this is not a rigorous error bound. Detailed paths follow the
Bullen slowness interpolant. Pierce points retain ObsPy's coarse interpolation
and linear-velocity turning-depth convention; these are different observables.
Time-gradient branches are one-sided at velocity interfaces and can jump at
caustics/first-arrival switches. No automatic differentiation through phase
changes or source-depth preprocessing is claimed.

## Build and checks

Install NumPy, CUDA-enabled PyTorch, ObsPy 1.4.2, and optionally geographiclib.
Build with CUDA toolkit 12.8 for the tested RTX 5090:

```sh
TAUP_NVCC=/path/to/nvcc TAUP_CUDA_ARCH=sm_120 sh cuda_taup/build.sh
python -m unittest discover -s cuda_taup/tests -v
python cuda_taup/scripts/validate.py --device cuda:0
python cuda_taup/scripts/benchmark.py --device cuda:0 --n 1024
```

The CUDA architecture is configurable when building from source. A compiled
wheel targeting SM120 is not a claim of binary compatibility with older GPUs.
See `results/` for original ObsPy comparisons, timings and test logs. No neural
surrogate is used in these measurements. [Licensing and notices](../LICENSE).

## Verification and timings

Against original ObsPy 1.4.2, 342 queries across iasp91/ak135/PREM match all
8447 arrival counts. Maximum time difference 7.33e-11 s, maximum angle
difference 2.51e-4 degrees. There are 36 full path/pierce cases; maximum point
time/distance/depth differences are 2.20e-5 s, 4.23e-8 radians and 2.63e-7 km.
Seven additional built-in models match 629 arrivals in 21 queries.

A modified iasp91 low-velocity model reveals PS/SP candidates returned by the
original API whose original shoot_ray endpoints miss the target by 17-61 degrees.
CUDA filters non-closing candidates, exposes GPU rejected_candidates and host
Arrivals.cuda_diagnostics, and emits a host warning. Raw count differences and
original ray checks are retained. This is an intentional deviation from those
original API artifacts, not a claim of exact counts for every custom model.

| Physical output | Batch | CUDA ms | Original CPU 8-process s | Speedup | Prepared CPU speedup |
|---|---:|---:|---:|---:|---:|
| P/p/Pn time and angles | 1024 | 81.9 | 2.961 | 36.2x | 32.6x |
| ttall time and angles | 1024 | 3734.7 | 77.745 | 20.8x | 20.2x |
| ttall ray paths | 32 | 299.7 | 7.622 | 25.4x | 25.0x |
| ttall pierce points | 32 | 277.4 | 4.750 | 17.1x | 16.7x |

Source-interface timings on RTX5090, depth10 km, preloaded models/plans, resident
inputs; median of five synchronized GPU calls. CPU batches measured once on
the same points, both original API and pre-parsed phase objects. Installed wheel
numerical checks are separate. Other tasks used 24 CPU workers and both GPUs;
observed load average was about29 and GPU utilization81/78 percent. These are
shared-host observations, not dedicated-machine guarantees. Single P-family
queries favor CPU (13.4 ms versus31.9 ms CUDA). No MLP is used in these numbers.
