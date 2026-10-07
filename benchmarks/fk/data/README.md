# Benchmark data dictionary

[Benchmark report](../README.md)

All times are seconds. Empty CSV sub-timings mean unavailable, not zero. Case
identifiers and configuration identifiers join the files; each case has ten
configurations. Model arrays in `protocol.json` use **native units**: km, km/s,
g/cm³, seconds, dimensionless Q. Convert them before high-level SI calls.

## Files

| File | Rows / contents |
|---|---|
| `environment.json` | Host software/hardware and 60 case-boundary load snapshots |
| `protocol.json` | Locked implementation identifiers, 15 full models, grids and registered comparison rules |
| `timings.csv` | 150 `(case, configuration)` summaries |
| `trials.csv` | 750 `(case, configuration, round)` measurements; rounds 0–4 |
| `warmups.csv` | 300 `(case, configuration, warmup)` measurements; warmups 0–1 |
| `figure-values.csv` | 144 plotted values with panel and metric identifiers |
| `reuse-ablation.json` | Separate panel h ablation: four receiver counts, eight threads, three trials per operation |
| `validation.json` | Archived scientific consistency-check summary |

## Summary fields

`layers` includes the half-space; `receivers` counts distances. `source_depth_km`
and `nominal_range_km` describe the sweep; use the actual `distances` model array
for station positions. `samples`, `dt_s`, `period_km`, and `wavenumber_nodes`
describe the numerical workload. `wavenumber_nodes` is the frequency-summed
count. The receiver/layer sweep has one common frozen grid; geometry cases do not.

`threads` is the native host-worker count. `median_s`, `min_s`, `max_s` are the
five complete-call wall-time statistics. `fk_serial_over_time`,
`fk_parallel_over_time`, `c1_over_time` divide the appropriate **same-host**
baseline median by the row's median. Values greater than one indicate the row
is faster than that baseline. `max_waveform_channel_error_vs_fk` is the maximum
per-channel relative L2 signal discrepancy versus same-host serial FK.

## Trial fields

`wall_seconds` is the primary end-to-end measurement under the report's timer
scope. `compute_seconds` and `fft_seconds` are recorded implementation
sub-timings; they need not sum to wall time. `signal_sha256` and
`spectrum_sha256` identify timed numerical outputs. Output arrays are not
included in this public export, so the hashes support internal trial-stability
inspection but are not a substitute for independent waveform validation.

## Figure fields

`panel` refers to panels a–h in the combined figure. `case`, `configuration`,
and `metric` identify a trace point. `value` is the plotted time or ratio;
`range_min` and `range_max` are plotted observed ranges, not statistical
confidence limits. Panels a–g are calculated from `timings.csv`. Panel h is a
separate within-GRTM receiver-reuse experiment whose scope is described in the
report; its values must not be pooled with the 750 FK-comparison trials.

The CSV values retain numerical precision for recalculation; displayed decimal
places in Markdown tables are only for readability.
