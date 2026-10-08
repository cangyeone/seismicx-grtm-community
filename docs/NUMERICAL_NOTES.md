# Physical conventions, accuracy and derivative limits

English | [简体中文](NUMERICAL_NOTES.zh-CN.md) | [Home](../README.md)

## Choose a supported problem

The point-source solver assumes horizontal, isotropic solid layers, with the
last layer a half-space. Vp > Vs > 0, density and Q must be positive. Fluid layers,
anisotropy, dipping interfaces and the zero-distance axial limit are not supported.
Receivers exactly on a material interface use the upper-side material convention.

The receiver-function solver is a separate **elastic**, plane-P-wave model.
It ignores optional Q columns and requires subcritical incidence in every layer.
It does not implement S receiver functions, automatic phase picking, station
data download or instrument correction.

## Finite Q and stress

The default point-source `constitutive="kernel"` uses real reference
`mu = rho * Vs0**2`, dispersive complex velocities `a(omega), b(omega)`, and
`lambda(omega) = mu * ((a/b)**2 - 2)`. This matches the inherited kernel convention.
It is not the usual real-density viscoelastic identification `mu* = rho*b**2`;
an effective complex density interpretation would use `rho_eff = mu/b**2`.
Do not treat agreement between CPU and GPU as independent validation of an
arbitrary attenuation law. `constitutive="elastic"` instead uses reference
elastic Lamé parameters and is an explicit finite-Q approximation.

Stress uses the receiver layer's constitutive parameters and is positive in
tension. Strain uses tensor shear components. Without source-history convolution,
reported physical arrays are impulse kernels rather than static responses.

## Integration convergence

Matching two implementations on the same quadrature checks arithmetic and
implementation consistency; it does not prove that the wavenumber integral is
converged. Increase time-window length, sampling resolution and integration
resolution as appropriate to the problem, and compare the observables used by
the scientific analysis. Source depth, epicentral range, layer contrast and
frequency band can change convergence and cost.

Saved fixed grids are useful for controlled local derivative comparisons. An
optimizer can select a different grid at its next base model, so inspect gradient
behavior throughout an inversion. Shallow-source/high-frequency and extremely
long-period problems deserve separate checks. Extended precision stabilizes
some internal operations but does not certify all forward values or gradients.
Extreme long-period structural gradients remain unverified.

## Structural sensitivities

Point-source tangents differentiate a fixed local quadrature and numerical branch
structure. They do not differentiate layer count, changes in source/receiver
layer membership, integer wavenumber cutoffs or PTAM extremum selection.
Interface-depth derivatives are rejected at a source/receiver interface. A zero
unselected Jacobian column means “not differentiated”, not “physically irrelevant”.

Tangents execute on CPU even with CUDA/Metal forward selection. `check_step=True`
adds an independent finite-difference comparison and additional solves; agreement
there is not a measure of integration truncation error. Only first derivatives
are supported. JAX defaults to dense structural Jacobians; the 0.6 preview adds
optional matrix-free first-order reverse mode. NumPy and PyTorch VJPs avoid
retaining the full structural Jacobian. See the [preview guide](RESEARCH_PREVIEW.md)
for native adjoint selection, fixed optimization grids and separate adaptive
quadrature limits.

## Receiver-function inversion

Match component signs, Gaussian bandwidth, normalization and time windows
between data and synthetics. Increase FFT padding/record length for long coda.
H–κ bounds and fixed crustal Vp can strongly affect the solution. A converged
optimizer and a full local rank do not establish that the model family is correct.
Covariance-derived standard deviations are nominal local conditional scales;
they do not include uncertainty in fixed Vp, density, mantle structure or model
discrepancy. Compare residuals by event and inspect bounds/rank diagnostics.

For measured agreement and timing scope see the [FK report](../benchmarks/fk/README.md).
For installed-interface checks see [release verification](../verification/README.md).
