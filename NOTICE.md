# Attribution and third-party notices

Paths below identify components of the separately maintained implementation;
those source files are not included in this community documentation repository.

SeismicX GRTM 0.5.1 is distributed under the SeismicX GRTM Research and
Noncommercial License 1.0 (see LICENSE), as selected by the project owner.
The C mathematical sources retain their existing author attributions, including
Yu Ziye and Cangye. This release does not change independently granted
third-party rights.

The CUDA complex implementation in
`grtm_all/grtm_cuda/cusrc/scilib/cucomplex.hpp` was written by John C. Travers
(2012), derived from LLVM libc++ revision 147853. Its original attribution and
MIT/University of Illinois license notice remain in that file. The complete
legacy licence is reproduced in `licenses/LLVM-libcxx.txt`, with the additional
copyright attribution to John C. Travers (2012). CUDA binary builds
also link NVIDIA's CUDA runtime. NVIDIA's toolkit and runtime license terms apply
to that component; the included terms are in `licenses/NVIDIA-CUDA.txt`.
This package does not distribute the GPU driver or CUDA compiler.

Compiled Python modules contain Cython-generated support code. Its applicable
licence and generated-code exception are reproduced in `licenses/Cython.txt`.
NumPy, SciPy, PyTorch, and JAX are separately installed dependencies and remain
under their respective licenses; they are not bundled into this wheel.

This notice records the existing provenance. It does not claim that all source
files were authored by the packaging implementer.

The Fortran implementation used for historical performance comparisons is an
external reference from [YunyiQian/grtm](https://github.com/YunyiQian/grtm), credited in its
original README to Yunyi Qian (SUSTech,
2020-12-20). Its source and build files are not distributed in the current
repository or source packages. Archived comparison measurements retain their
original provenance and do not imply authorship of that implementation.
