# Third-party notices

`src/moead.py` documents its adaptation of the subproblem, neighborhood,
arithmetic crossover and external archive structure from
`multi-objective-optimization-main/mopt/algorithms/moead.py`.
The upstream MIT notice is included unchanged in `licenses/mopt-MIT.txt`:
Copyright (c) 2026 Mohammad Sadegh Hajibabaie.
Project modifications and additional operators do not remove this attribution.

NumPy, SciPy and scikit-learn are installed as dependencies and are not vendored.
Their licenses remain applicable to those packages. No PyTorch models, pymoo
runtime, third-party checkpoints or measured spectral libraries are bundled.

The lightweight VCA-style/FCLS/NNLS numerical implementation is the project's
query baseline. It is not presented as an official upstream implementation of
every algorithm bearing those names.
