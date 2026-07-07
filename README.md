# Virtual distillation protocol for quantum error mitigation in bosonic modes

This repository contains numerical simulations for the virtual distillation protocol applied on bosonic syxstems. The plots used in the paper _Error Mitigation in Bosonic Systems via Virtual Distillation_ (2607.04914) were produced using Jupyter notebooks and auxiliary function files (Python and Julia). The files corresponding to the figures are organized as follows:
 - `OBSERVABLE ESTIMATION` for Figs. 2, 4, 6, 9, where expectation values of number operator, parity operator and quadrature operators are estimated on several input states with various noise sources,
 - `PHASE ESTIMATION` for Figs 3, 5, where Method 2 is applied to reconstructing the Wigner function of a Fock state and the characteristic function of the photon number distribution for a $N00N$ state,
 - `CORRELATORS` for Fig 8, where two-mode correlators are estimated within an atomic boson sampling model.


In the `_old_code` folder, code and plots used for the internship report (`virtual_distillation.ipynb`), as well as some tests on scalability and eigenvector drift (`tests.ipynb`) are also accessible.
