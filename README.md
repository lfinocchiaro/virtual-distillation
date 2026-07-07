# Virtual distillation protocol for quantum error mitigation in bosonic modes

This repository contains the numerical simulations for the virtual distillation protocol. The main file is `paper_notebook.ipynb`, which produces the plots used in the paper _Error Mitigation in Bosonic Systems via Virtual Distillation_ (2607.04914). It uses auxiliary functions from `paper_functions.py`.

The Virtual Distillation protocol is performed on Fock, binomial and cat states with losses for number operator measurements, on even, tri and square cat states for parity operator measurement, and on squeezed coherent states with dephasing for quadrature measurements. A noisy implementation is carried out to assess the robustness of the protocol. Additional plots, in Julia, can be found in the `JuliaSimulations` folder.

In the `old_code` folder, the code and plots used for the internship report (`virtual_distillation.ipynb`), as well as some tests on scalability and eigenvector drift (`tests.ipynb`) are also accessible.
