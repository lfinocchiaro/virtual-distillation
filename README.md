# Virtual distillation protocol for quantum error mitigation in bosonic modes

This repository contains the numerical simulations for the virtual distillation protocol. The full computation is done for initial noisy Fock states on which we measure the number operator. We consider noisy beam splitter operations parametrized by a gaussian noise on the target angles.

We also added comparisons with other input states such as binomial, cat and GKP states.

All the results are found in the main notebook virtual_distillation.ipynb which walks through everything. The functions are separated in functions.py for clarity.

In the testing file we perform some computations on scalability by pushing the protocol to high-mode or high-Fock state regimes, as well as some tests and analysis on eigenvector drift.
