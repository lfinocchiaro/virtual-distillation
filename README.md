# Virtual distillation protocol for quantum error mitigation in bosonic modes

This repository contains the numerical simulations for the virtual distillation protocol. The full computation is done for initial noisy Fock states on which we measure the number operator. We consider noisy beam splitter operations parametrized by a gaussian noise on the target angles.

We also added comparisons with other input states such as binomial, cat and GKP states.

Lastly, we perform some computations on scalability by pushing the protocol to high-mode or high-Fock state.

The notebook virtual_distillation.ipynb walks through everything. The functions in functions.py are separated for clarity.