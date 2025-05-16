# Implementation of HubbardNet
This is an implementation of the Neural Network proposed on the next article:

[**HubbardNet: Efficient predictions of the Bose-Hubbard model spectrum with deep neural networks**](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043084)

## Mathematical description
Bose-Hubbard model is used to describe physical systems such as bosonic atoms in an optical lattice, the hamiltonian is given by:

![Hamiltonian](/img/hamiltonian_mathtext.png)

In this code I only used the network for finding the ground state of the case of a 1-dimensional lattice with periodic boundary.

## Results

This is the plot of the ground state founded with the network and compared with exact diagonalization, considering $\mu$ and $V_i$ as 0, and the values of the ground state as function of $U$.

![Hubbard-Net vs Exact diagonalization, in this case we consider  $\mu$ and $V_i$ as 0, and the values of the ground state as function of $U$](/img/result.png)


