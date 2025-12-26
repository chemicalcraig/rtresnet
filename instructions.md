# Overview
Goal is to use a resnet to predict the time-dependent evolution of the complex-valued electronic density matrix, rho, of an molecular system. Training data will be obtained from real-time time-dependent density functional theory simulations performed using NWChem. At each time step in the simulation the density matrix will be dumped to a file. Post-processing of the simulation will provide the AO overlap matrix, S, the time-dependent external field data, and a series of density matrices at each time step. The density matrices will be aggregated into a single numpy array for training, "densities/density_series.npy". The density matrix is propagated in the non-orthonormal AO basis. NWChem calculations can be either "open-shell" or "closed-shell." The density_series.npy file is of shape (N_steps, N_spin, N_bf, N_bf) where N_steps is the number of steps in the rttddft simulation, N_spin = 1 for closed shell and 2 for open shell, N_bf is the number of basis functions.
# Software to use
Use virtual environment in ~/pyvenv for python
tools:
- pytorch
- numpy
- matplotlib
- cuda

# Physics Constraints
Some physics-informed constraints are:
- Hermiticity - error penalty
- Idempotency - error penalty, implement McWeeney purification at each step
- Tr(rho S) = N_electrons - error penalty, with optional trace projection that scales rho to enforce electron conservation
- Positivity: rho must remain positive semi-definite

# Other Considerations:
- input parameters defined in JSON file.
- Include options for training and prediction
- Provide strategy for generalizing to other molecular systems with different geometries and numbers of atoms and basis functions. Graph encoder/decoder layers?
