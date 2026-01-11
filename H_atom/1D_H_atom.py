# In this script, we will build a PINN that solves the hydrogen atom
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math 


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# We create a neural network that will be a MLP
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # We use this to store the energy
        self.E_history = [] 

    def forward(self, x):
        return self.net(x)


model = NeuralNetwork().to(device)
print(model)

# We define a trainable eigenvalue E (start near true ground state -0.5 Ha)
E_ref =-0.5
E = nn.Parameter(torch.tensor([-0.7], dtype=torch.float32, device=device))
# Right now the learning rate is fixed
# In Paolo's thesis it is decreased during training according to
# nu(s)=n(0)*(1+s/deltaS)^-1
optimizer = torch.optim.Adam(list(model.parameters()) + [E], lr=1e-3)
R_max = 30.0
eps = 1e-4


#Let's define all residuals given by PDE loss, Boundary conditions, eigenvalues
def u_trial(r):
    """
    Enforce boundary conditions via a trial function:
    u(eps) ~ 0 due to factor (r - eps) so that u(r_max) = 0 due to factor (r_max - r)
    """
    nn_out = model(r)
    return (r - eps) * (R_max - r) * nn_out

def pde_residual(r):
    r.requires_grad_(True)
    u = u_trial(r)
    du = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, r, torch.ones_like(du), create_graph=True)[0]

    # l=0: u'' + 2(E + 1/r)u = 0
    # l=0: equivalent to (-0.5u''-1/r)R(r)=ER(r) (from Paolo's Thesis)
    res = d2u + 2.0 * (E + 1.0 / r) * u
    return res
def normalization_penalty(r_grid):
    # Enforce ∫ u(r)^2 dr = 1  on [eps, r_max] that wavefunction is normalized inside the domain
    
    u = u_trial(r_grid)
    # simple trapezoid rule
    dr = (R_max - eps) / (r_grid.shape[0] - 1)
    integral = dr * (0.5*u[0]**2 + (u[1:-1]**2).sum() + 0.5*u[-1]**2)
    return (integral - 1.0)**2



def analytical_solution(r):
    R = 2*r*math.e**(-r)



# Training points
N_colloc = 2000 # Paolo used 2000 points
N_norm = 2000

for epoch in range(20000):
    optimizer.zero_grad()

    # Collocation points (sample uniformly; you can improve later with non-uniform sampling)
    r_c = (eps + (R_max - eps) * torch.rand(N_colloc, 1, device=device))

    res = pde_residual(r_c)
    loss_pde = (res**2).mean()

    # Normalization grid (fixed uniform)
    r_grid = torch.linspace(eps, R_max, N_norm, device=device).view(-1, 1)
    loss_norm = normalization_penalty(r_grid)

    # We know the true ground state so take this as a reference
    loss_energy_prior = torch.relu(E - E_ref)**2

    # Total loss
    loss = loss_pde + 10.0 * loss_norm + 1.0 * loss_energy_prior
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"it={epoch:5d}  loss={loss.item():.3e}  loss_pde={loss_pde.item():.3e}  "
              f"loss_norm={loss_norm.item():.3e}  E={E.item():.6f}")

    if epoch % 10 == 0:
            # append energy to the energy tracker
            model.E_history.append(E.item())

# Plot the result
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Plot energy convergence 
ax[0].plot(model.E_history)
ax[0].axhline(y=-0.5, color='r', linestyle='--', label='RHF Limit (-2.86)')
ax[0].set_title("Energy Convergence")
ax[0].set_xlabel("Iterations (x10)")
ax[0].set_ylabel("Energy (Hartrees)")
ax[0].legend()

# Plot 2: Radial Wavefunction
# Generate a line of points from r=0 to r=4 along x-axis
r_test = np.linspace(0.01, 4.0, 100)
x_test = r_test
y_test = np.zeros_like(r_test)
z_test = np.zeros_like(r_test)
pts_test = np.stack([x_test, y_test, z_test], axis=1)



ax[1].plot(model.E_history)
# Compare with rough analytical approx for He: psi ~ e^(-1.69 r)
ax[1].set_title("1s Orbital Shape")
ax[1].set_xlabel("Distance r (Bohr)")
ax[1].legend()

plt.show()