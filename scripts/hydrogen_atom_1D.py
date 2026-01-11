import torch
import torch.nn as nn

# ----------------------------
# Simple MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        net += [nn.Linear(layers[-2], layers[-1])]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# ----------------------------
# PINN for hydrogen radial equation
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

r_max = 30.0
eps = 1e-4

model = MLP([1, 64, 64, 64, 1]).to(device)

# Trainable eigenvalue E (start near true ground state -0.5 Ha)
E = nn.Parameter(torch.tensor([-0.7], dtype=torch.float32, device=device))

optimizer = torch.optim.Adam(list(model.parameters()) + [E], lr=1e-3)

def u_trial(r):
    """
    Enforce boundary conditions via a trial function:
    u(eps) ~ 0 due to factor (r - eps)
    u(r_max) = 0 due to factor (r_max - r)
    """
    nn_out = model(r)
    return (r - eps) * (r_max - r) * nn_out

def pde_residual(r):
    r.requires_grad_(True)
    u = u_trial(r)
    du = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, r, torch.ones_like(du), create_graph=True)[0]

    # l=0: u'' + 2(E + 1/r)u = 0
    res = d2u + 2.0 * (E + 1.0 / r) * u
    return res

def normalization_penalty(r_grid):
    # Enforce ∫ u(r)^2 dr = 1  on [eps, r_max]
    u = u_trial(r_grid)
    # simple trapezoid rule
    dr = (r_max - eps) / (r_grid.shape[0] - 1)
    integral = dr * (0.5*u[0]**2 + (u[1:-1]**2).sum() + 0.5*u[-1]**2)
    return (integral - 1.0)**2

# Training points
N_colloc = 2000
N_norm = 2000

for it in range(5000):
    optimizer.zero_grad()

    # Collocation points (sample uniformly; you can improve later with non-uniform sampling)
    r_c = (eps + (r_max - eps) * torch.rand(N_colloc, 1, device=device))

    res = pde_residual(r_c)
    loss_pde = (res**2).mean()

    # Normalization grid (fixed uniform)
    r_grid = torch.linspace(eps, r_max, N_norm, device=device).view(-1, 1)
    loss_norm = normalization_penalty(r_grid)

    # Optional: discourage positive E (bound states have E<0)
    loss_energy_prior = torch.relu(E)**2

    # Total loss
    loss = loss_pde + 10.0 * loss_norm + 1.0 * loss_energy_prior
    loss.backward()
    optimizer.step()

    if it % 500 == 0:
        print(f"it={it:5d}  loss={loss.item():.3e}  loss_pde={loss_pde.item():.3e}  "
              f"loss_norm={loss_norm.item():.3e}  E={E.item():.6f}")
