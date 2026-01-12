import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Using Tanh is good, but Sigmoid or Swish can sometimes be smoother for exp decays
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.E_history = [] 

    def forward(self, x):
        return self.net(x)

model = NeuralNetwork().to(device)
R_max = 20.0 # 30 is very far; 20 is sufficient for ground state
eps = 1e-5   # Small shift to avoid div by zero

# We start at -0.9. The target is -0.5. 
E = torch.tensor([-0.9], dtype=torch.float32, device=device, requires_grad=True)

# 2. CRITICAL: Add E to the optimizer
optimizer = torch.optim.Adam(list(model.parameters()) + [E], lr=1e-3)
def u_trial(r):
    """
    Improved Ansatz: u(r) = r * e^(-r) * NN(r)
    1. 'r' handles the cusp at the nucleus (u(0)=0).
    2. 'e^(-r)' handles the decay at infinity.
    3. The NN learns the deviation (which is just a constant for 1s).
    """
    nn_out = model(r)
    # We explicitly encode the asymptotic behavior
    return r * torch.exp(-r) * nn_out

def get_derivatives(r):
    r.requires_grad_(True)
    u = u_trial(r)
    # Compute first derivative
    du = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    # Compute second derivative
    d2u = torch.autograd.grad(du, r, torch.ones_like(du), create_graph=True)[0]
    return u, du, d2u

def pde_residual(r, u, d2u, current_E):
    # Radial Schrodinger Equation: u'' + 2(E + 1/r)u = 0
    # Note: We pass current_E explicitly
    res = d2u + 2.0 * (current_E + 1.0 / r) * u
    return res

# Training Loop
N_colloc = 2000 
loss_history = []
print("Starting training with Pure Backprop for Energy...")

for epoch in range(5001):
    optimizer.zero_grad()

    # --- Standard Sampling ---
    r_near = (5.0 - eps) * torch.rand(N_colloc // 2, 1, device=device) + eps
    r_far = (R_max - 5.0) * torch.rand(N_colloc // 2, 1, device=device) + 5.0
    r_c = torch.cat([r_near, r_far], dim=0)

    u, du, d2u = get_derivatives(r_c)

    # --- PDE Loss ---
    # Now, E is part of the graph. 
    # The optimizer will calculate d(Loss)/dE and nudge E to minimize the residual.
    res = pde_residual(r_c, u, d2u, E)
    loss_pde = (res**2).mean()

    # --- Normalization Loss ---
    # This is now the ONLY thing stopping the network from outputting zero.
    # We increase the weight slightly to be safe.
    r_grid = torch.linspace(eps, 10.0, 1000, device=device).view(-1, 1)
    u_grid = u_trial(r_grid)
    dr = (10.0 - eps) / 1000
    integral = torch.sum(u_grid**2) * dr
    loss_norm = (integral - 1.0)**2

    loss = loss_pde + 10.0 * loss_norm # Increased weight for safety

    loss.backward()
    optimizer.step()
    
    # Save history
    model.E_history.append(E.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Energy: {E.item():.5f} Ha")


E_hist = np.array(model.E_history)
epochs_hist = np.arange(len(E_hist)) * 10

r_plot = np.linspace(eps, 15.0, 500)
r_plot_t = torch.tensor(r_plot, dtype=torch.float32, device=device).view(-1, 1)

with torch.no_grad():
    u_pred = u_trial(r_plot_t).cpu().numpy().flatten()

# Exact solution: u(r) = 2 * r * exp(-r)
u_exact = 2.0 * r_plot * np.exp(-r_plot)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 
ax[0].plot(epochs_hist, E_hist, label="Learned E")
ax[0].axhline(y=-0.5, color='r', linestyle='--', label='Exact (-0.5)')
ax[0].set_title("Energy Convergence")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Energy (Ha)")
ax[0].legend()
ax[0].grid(True)

# 
ax[1].plot(r_plot, u_pred, 'b-', linewidth=2, label="PINN u(r)")
ax[1].plot(r_plot, u_exact, 'r--', linewidth=2, label="Exact u(r)")
ax[1].set_title("Radial Wavefunction (l=0)")
ax[1].set_xlabel("r (Bohr)")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()