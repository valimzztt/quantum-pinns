import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input is now 3 dimensions (x, y, z)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.E_history = [] 

    def forward(self, x):
        return self.net(x)

model = NeuralNetwork().to(device)

# Initialize Energy (Start near -0.5)
E = torch.tensor([-0.6], dtype=torch.float32, device=device, requires_grad=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def get_r(pts):
    # pts shape: [batch, 3] -> calculate euclidean distance
    return torch.sqrt(torch.sum(pts**2, dim=1, keepdim=True) + 1e-8)

def psi_trial(pts):
    """
    3D Ansatz: psi(x,y,z) = e^(-r) * NN(x,y,z)
    We don't strictly need the 'r' prefactor here like we did in 1D radial eq
    because the 3D wavefunction doesn't have the u(r)=r*R(r) transformation.
    Ground state is simply e^(-r).
    """
    r = get_r(pts)
    nn_out = model(pts)
    return torch.exp(-r) * nn_out

def get_derivatives_3d(pts):
    pts.requires_grad_(True)
    psi = psi_trial(pts)
    
    # First derivatives (gradient)
    grad_psi = torch.autograd.grad(psi, pts, torch.ones_like(psi), create_graph=True)[0]
    
    # Second derivatives (Laplacian)
    # The laplacian is the trace of the Hessian (sum of d2/dx2 + d2/dy2 + d2/dz2)
    # We compute it by taking the divergence of the gradient
    
    d2psi_dx2 = torch.autograd.grad(grad_psi[:, 0:1], pts, torch.ones_like(grad_psi[:, 0:1]), create_graph=True)[0][:, 0:1]
    d2psi_dy2 = torch.autograd.grad(grad_psi[:, 1:2], pts, torch.ones_like(grad_psi[:, 1:2]), create_graph=True)[0][:, 1:2]
    d2psi_dz2 = torch.autograd.grad(grad_psi[:, 2:3], pts, torch.ones_like(grad_psi[:, 2:3]), create_graph=True)[0][:, 2:3]
    
    lap_psi = d2psi_dx2 + d2psi_dy2 + d2psi_dz2
    
    return psi, lap_psi

# Training Config
N_batch = 2000
L_box = 6.0 # Box size +/- 6.0 Bohr


for epoch in range(5001):
    optimizer.zero_grad()

    # --- 1. Smart 3D Sampling ---
    # 50% points in a small sphere (radius 2) around nucleus
    # 50% points in the larger box
    
    # Gaussian sampling for core (centered at 0,0,0)
    pts_core = torch.randn(N_batch // 2, 3, device=device) * 1.5 
    
    # Uniform sampling for tail
    pts_tail = (torch.rand(N_batch // 2, 3, device=device) * 2 * L_box) - L_box
    
    pts = torch.cat([pts_core, pts_tail], dim=0)

    # --- 2. Physics Loss ---
    psi, lap_psi = get_derivatives_3d(pts)
    r = get_r(pts)
    
    # Hamiltonian: -0.5 * Laplacian - (1/r) * psi
    # Schrodinger: H psi = E psi  =>  H psi - E psi = 0
    
    H_psi = -0.5 * lap_psi - (1.0 / r) * psi
    res = H_psi - E * psi
    
    loss_pde = (res**2).mean()

    # --- 3. Normalization Loss ---
    # Approximate integral over box volume L^3
    # Integral ~ mean(psi^2) * Volume
    # Note: This is rough because of non-uniform sampling, but sufficient to prevent psi=0
    # For better accuracy, we'd use importance sampling weights.
    vol = (2 * L_box)**3
    norm_integral = torch.mean(psi**2) * vol # Very rough approximation
    loss_norm = (norm_integral - 1.0)**2
    
    loss = loss_pde + 10.0 * loss_norm
    loss.backward()
    optimizer.step()

    # --- 4. Rayleigh Quotient Update ---
    if epoch % 20 == 0:
        with torch.no_grad():
            # E = <psi|H|psi> / <psi|psi>
            # We calculate this on the current batch
            num = torch.sum(psi * H_psi)
            den = torch.sum(psi * psi)
            E_new = num / (den + 1e-8)
            
            # Smooth update
            E.data = 0.9 * E.data + 0.1 * E_new
            model.E_history.append(E.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Energy: {E.item():.4f} Ha")

# =========================
# Plotting 3D Results
# =========================
# We will plot a 1D slice along the z-axis (x=0, y=0) to compare with exact solution

z_plot = np.linspace(0, 8.0, 200)
x_plot = np.zeros_like(z_plot)
y_plot = np.zeros_like(z_plot)

pts_plot = np.stack([x_plot, y_plot, z_plot], axis=1)
pts_plot_t = torch.tensor(pts_plot, dtype=torch.float32, device=device)

with torch.no_grad():
    psi_pred = psi_trial(pts_plot_t).cpu().numpy().flatten()

# Exact 3D ground state: psi = (1/sqrt(pi)) * e^(-r)
# Note: The normalization constant might be different depending on how the NN converged
# so we normalize the max value to match for visual shape comparison.
psi_exact = (1.0 / np.sqrt(np.pi)) * np.exp(-z_plot)

# Normalize both to peak at 1 for shape comparison
psi_pred_norm = psi_pred / np.max(np.abs(psi_pred))
psi_exact_norm = psi_exact / np.max(np.abs(psi_exact))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Energy
ax[0].plot(np.array(model.E_history))
ax[0].axhline(y=-0.5, color='r', linestyle='--')
ax[0].set_title("Energy Convergence")
ax[0].set_ylabel("Energy (Ha)")

# Wavefunction Slice
ax[1].plot(z_plot, psi_pred_norm, label="PINN Slice (x=0, y=0)")
ax[1].plot(z_plot, psi_exact_norm, 'k--', label="Exact e^(-r)")
ax[1].set_title("Wavefunction Cross-Section")
ax[1].set_xlabel("z (Bohr)")
ax[1].legend()

plt.show()