import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Configuration ---
# Helium Ground State Energy is approx -2.9037 Hartree
E_REF_HELIUM = -2.9037 
E_INIT = -2.5 # Start close to expected value
EPOCHS = 10000
N_COLLOCATION = 4000
N_NORM_SAMPLES = 4000 # More points needed for 6D integral
LR = 1e-3
L_BOX = 4.0 # Electrons are tightly bound in Helium (smaller box than H is okay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for Helium Atom (6D)")

class HeliumPINN(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        
        # Input is 6 Dimensions: (x1, y1, z1, x2, y2, z2)
        layers = [nn.Linear(6, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.net.apply(self.init_weights)
        
        # Trainable Energy Parameter
        self.E = nn.Parameter(torch.tensor([E_INIT]))
        self.E_history = []

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
            if m.bias is not None: m.bias.data.fill_(0.0)

    def forward(self, x):
        # x shape: [Batch, 6] -> (x1, y1, z1, x2, y2, z2)
        
        # 1. Physics Ansatz: e^(-2*r1) * e^(-2*r2)
        # Z=2 for Helium nucleus. This handles the cusp at the nucleus.
        r1 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
        r2 = torch.sqrt(x[:, 3:4]**2 + x[:, 4:5]**2 + x[:, 5:6]**2 + 1e-6)
        envelope = torch.exp(-2.0 * (r1 + r2))
        
        # 2. Permutation Symmetry Enforcement
        # We run the network twice: once with (r1, r2) and once with (r2, r1)
        # and sum the results. This FORCES Psi(r1, r2) = Psi(r2, r1)
        
        # Swap the inputs for the second pass
        # Indices: 0,1,2 (elec 1) <-> 3,4,5 (elec 2)
        x_swapped = torch.cat([x[:, 3:6], x[:, 0:3]], dim=1)
        
        # Output is symmetric sum
        # We add 1.0 to allow the network to learn a perturbation around the simple orbital
        nn_output = self.net(x) + self.net(x_swapped)
        
        return envelope * (1.0 + nn_output)

def compute_distances(x):
    # Extract coordinates
    x1, y1, z1 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    x2, y2, z2 = x[:, 3:4], x[:, 4:5], x[:, 5:6]
    
    # Distance to nucleus (at 0,0,0)
    r1 = torch.sqrt(x1**2 + y1**2 + z1**2 + 1e-6)
    r2 = torch.sqrt(x2**2 + y2**2 + z2**2 + 1e-6)
    
    # Distance between electrons
    r12 = torch.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 + 1e-6)
    
    return r1, r2, r12

def physics_loss(model, x):
    x.requires_grad_(True)
    psi = model(x)
    
    # --- 1. Kinetic Energy (Laplacian in 6D) ---
    # We need grads w.r.t all 6 inputs
    grads = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    
    laplacian = 0
    # Loop over all 6 coordinates (x1...z2)
    for i in range(6):
        grad_2 = torch.autograd.grad(grads[:, i], x, torch.ones_like(grads[:, i]), create_graph=True)[0]
        laplacian += grad_2[:, i].view(-1, 1)
        
    kinetic = -0.5 * laplacian
    
    # --- 2. Potential Energy ---
    r1, r2, r12 = compute_distances(x)
    
    # V = -Z/r1 - Z/r2 + 1/r12 (Z=2 for Helium)
    potential = (-2.0 / r1) - (2.0 / r2) + (1.0 / r12)
    
    # --- 3. Residual ---
    residual = (kinetic + potential * psi) - (model.E * psi)
    loss_pde = torch.mean(residual**2)
    
    # --- 4. Normalization (Monte Carlo 6D) ---
    # We generate a separate UNIFORM batch for integration
    # Volume = (2*L)^6 -> Huge! So we integrate over a smaller relevant box [-4, 4]
    box_side = 2 * L_BOX
    vol_6d = box_side ** 6 
    
    x_norm = (torch.rand(N_NORM_SAMPLES, 6, device=x.device) * box_side) - L_BOX
    psi_norm = model(x_norm)
    
    # Integral = Volume * Mean(psi^2)
    integral = vol_6d * torch.mean(psi_norm**2)
    loss_norm = (integral - 1.0)**2
    
    return loss_pde + loss_norm

def train_helium():
    model = HeliumPINN().to(device)
    # Using a slightly lower LR usually helps with many-body stability
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Starting Helium Training (Target E ~ {E_REF_HELIUM} Ha)")
    
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        
        # --- Sampling Strategy: Mixed Gaussian + Uniform ---
        # 6D Gaussian sampling: Both electrons clustered near nucleus
        # sigma=1.0 is good for Helium (tighter than H)
        x_gauss = torch.randn(N_COLLOCATION, 6, device=device) * 1.0
        
        loss = physics_loss(model, x_gauss)
        loss.backward()
        optimizer.step()
        
        # Track history
        model.E_history.append(model.E.item())
        
        if epoch % 100 == 0:
            print(f"Ep {epoch} | Loss: {loss.item():.5f} | E: {model.E.item():.5f} Ha")
            
    return model

if __name__ == "__main__":
    start = time.time()
    model = train_helium()
    duration = time.time() - start
    print(f"Training Time: {duration:.2f}s")
    
    # --- Plot Energy Convergence ---
    plt.figure(figsize=(8, 6))
    plt.plot(model.E_history, label="PINN Energy", linewidth=2)
    plt.axhline(E_REF_HELIUM, color='k', linestyle='--', label=f"Exact ({E_REF_HELIUM})")
    plt.xlabel("Epochs")
    plt.ylabel("Energy (Ha)")
    plt.title("Helium Ground State Energy Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Plot Radial Distribution P(r1) ---
    # To visualize the result, we integrate out electron 2
    # We essentially plot |Psi(r1, 0)|^2 along one axis
    r_vals = np.linspace(0, 4, 100)
    
    # Construct input batch: Electron 1 moves along z, Electron 2 fixed at some avg distance (e.g. 1.0)
    # Note: To get the true radial density P(r) requires integrating out r2. 
    # Here we just look at a "slice" of the wavefunction.
    inputs = np.zeros((100, 6))
    inputs[:, 2] = r_vals      # z1 varies
    inputs[:, 5] = 1.0         # z2 fixed at 1.0 (approx avg distance)
    
    inputs_t = torch.tensor(inputs, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        psi = model(inputs_t).cpu().numpy().flatten()
        
    plt.figure(figsize=(8, 6))
    plt.plot(r_vals, psi**2, linewidth=2)
    plt.xlabel(r"$r_1$ (Bohr)")
    plt.ylabel(r"$|\Psi(r_1, r_2=1.0)|^2$")
    plt.title("Helium Wavefunction Slice")
    plt.grid(True, alpha=0.3)
    plt.show()