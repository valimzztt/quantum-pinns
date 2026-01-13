import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for H2+ (R=1)")
# In this script, we include an energy loss term 

class PINN_H2_Param(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.E_history = []

    def forward(self, x):
        return self.net(x)

# In Paolo´s thesis, he says that R is defined as HALF the internuclear distance"
# Therefore, nuclei are at +/- 1.0
Z_pos = 1.0 

model = PINN_H2_Param().to(device)

def compute_distances(r, z):
    d1 = torch.sqrt(r**2 + (z - Z_pos)**2)
    d2 = torch.sqrt(r**2 + (z + Z_pos)**2)
    return d1, d2

def psi_trial(r, z):
    d1, d2 = compute_distances(r, z)
    # LCAO Ansatz: exp(-|r-R1|) + exp(-|r-R2|)
    lcao = torch.exp(-d1) + torch.exp(-d2)
    nn_out = model(torch.cat([r, z], dim=1))
    return lcao * nn_out

def laplacian_2d(psi, r, z):
    grads = torch.autograd.grad(psi, [r, z], torch.ones_like(psi), create_graph=True)
    dpsi_dr, dpsi_dz = grads[0], grads[1]
    d2psi_dr2 = torch.autograd.grad(dpsi_dr, r, torch.ones_like(dpsi_dr), create_graph=True)[0]
    d2psi_dz2 = torch.autograd.grad(dpsi_dz, z, torch.ones_like(dpsi_dz), create_graph=True)[0]
    return d2psi_dr2 + (1.0 / (r + 1e-6)) * dpsi_dr + d2psi_dz2

def train(model, N_f, epochs):
    print(f"\n--- Training with Manual Weight Control ---")
    
    # 1. Initialize Parameters
    E_param = torch.tensor([-0.6], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(list(model.parameters()) + [E_param], lr=1e-3)
    
    # 2. Define Initial Weights
    w_norm = 10.0      # Weight for Normalization (keep high usually)
    w_energy = 100.0   # Initial Weight for Energy Constraint
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # --- MANUAL WEIGHT SCHEDULING ---
        # Example: Relax the energy constraint after 1000 epochs.
        # We want to force E < 0 early on, but later we trust the PDE more.
        if epoch == 1000:
            print(">> Relaxing Energy Constraint Weight...")
            w_energy = 10.0
        elif epoch == 3000:
             print(">> Removing Energy Constraint (Let Physics drive)...")
             w_energy = 0.0

        # 1. Sampling
        r = torch.abs(torch.randn(N_f, 1, device=device) * 2.5)
        z = torch.randn(N_f, 1, device=device) * 3.0
        r.requires_grad_(True)
        z.requires_grad_(True)
        
        # 2. Physics
        psi = psi_trial(r, z)
        lap_psi = laplacian_2d(psi, r, z)
        d1, d2 = compute_distances(r, z)
        V = -1.0/d1 - 1.0/d2
        
        # 3. Calculate Losses
        
        # A) PDE Loss (The Physics)
        res = -0.5 * lap_psi + (V - E_param) * psi
        loss_pde = (res**2).mean()
        
        # B) Normalization Loss (The Constraint)
        loss_norm = (torch.mean(psi**2) - 0.1)**2 
        
        # C) Energy Constraint Loss (The "Energy Loss")
        # Penalize if Energy is positive (unphysical for bound states)
        # ReLU(E) is 0 if E is negative (Good).
        # ReLU(E) is E if E is positive (Bad -> Loss increases).
        loss_energy = torch.relu(E_param + 0.05) 
        
        # 4. Weighted Sum
        loss = loss_pde + (w_norm * loss_norm) + (w_energy * loss_energy)
        
        loss.backward()
        optimizer.step()
        
        model.E_history.append(E_param.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | E: {E_param.item():.4f} | w_E: {w_energy}")

    return model


def compute_3d_norm(func_to_eval_psi, z_limit=10.0, r_limit=5.0, N=200):
    """
    Computes the normalization factor A such that Integral(|A*psi|^2) dV = 1.
    Uses cylindrical integration: Integral(|psi|^2 * 2*pi*r) dr dz
    """
    z_lin = np.linspace(-z_limit, z_limit, N)
    r_lin = np.linspace(0, r_limit, N) # r starts at 0
    Z_grid, R_grid = np.meshgrid(z_lin, r_lin)
    
    # Convert to Tensor
    z_t = torch.tensor(Z_grid.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    r_t = torch.tensor(R_grid.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    
    with torch.no_grad():
        if func_to_eval_psi == "reference":
            # Manual LCAO calc
            d1 = np.sqrt(R_grid**2 + (Z_grid - Z_pos)**2)
            d2 = np.sqrt(R_grid**2 + (Z_grid + Z_pos)**2)
            psi_val = np.exp(-d1) + np.exp(-d2)
        else:
            # Neural Network calc
            psi_val = func_to_eval_psi(r_t, z_t).cpu().numpy().reshape(N, N)
            
    # Cylindrical Integral: sum( psi^2 * 2*pi*r * dr * dz )
    integrand = (psi_val**2) * (2 * np.pi * R_grid)
    
    # Integrate over R then Z
    dr = r_lin[1] - r_lin[0]
    dz = z_lin[1] - z_lin[0]
    
    # Simple summation approximation (Riemann sum)
    total_integral = np.sum(integrand) * dr * dz
    
    return np.sqrt(total_integral)

if __name__ == "__main__":
    start_time = time.time()
    
    # Train
    model_trained = train(model, N_f=5000, epochs=5000)
    # 1. Define the Plotting Axis (Internuclear axis)
    # The paper plots x from -10 to 10
    axis_vals = np.linspace(-10, 10, 400)
    
    # Prepare tensors for the axis (r=0 along the axis)
    z_axis_t = torch.tensor(axis_vals, dtype=torch.float32, device=device).view(-1, 1)
    r_axis_t = torch.zeros_like(z_axis_t)
    
    # 2. Get Raw Predictions along the axis
    model_trained.eval()
    with torch.no_grad():
        psi_sim_raw = psi_trial(r_axis_t, z_axis_t).cpu().numpy().flatten()
    
    # 3. Get Raw Reference (LCAO) along the axis
    d1 = np.abs(axis_vals - Z_pos)
    d2 = np.abs(axis_vals + Z_pos)
    psi_ref_raw = np.exp(-d1) + np.exp(-d2)
    
    # 4. COMPUTE 3D NORMALIZATION FACTORS
    # We integrate the 3D volume to find the correct physical scaling
    norm_sim = compute_3d_norm(psi_trial)
    norm_ref = compute_3d_norm("reference")
    
    print(f"Norm Factor Sim: {norm_sim:.4f}")
    print(f"Norm Factor Ref: {norm_ref:.4f}")
    
    # 5. Apply Normalization
    psi_sim_norm = psi_sim_raw / norm_sim
    psi_ref_norm = psi_ref_raw / norm_ref
    
    # 6. Plot
    plt.figure(figsize=(8, 6))
    
    # Simulation (Blue Solid)
    plt.plot(axis_vals, psi_sim_norm**2, color='#4472C4', linewidth=3, 
             label='Simulation')
    
    # Reference (Orange Dotted)
    # Using 'dotted' or thick dots to match paper style
    plt.plot(axis_vals, psi_ref_norm**2, color='#ED7D31', linestyle=':', linewidth=4, 
             label='Reference')
    
    # Styling to match Figure 4.20
    plt.xlabel(r'$\boldsymbol{x \ [a_0]}$', fontsize=16)
    plt.ylabel(r'$\boldsymbol{\Psi^2 \ [a_0^{-3}]}$', fontsize=16)
    plt.title(f'Result for R={Z_pos:.0f} (Half-dist)', fontsize=14)
    
    plt.xlim(-10, 10)
    # The peak for R=1 in the paper is approx 0.20
    # Our 3D normalization should hit this automatically.
    plt.ylim(0, 0.25) 
    
    plt.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.tick_params(direction='in', top=True, right=True, labelsize=12)
    
    plt.tight_layout()
    plt.show()