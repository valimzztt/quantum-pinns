import pandas as pd 
from utils.benchmarking.HPinnBenchmark_3D import HPinnBenchmark3D as HPinnBenchmark
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils.filemanager import load_config
# Load Config
filepath = os.path.join(current_dir, "configs", "3D_H_atom.yaml")
config = load_config(filepath)

# Retrieve params
epochs = config['training']['epochs']
N_f = config['training']['n_collocation'] 
lr = config['training']['learning_rate']
L_max = config['physics']['L_max']
L_min = config['physics']['L_min']
num_dense_layers =config['training']['num_dense_layers']
num_dense_nodes = config['training']['num_dense_nodes']
w_pde = config['loss_weights']['pde']
w_norm = config['loss_weights']['normalization']
w_energy = config['loss_weights']['energy_constraint']
sampling_strategy = config['training']['train_distribution'] 
E_ref= config['physics']['E_ref']
E_init= config['physics']['E_init']
initializer = config['training']['initializer']
normalization = config['training']['normalization']
L_max= config['physics']['L_max']
L_min= config['physics']['L_min']
# Import your model class
from H_atom_refactored.H_atom_3D_energytrained import HydrogenPINN3D

# --- 1. Define a Local Training Function to Track History ---
# We define this HERE to ensure we capture E_history every epoch
def train_tracker(model, N_f, epochs, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    box_volume = (L_max - L_min)**3 
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # Sampling (Gaussian mix)
        inputs = (torch.randn(N_f, 3, device=device) * 2.0).requires_grad_(True)
        
        # Physics Loss
        psi = model(inputs)
        
        # Laplacian
        grads = torch.autograd.grad(psi, inputs, torch.ones_like(psi), create_graph=True)[0]
        laplacian = 0
        for i in range(3):
            grad_2 = torch.autograd.grad(grads[:, i], inputs, torch.ones_like(grads[:, i]), create_graph=True)[0]
            laplacian += grad_2[:, i].view(-1, 1)
        
        kinetic = -0.5 * laplacian
        r = torch.sqrt(inputs[:, 0:1]**2 + inputs[:, 1:2]**2 + inputs[:, 2:3]**2 + 1e-6)
        potential = -1.0 / r
        
        residual = (kinetic + potential * psi) - (model.E * psi)
        loss_pde = torch.mean(residual**2)
        
        # Normalization (Separate Uniform Batch)
        inputs_norm = (torch.rand(5000, 3, device=device) * (L_max - L_min)) + L_min
        psi_norm = model(inputs_norm)
        integral = box_volume * torch.mean(psi_norm**2)
        loss_norm = (integral - 1.0)**2
        
        loss = loss_pde + loss_norm*w_norm
        loss.backward()
        optimizer.step()
        
        # --- TRACKING ---
        # Explicitly save the energy value this epoch
        model.E_history.append(model.E.item())
        
    return model

# --- 2. Setup Benchmark ---
benchmark = HPinnBenchmark(device='cpu', n_test=100000)
results = []
energy_histories = [] 

# Setup Plotting (1 Row, 2 Columns)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Prepare Z-axis for plotting wavefunction
z_vals = np.linspace(-6, 6, 300)
inputs_z = np.zeros((300, 3))
inputs_z[:, 2] = z_vals 
inputs_torch = torch.tensor(inputs_z, dtype=torch.float32)

# Analytical Solution
r_vals = np.abs(z_vals)
psi_exact = (1.0 / np.sqrt(np.pi)) * np.exp(-r_vals)
prob_exact = psi_exact**2
ax[0].plot(z_vals, prob_exact, 'k--', linewidth=2.5, label='Analytical (Exact)')

# Define Architectures
configs = [
    (16, 1, "Small (16, 1)"),
    (32, 2, "Medium (32, 2)"),
    (64, 2, "Medium (Wide) (64, 2)"),
    (32, 3, "Deep (32, 3)"),
    (64, 3, "Deep (Wide) (64, 3)"),
]

# FORCED START ENERGY for fair comparison
forced_E_init = -0.8

for width, layers, name in configs:
    print(f"Benchmarking {name}...")
    
    # Initialize
    model = HydrogenPINN3D(width=width, depth=layers).to('cpu')

    model.E = nn.Parameter(torch.tensor([forced_E_init], dtype=torch.float32))
    model.E_history = [] # clear history
    
    start = time.time()
    
    # Use our local tracker function
    train_tracker(model, N_f, epochs, device='cpu')
    
    duration = time.time() - start
    
    # Store History
    energy_histories.append((name, model.E_history))
    
    # Evaluate Metrics
    metrics = benchmark.evaluate(model, duration)
    metrics["Architecture"] = name
    results.append(metrics)
    
    # Plot Wavefunction (Left Subplot)
    model.eval()
    with torch.no_grad():
        psi_pred = model(inputs_torch).numpy().flatten()
        prob_density = psi_pred**2
        ax[0].plot(z_vals, prob_density, linewidth=2, alpha=0.8, label=f"{name}")

# --- 3. Plot Energy Evolution (Right Subplot) ---
for name, history in energy_histories:
    # Create x-axis based on actual history length
    epochs_axis = np.linspace(0, epochs, len(history))
    
    # Plot
    ax[1].plot(epochs_axis, history, linewidth=2, alpha=0.8, label=name)

# Formatting
ax[0].set_xlabel(r'$z \ [a_0]$', fontsize=14)
ax[0].set_ylabel(r'$|\psi|^2$', fontsize=14)
ax[0].set_title(f'Wavefunction (Epochs={epochs})', fontsize=16)
ax[0].grid(True, alpha=0.3)
ax[0].legend(fontsize=10)
ax[0].set_yscale('log') 
ax[0].set_ylim(1e-5, 2.0) 

ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Energy (Ha)', fontsize=14) 
ax[1].set_title(f'Energy Convergence (Start E={forced_E_init})', fontsize=14)
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=10)
ax[1].set_ylim(forced_E_init - 0.1, -0.4) 
ax[1].axhline(-0.5, color='k', linestyle='--', linewidth=2, label="Exact (-0.5)")

plt.tight_layout()
plt.show()

df = pd.DataFrame(results)
print("\nFinal Benchmark Results (3D):")
df["Energy_Error"] = (df["Final_Energy"] - (-0.5)).abs()

# Sort by Energy Error to see which architecture learned the eigenvalue best
print(df.sort_values(by="Energy_Error")[["Architecture", "Energy_Error", "Final_Energy", "Time_Sec"]])