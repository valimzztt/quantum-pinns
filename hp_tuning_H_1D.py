import pandas as pd 
from utils.benchmarking.HPinnBenchmark_1D import HPinnBenchmark1D as HPinnBenchmark
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys 
from utils.filemanager import load_config
from H_atom_refactored.H_atom_1D_energytrained import HydrogenPINN1D, train1D

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Load Config
filepath = os.path.join(current_dir, "configs", "1D_H_atom.yaml")
config = load_config(filepath)

# Retrieve params
epochs = config['training']['epochs']
N_f = config['training']['n_collocation'] 
lr = config['training']['learning_rate']

benchmark = HPinnBenchmark(device='cpu', n_test=100000, max_r=15.0)
results = []
energy_histories = [] # List to store history for plotting later
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
r_vals = np.linspace(0.01, 10.0, 300) 
inputs_r = r_vals.reshape(-1, 1)
inputs_torch = torch.tensor(inputs_r, dtype=torch.float32)

# Exact u(r) = 2*r*exp(-r) -> R(r) = 2*exp(-r)
u_exact = 2.0 * r_vals * np.exp(-r_vals)
prob_exact = u_exact**2 # This is |u(r)|^2
ax[0].plot(r_vals, prob_exact, 'k--', linewidth=2.5, label='Analytical (Exact)')

# Define Architectures
configs = [
    (16, 1, "Small (16, 1)"),
    (32, 2, "Medium (32, 2)"),
    (64, 2, "Medium (Wide) (64, 2)"),
    (32, 3, "Deep (32, 3)"),
    (64, 3, "Deep (Wide) (64, 3)"),
]

for width, layers, name in configs:
    print(f"Benchmarking {name}...")
    model = HydrogenPINN1D(width=width, depth=layers).to('cpu')
    # This overwrites whatever was in the YAML config 
    # model.E = torch.nn.Parameter(torch.tensor([-0.8], dtype=torch.float32))
    start = time.time()
    train1D(model, N_f, epochs) 
    duration = time.time() - start

    energy_histories.append((name, model.E_history))
  
    metrics = benchmark.evaluate(model, duration)
    metrics["Architecture"] = name
    results.append(metrics)
    model.eval()
    with torch.no_grad():
        u_pred = model(inputs_torch).numpy().flatten()

        if np.mean(u_pred) < 0:
            u_pred = -u_pred
            
        prob_density = u_pred**2 
        ax[0].plot(r_vals, prob_density, linewidth=2, alpha=0.8, label=f"{name}")

# --- Plot 2: Energy History ---
for name, history in energy_histories:
    epochs_axis = np.linspace(0, epochs, len(history))
    
    ax[1].plot(epochs_axis, history, linewidth=2, alpha=0.8, label=name)

ax[1].axhline(-0.5, color='k', linestyle='--', linewidth=2, label="Exact (-0.5)")

ax[0].set_xlabel(r'$r \ [a_0]$', fontsize=14)
ax[0].set_ylabel(r'$|u(r)|^2$', fontsize=14) 
ax[0].set_title(f'Radial Probability Density (Log Scale)', fontsize=14)
ax[0].grid(True, alpha=0.3)
ax[0].legend(fontsize=10)
ax[0].set_yscale('log')
ax[0].set_ylim(1e-8, 5.0) 
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Energy (Ha)', fontsize=14) 
ax[1].set_title(f'Energy Convergence (Start E = -0.8)', fontsize=14)
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=10)
ax[1].set_ylim(-0.9, -0.4) 

plt.tight_layout()
plt.show()


df = pd.DataFrame(results)
print("\nFinal Benchmark Results (1D):")
df["Energy_Error"] = (df["Final_Energy"] - (-0.5)).abs()
print(df.sort_values(by="L2_Error_R")[["Architecture", "L2_Error_R", "Energy_Error", "Final_Energy", "Time_Sec"]])
