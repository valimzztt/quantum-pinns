import pandas as pd 
from utils.benchmarking import PinnBenchmark
from utils.training import train_model
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
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from H_atom.H_atom_3D_energytrained import HydrogenPINN

# 1. Setup Benchmark
benchmark = PinnBenchmark(device='cpu', n_test=100000)
results = []

# We will plot all results together for better comparison 
fig, ax = plt.subplots(figsize=(10, 6))
z_vals = np.linspace(-6, 6, 300)
inputs_z = np.zeros((300, 3))
inputs_z[:, 2] = z_vals #(x=0, y=0)
inputs_torch = torch.tensor(inputs_z, dtype=torch.float32)

r_vals = np.abs(z_vals)
psi_exact = (1.0 / np.sqrt(np.pi)) * np.exp(-r_vals)
prob_exact = psi_exact**2
ax.plot(z_vals, prob_exact, 'k--', linewidth=2.5, label='Analytical (Exact)')


# We define Architectures to Test
configs = [
    (16, 1, "Small (16, 1)"),
    (32, 2, "Medium (32, 2)"),
    (64, 2, "Deep (64, 2)"),
]

# We define also the learning rate
lr = 0.005 # learning rate
epochs = 2000 
N_f = 3000 # sampling points

for width, layers, name in configs:
    print(f"Benchmarking {name}...")
    model = HydrogenPINN(width=width, depth=layers).to('cpu')
    
    start = time.time()
    train_model(model,N_f, epochs, device='cpu', learning_rate = lr)
    duration = time.time() - start
    
    # Evaluate Metrics
    metrics = benchmark.evaluate(model, duration)
    metrics["Architecture"] = name
    results.append(metrics)
    
    model.eval()
    with torch.no_grad():
        # Predict on our prepared z-axis line
        psi_pred = model(inputs_torch).numpy().flatten()
        
        # Calculate probability density
        prob_density = psi_pred**2
        
        # Plot this model's curve
        ax.plot(z_vals, prob_density, linewidth=2, alpha=0.8, label=f"{name}")

# 3. Finalize Plot
ax.set_xlabel(r'$z \ [a_0]$', fontsize=14)
ax.set_ylabel(r'$|\psi|^2$', fontsize=14)
ax.set_title(f'Wavefunction Comparison (Epochs=2000)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_yscale('log') # Log scale helps see the decay in the tails
ax.set_ylim(1e-5, 1.0) # Set limits to avoid log(0) issues

plt.tight_layout()
plt.show()

# We will also print the result section for each architecture
df = pd.DataFrame(results)
print("\nFinal Benchmark Results:")
print(df.sort_values(by="L2_Error_Psi")[["Architecture", "L2_Error_Psi", "Abs_Error_E", "Time_Sec", "Final_Energy", "Params"]])

