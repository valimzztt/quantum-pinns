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
sys.path.append(current_dir )
filepath = os.path.join(current_dir, "configs", "1D_H_atom.yaml")
config = load_config(filepath)
epochs = config['training']['epochs']
N_f = config['training']['n_collocation'] 

benchmark = HPinnBenchmark(device='cpu', n_test=100000, max_r=15.0)
results = []
histories = [] # Store energy history for plotting

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

r_vals = np.linspace(0.01, 10.0, 300) 
inputs_r = r_vals.reshape(-1, 1)
inputs_torch = torch.tensor(inputs_r, dtype=torch.float32)

# Exact u(r)
u_exact = 2.0 * r_vals * np.exp(-r_vals)
prob_exact = u_exact**2
ax[1].plot(r_vals, prob_exact, 'k--', linewidth=2.5, label='Analytical (Exact)')

fixed_width = 32
fixed_depth = 2

energy_configs = [
    (-0.1, "E_init = -0.1"),
    (-0.5, "E_init = -0.5 (Exact)"),
    (-0.8, "E_init = -0.8"),
    (-1.5, "E_init = -1.5")
]

for e_start, name in energy_configs:
    print(f"Benchmarking with {name}...")
    # Initialize 
    model = HydrogenPINN1D(width=fixed_width, depth=fixed_depth).to('cpu')
    model.E = torch.nn.Parameter(torch.tensor([e_start], dtype=torch.float32)) # overwrite init energy!

    start = time.time()
    train1D(model, N_f, epochs) 
    duration = time.time() - start
    histories.append((name, model.E_history))

    # Evaluate Metrics
    metrics = benchmark.evaluate(model, duration)
    metrics["Configuration"] = name
    metrics["Initial_E"] = e_start
    results.append(metrics)


    model.eval()
    with torch.no_grad():
        u_pred = model(inputs_torch).numpy().flatten()
        if np.mean(u_pred) < 0: u_pred = -u_pred
        prob_density = u_pred**2

        label_text = f"{name} -> E={model.E.item():.3f}"
        ax[1].plot(r_vals, prob_density, linewidth=2, alpha=0.8, label=label_text)

for name, history in histories:
    epochs_axis = np.arange(len(history)) 
  
    ax[0].plot(epochs_axis, history, linewidth=2, label=name)


ax[0].axhline(-0.5, color='k', linestyle='--', linewidth=2, label="Exact (-0.5)")
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Energy (Ha)', fontsize=14)
ax[0].set_title('Energy Convergence', fontsize=16)
ax[0].grid(True, alpha=0.3)
ax[0].legend(fontsize=10)

ax[1].set_xlabel(r'$r \ [a_0]$', fontsize=14)
ax[1].set_ylabel(r'$|u(r)|^2$ (log scale)', fontsize=14) 
ax[1].set_title(f'Wavefunction', fontsize=16)
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=10)
ax[1].set_yscale('log')
ax[1].set_ylim(1e-8, 5.0) 

plt.tight_layout()
plt.show()

# Print Table
df = pd.DataFrame(results)
print("\nFinal Benchmark Results:")
df["Energy_Error"] = (df["Final_Energy"] - (-0.5)).abs()
print(df.sort_values(by="Energy_Error")[["Configuration", "Final_Energy", "Energy_Error", "L2_Error_R", "Time_Sec"]])