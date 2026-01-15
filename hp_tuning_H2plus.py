import pandas as pd 
from utils.benchmarking.H2PlusPinnBenchmark import HPinnBenchmark1D as HPinnBenchmark
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys 
from utils.filemanager import load_config
from H2Plus_refactored.H2plus_3D import H2PlusPINN3D
from utils.filemanager import load_config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.filemanager import load_config
from  utils.memory_usage import get_memory_usage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for H2+ (3D)")
# We store the data in YAML files so that it will be easier to track the different parameters 
filepath = os.path.join(parent_dir, "configs", "3D_H2plus.yaml")
config = load_config(path=filepath)
# Retrieve all the info from YAML file
epochs = config['training']['epochs']
N_f = config['training']['n_collocation']
lr = config['training']['learning_rate']
w_pde = config['loss_weights']['pde']
w_norm = config['loss_weights']['normalization']
w_energy = config['loss_weights']['energy_constraint']
sampling_strategy = config['training']['train_distribution'] 
E_ref= config['physics']['E_ref']
E_init= config['physics']['E_init']
R_dist = config['physics']['R_dist']
num_dense_layers =config['training']['num_dense_layers']
num_dense_nodes = config['training']['num_dense_nodes']
initializer = config['training']['initializer']

L_max= config['physics']['L_max']
L_min= config['physics']['L_min']



# Define the PinnBechmark 
benchmark = HPinnBenchmark(device='cpu', n_test=100000)
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

