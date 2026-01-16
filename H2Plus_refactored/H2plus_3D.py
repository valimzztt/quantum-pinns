import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os 
import sys

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

class H2PlusPINN3D(nn.Module):
    def __init__(self, width=num_dense_nodes, depth=num_dense_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, width))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
            
        # the NN outputs one value 
        layers.append(nn.Linear(width, 1))
        
        self.net = nn.Sequential(*layers)
        
        # if Glorot is on, we apply the initialization function to every layer in self.net
        if initializer == "Glorot_uniform":
            print("Using Glorot Initialization")
            self.net.apply(self.init_weights)
        
        # We add the Trainable energy parameter
        self.E = nn.Parameter(torch.tensor([E_init])) 
        self.E_history = [] 
    def init_weights(self, m):
        # This function checks if a layer is a Linear layer
        if isinstance(m, nn.Linear):
            # Apply Xavier (Glorot) Uniform initialization to the weights
            # Gain=1.0 is standard for Tanh, though some papers suggest 5/3 (approx 1.67)
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
            
            # Initialize biases to zero (standard practice)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)
    

# Setup Geometry (Nuclei at +/- 1.0 on Z-axis)
Z_pos = 1.0 

def compute_distances_3d(x, y, z):
    # Nuclei are at (0, 0, Z_pos) and (0, 0, -Z_pos)
    # x and y are centered at 0
    d1 = torch.sqrt(x**2 + y**2 + (z - Z_pos)**2 + 1e-6)
    d2 = torch.sqrt(x**2 + y**2 + (z + Z_pos)**2 + 1e-6)
    return d1, d2

def psi_trial(model, x, y, z):
    d1, d2 = compute_distances_3d(x, y, z)
    # LCAO Ansatz 
    lcao = torch.exp(-d1) + torch.exp(-d2)

    # Concatenate inputs for NN: shape (N, 3)
    inputs = torch.cat([x, y, z], dim=1)
    nn_out = model(inputs)
    
    return lcao * nn_out

def laplacian_3d(psi, x, y, z):
    grads = torch.autograd.grad(psi, [x, y, z], torch.ones_like(psi), create_graph=True)
    dpsi_dx, dpsi_dy, dpsi_dz = grads[0], grads[1], grads[2]
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, torch.ones_like(dpsi_dx), create_graph=True)[0]
    d2psi_dy2 = torch.autograd.grad(dpsi_dy, y, torch.ones_like(dpsi_dy), create_graph=True)[0]
    d2psi_dz2 = torch.autograd.grad(dpsi_dz, z, torch.ones_like(dpsi_dz), create_graph=True)[0]
    
    return d2psi_dx2 + d2psi_dy2 + d2psi_dz2

# The loss function is defined
def physics_loss(model, x, y, z, epoch):
    current_w_energy = w_energy
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    psi = psi_trial(model, x, y, z)
    lap_psi = laplacian_3d(psi, x, y, z)
    d1, d2 = compute_distances_3d(x, y, z)
    V = -1.0/d1 -1.0/d2
    
    # Residual
    res= -0.5 * lap_psi + (V - model.E) * psi
    loss_pde = (res**2).mean()

    # Norm Penalty
    volume = (L_max -L_min)**3  
    n_norm = 10000 # increase this value if not variance is too large
    x_u =(torch.rand(n_norm, 1, device=x.device)*(L_max - L_min))+L_min
    y_u =(torch.rand(n_norm, 1, device=x.device)*(L_max - L_min))+L_min
    z_u =(torch.rand(n_norm, 1, device=x.device)*(L_max - L_min))+L_min

    psi_u = psi_trial(model, x_u, y_u, z_u)
    # Monte Carlo Integral: V*Mean(psi^2)
    integral=volume*torch.mean(psi_u**2)
    loss_norm = (integral - 1.0)**2
    # Energy Constraint
    loss_energy = model.E
    #loss_energy = torch.relu(model.E - E_ref)  
    if epoch % 100 == 0: print('loss_pde', loss_pde, 'loss_norm', loss_norm, 'loss_energy', loss_energy)

    total_loss = loss_pde + w_norm*loss_norm + w_energy*loss_energy
    return total_loss, loss_pde, loss_norm, loss_energy


def train_3D(model, N_f, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\n Training 3D H2+ (N={N_f}) ---")
    loss_history = {"total": [], "pde": [], "norm": [], "loss_energy": [], "energy": []}
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        # CHANGE 3: Sampling in 3D Box
        # We sample x, y, z from Normal distributions
        # Focus sampling near origin but allow spread
        x = torch.randn(N_f, 1, device=device) * 2.0
        y = torch.randn(N_f, 1, device=device) * 2.0
        z = torch.randn(N_f, 1, device=device) * 3.0 # Elongated along Z axis
        
        loss, loss_pde, loss_norm,  loss_energy = physics_loss(model, x, y, z, epoch) # loss is the total loss
        loss.backward()
        optimizer.step()
        # Let´s store all losses so that we can plot them
        loss_history["total"].append(loss.item())
        loss_history["pde"].append(loss_pde.item())
        loss_history["norm"].append(loss_norm.item())
        loss_history["loss_energy"].append(loss_energy.item())
        loss_history["energy"].append(model.E.item())
        model.E_history.append(model.E.item())
        
        if epoch % 500 == 0:
            print(f"Ep {epoch} | Loss: {loss.item():.4f} | E: {model.E.item():.4f}")

    return model, loss_history

def get_3d_normalization_factor(model, func_type="sim", n_samples=100000):
    """
    Calculates the normalization constant A such that Integral(|A*psi|^2) dV = 1
    Using Monte Carlo Integration over a large box.
    """
    side_length =L_max -L_min
    Volume = side_length**3 
    # torch.rand gives 0, 1], we scale then shift
    x = (torch.rand(n_samples, 1, device=device) * side_length) + L_min
    y = (torch.rand(n_samples, 1, device=device) * side_length) + L_min
    z = (torch.rand(n_samples, 1, device=device) * side_length) + L_min
    
    with torch.no_grad():
        if func_type == "sim":
            psi_vals = psi_trial(model, x, y, z)
        else:
            # Run LCAO Reference
            d1, d2 = compute_distances_3d(x, y, z)
            psi_vals = torch.exp(-d1) + torch.exp(-d2)
            
    # Integral = Volume*Mean(psi^2)
    integral_psi2 = Volume * torch.mean(psi_vals**2).item()
    return 1.0 / np.sqrt(integral_psi2)

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    initial_memory = get_memory_usage()
    start_time = time.time()
    N_f_label = "N_f= " + str(N_f)
    # Initialize the correct class
    model = H2PlusPINN3D(width=num_dense_nodes, depth=num_dense_layers).to(device)
    # Train
    model_trained, history = train_3D(model, N_f=N_f, epochs=epochs)
    memory_used  = get_memory_usage() - initial_memory 
    print(f"Memory used: {memory_used:.2f} MB")
    memory_used  = get_memory_usage() - initial_memory 
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Time: {time.time() - start_time:.2f}s")
    axis_vals = np.linspace(L_min, L_max, 400)
    z_t = torch.tensor(axis_vals, dtype=torch.float32, device=device).view(-1, 1)
    x_t = torch.zeros_like(z_t)
    y_t = torch.zeros_like(z_t)
    with torch.no_grad():
        psi_pred_raw = psi_trial(model_trained, x_t, y_t, z_t).cpu().numpy().flatten()
    prob_density_raw = psi_pred_raw**2     # probability density

    # First plot the history of the losses
    plt.figure(figsize=(10, 6))
    plt.plot(history["total"], label='Total Loss', color='black', linewidth=2)
    plt.plot(history["pde"], label='PDE Loss', linestyle='--', color='blue', alpha=0.7)
    #plt.plot(-history["loss_energy"], label='Energy loss', linestyle='-.', color='cyan')
    plt.plot(history["norm"], label='Norm Loss', linestyle='--', color='orange', alpha=0.7)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log') # Use Log scale to see differences clearly
    plt.title('Training Loss Components', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(axis_vals, psi_pred_raw**2, color='#4472C4', linewidth=3, 
             label='3D Simulation')
    plt.xlabel(r'$\boldsymbol{x \ [a_0]}$', fontsize=16)
    plt.ylabel(r'$\boldsymbol{\Psi^2 \ [a_0^{-3}]}$', fontsize=16)
    plt.title(f'3D Result for R={Z_pos:.0f}', fontsize=14)
    
    plt.xlim(-10, 10)
    plt.ylim(0, 0.25) 
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    """ norm_sim = get_3d_normalization_factor(model_trained, "sim")
    norm_ref = get_3d_normalization_factor(model_trained, "ref")
    
    # ... (Plotting code remains the same, just ensure psi_trial calls use model_trained) ...
    axis_vals = np.linspace(-10, 10, 400)
    z_t = torch.tensor(axis_vals, dtype=torch.float32, device=device).view(-1, 1)
    x_t = torch.zeros_like(z_t)
    y_t = torch.zeros_like(z_t)
    
    with torch.no_grad():
        psi_pred_raw = psi_trial(model_trained, x_t, y_t, z_t).cpu().numpy().flatten()
    
    # Apply normalization
    psi_sim_norm = psi_pred_raw * norm_sim
     # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(axis_vals, psi_pred_raw**2, color='#4472C4', linewidth=3, 
             label='3D Simulation')
    #plt.plot(axis_vals, psi_ref_raw**2, color='#ED7D31', linestyle=':', linewidth=4, 
    #         label='Reference (LCAO)')
    
    plt.xlabel(r'$\boldsymbol{x \ [a_0]}$', fontsize=16)
    plt.ylabel(r'$\boldsymbol{\Psi^2 \ [a_0^{-3}]}$', fontsize=16)
    plt.title(f'3D Result for R={Z_pos:.0f}', fontsize=14)
    
    plt.xlim(-10, 10)
    plt.ylim(0, 0.25) 
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() """
