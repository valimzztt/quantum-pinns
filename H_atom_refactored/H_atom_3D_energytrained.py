import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import sys 
from skopt.sampler import Hammersly
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from  utils.memory_usage import get_memory_usage
from utils.filemanager import load_config

# We store the data in YAML files so that it will be easier to track the different parameters 
filepath = os.path.join(parent_dir, "configs", "3D_H_atom.yaml")
config = load_config(path=filepath)
# Retrieve all the info from YAML file
epochs = config['training']['epochs']
N_f = config['training']['n_collocation']
lr = config['training']['learning_rate']
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
# We aim to solve the H atom in 3D dimensions
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for H atom (3D)")

class HydrogenPINN3D(nn.Module):
    def __init__(self, width=num_dense_nodes, depth=num_dense_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, width))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
            
        # Output layer (width -> 1)
        layers.append(nn.Linear(width, 1))
        
        self.net = nn.Sequential(*layers)
        
        # if Glorot, we apply the initialization function to every layer in self.net
        if initializer == "Glorot_uniform":
            print("Using Glorot Initialization")
            self.net.apply(self.init_weights)
        
        # Trainable energy parameter
        self.E = nn.Parameter(torch.tensor([E_init])) 

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
        # Calculate distance r
        r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
        # Ansatz: e^(-r). This helps the network drastically, but the NN
        # still needs to learn the normalization scaling factor.
        ansatz = torch.exp(-1.0 * r)
        return ansatz * (1.0 + self.net(x))

box_volume = (L_max - L_min)**3 
N_norm = 50000 

# The loss function is defined
def physics_loss(model, x):
    x.requires_grad = True
    psi = model(x)
    # Kinetic Energy (Laplacian)
    grads = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    laplacian = 0
    for i in range(3):
        grad_2 = torch.autograd.grad(grads[:, i], x, torch.ones_like(grads[:, i]), create_graph=True)[0]
        laplacian += grad_2[:, i].view(-1, 1)
    kinetic = -0.5 * laplacian
    
    # Potential Energy (-1/r)
    r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
    potential = -1.0 / r
    
    # Residual: (H - E)Psi = 0
    residual = (kinetic + potential * psi) - (model.E * psi)
    loss_pde = torch.mean(residual**2)
    # Point anchoring 
    if normalization == "Point_Anchoring":
        center_point = torch.zeros(1, 3)
        psi_center = model(center_point)
        # value of origin for 1s orbital isapprox 0.564
        loss_norm  = (psi_center - 0.564)**2 
    else: 
        # Should this be inside or outside of the loop
        x_norm = (torch.rand(N_norm, 3, device=device) * (L_max - L_min)) + L_min
        # Calculate Psi on uniform points regardless of what sampling strategy you used
        psi_norm = model(x_norm)
        # Monte Carlo Integral, which Volume*mn(Psi^2)
        integral_approx = box_volume*torch.mean(psi_norm**2)
        loss_norm = (integral_approx - 1.0)**2
    total_loss = loss_pde + w_norm*loss_norm
    return total_loss, loss_pde, loss_norm


def train(N_f, epochs):
    print(f"\nTraining with N_f = {N_f} points")
    print(f"Sampling Strategy: {sampling_strategy} ---")
    loss_history = {"total": [], "pde": [], "norm": [], "energy": []}
    model = HydrogenPINN3D(width=num_dense_nodes, depth=num_dense_layers).to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        if sampling_strategy == "Hammersley":
            sampler = Hammersly(min_skip=1, max_skip=1000)
            # tHE domain of the sampler (3 dimensions, 0 to 1)
            space = [(0.0, 1.0)] * 3 
            raw_points = sampler.generate(space, N_f)
            h_points = torch.tensor(raw_points, dtype=torch.float32)
            inputs = (h_points * 10.0 - 5.0).requires_grad_(True)
        else:
            # The default is a standard Pseudo-Random Uniform Sampling
            inputs = (torch.rand(N_f, 3) * 10.0 - 5.0).float().requires_grad_(True)
            
        loss, pde_val, norm_val = physics_loss(model, inputs)
        loss.backward()
        optimizer.step()
        # Let´s store all losses so that we can plot them
        loss_history["total"].append(loss.item())
        loss_history["pde"].append(pde_val.item())
        loss_history["norm"].append(norm_val.item())
        loss_history["energy"].append(model.E.item())

        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.5f}, Energy={model.E.item():.4f}")

    return model, loss_history

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    initial_memory = get_memory_usage()
    start_time = time.time()
    N_f_label =  "N_f= " + str(N_f)
    model_trained, history = train(N_f=N_f, epochs=epochs)
    memory_used  = get_memory_usage() - initial_memory 
    print(f"Memory used: {memory_used:.2f} MB")
    z_vals = np.linspace(-5, 5, 200)
    #input tensor: x=0, y=0, z varies
    inputs = np.zeros((200, 3))
    inputs[:, 2] = z_vals # Set z column
    inputs_torch = torch.tensor(inputs, dtype=torch.float32)

    with torch.no_grad():
        psi_pred = model_trained(inputs_torch).numpy().flatten()
    prob_density_sim = psi_pred**2     # probability density

    # Analytical Solution: Psi = (1/sqrt(pi)) * e^(-r)
    r_vals = np.abs(z_vals)     # r = |z| along the z-axis
    psi_exact = (1.0 / np.sqrt(np.pi)) * np.exp(-r_vals)
    prob_density_exact = psi_exact**2

    plt.figure(figsize=(10, 6))
    plt.plot(history["total"], label='Total Loss', color='black', linewidth=2)
    plt.plot(history["pde"], label='PDE Loss', linestyle='--', color='blue', alpha=0.7)
    plt.plot(history["norm"], label='Norm Loss', linestyle='--', color='orange', alpha=0.7)
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log') # Use Log scale to see differences clearly
    plt.title('Training Loss Components', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6, 5))
    plt.semilogy(z_vals, prob_density_sim, label='Simulation', linewidth=3, alpha=0.8)
    plt.semilogy(z_vals, prob_density_exact, '--', label='Analytical', linewidth=3)
    plt.xlabel(r'$z [a_0]$', fontsize=14)
    plt.ylabel(r'$\Psi^2 [a_0^{-3}]$', fontsize=14)
    plt.title(f'Hydrogen Density ({N_f_label} points)', fontsize=14)
    plt.ylim(1e-5, 0.5)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    print(f"Final Memory: {get_memory_usage():.2f} MB")
    #  Create a 2D grid in the x-y plane at z=0
    L_plot = 10.0 
    N_grid = 200 
    x = np.linspace(-L_plot, L_plot, N_grid)
    y = np.linspace(-L_plot, L_plot, N_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) # z is 0 everywhere on this slice
    pts_grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)  # Flatten grid and convert to tensor for the network
    pts_grid_t = torch.tensor(pts_grid, dtype=torch.float32)
    model_trained.eval() # Set to evaluation mode
    with torch.no_grad():
        # Get psi predictions
        psi_pred_flat = model_trained(pts_grid_t).numpy().flatten()
        prob_density_flat = psi_pred_flat**2
        prob_density_2d = prob_density_flat.reshape(N_grid, N_grid) # Reshape back to 2D grid 

    log_prob = np.log10(prob_density_2d + 1e-10) # Add a small epsilon to avoid log(0
    # Plots 
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(log_prob, extent=[-L_plot, L_plot, -L_plot, L_plot], 
                    origin='lower', cmap='cividis', vmin=-5, vmax=0)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('log10(Psi^2)', fontsize=14, fontweight='bold')
    ax.set_xlabel('x [a₀]', fontsize=18, fontweight='bold')
    ax.set_ylabel('y [a₀]', fontsize=18, fontweight='bold')
    ax.set_title('Hydrogen 1s Probability Density Slice (z=0)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.show()

