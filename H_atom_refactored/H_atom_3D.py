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
L_min = -5.0
L_max = 5.0 
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
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
            # Initialize bias to zerro (standard for Gorot initializer)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        # Calculate distance r
        r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
        # Ansatz: e^(-r). This helps the network drastically, but the NN
        # still needs to learn the normalization scaling factor.
        ansatz = torch.exp(-1.0 * r)
        return ansatz * (1.0 + self.net(x))

def physics_loss(model, x):
    x.requires_grad = True
    psi = model(x)
    # Calcultae the Kinetic Energy
    grads = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    laplacian = 0
    for i in range(3):
        grad_2 = torch.autograd.grad(grads[:, i], x, torch.ones_like(grads[:, i]), create_graph=True)[0]
        laplacian += grad_2[:, i].view(-1, 1)
    kinetic = -0.5 * laplacian
    # Potential Energy (-1/r), x is a tensor that contains (x,y,z) coordinates
    r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
    potential = -1.0 / r
    
    # Residual:(H - E)Psi = 0
    residual = (kinetic + potential * psi) - (model.E * psi)
    loss_pde = torch.mean(residual**2)
    # Normalization Loss (is this valid?: Monte Carlo average)
    domain_volume = (L_max-L_min)**3  # dimension of the box
    integral_approx = torch.mean(psi**2) * domain_volume
    loss_norm = (integral_approx - 1.0)**2
    # Total Loss
    loss = loss_pde + w_norm * loss_norm
    return  loss


def train3D(N_f, epochs):
    print(f" Training with N_f = {N_f} points")
    print(f"Sampling Strategy: {sampling_strategy}")
    model = HydrogenPINN3D(width=num_dense_nodes, depth=num_dense_layers).to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        if sampling_strategy == "Hammersley":
            # 1. Generate Points using skopt (returns list of lists)
            # skopt returns standard Python lists, so we convert to Tensor
            sampler = Hammersly(min_skip=1, max_skip=1000)
            # tHE domain of the sampler (3 dimensions, 0 to 1)
            space = [(0.0, 1.0)] * 3 
            raw_points = sampler.generate(space, N_f)
            h_points = torch.tensor(raw_points, dtype=torch.float32)
            # inputs = (h_points * (Max - Min)) + Min to map to box
            inputs = (h_points * (L_max - L_min) + L_min).requires_grad_(True)
        elif sampling_strategy == "Gaussian": # Recommended for 3D Atoms
            # Strategy: 50% Gaussian (to learn the core), 50% Uniform (to learn the boundaries)
            points_gauss = torch.randn(N_f // 2, 3) * 1.5  # sigma=1.5 roughly matches the Hydrogen 1s size
            points_uniform = (torch.rand(N_f // 2, 3) * 10.0 - 5.0)
            inputs = torch.cat([points_gauss, points_uniform], dim=0).float().requires_grad_(True)

        else:
            # The default is a standard Pseudo-Random Uniform Sampling (not recommended)
            inputs = (torch.rand(N_f, 3) * 10.0 - 5.0).float().requires_grad_(True)
            
        loss = physics_loss(model, inputs)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.5f}, Energy={model.E.item():.4f}")

    return model

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    initial_memory = get_memory_usage()
    start_time = time.time()
    N_f_label =  "N_f= " + str(N_f)
    model_trained = train3D(N_f=N_f, epochs=epochs)
    E_hist = np.array(model_trained.E_history)
    epochs_hist = np.arange(len(E_hist)) * 10
    
    # Create z-axis points for evaluation
    z_vals = np.linspace(L_min, L_max, 200)
    # Create input tensor: x=0, y=0, z varies
    inputs = np.zeros((200, 3))
    inputs[:, 2] = z_vals # Set z column
    inputs_torch = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        psi_pred = model_trained(inputs_torch).numpy().flatten()
    prob_density_sim = psi_pred**2     # probability density
    memory_used = get_memory_usage() - initial_memory
    print(f"Total memory used: {memory_used:.2f} MB")
    # First plot is the energy history
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs_hist, E_hist, label="Learned E")
    ax.axhline(y=-0.5, color='r', linestyle='--', label='Exact (-0.5)')
    ax.set_xlabel(r'$r \ [a_0]$', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Abs. error on $R^2 \ [a_0^{-3}]$', fontsize=16, fontweight='bold')
    ax.set_title('Absolute Error of Psi^2', fontsize=14)
    ax.set_xlim(0, 10) 
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Calculate Analytical Solution: Psi = (1/sqrt(pi)) * e^(-r)
    r_vals = np.abs(z_vals)     # r = |z| along the z-axis
    psi_exact = (1.0 / np.sqrt(np.pi)) * np.exp(-r_vals)
    prob_density_exact = psi_exact**2
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
    
    #  Create a 2D grid in the x-y plane at z=0
    L_plot = 10.0 
    N_grid = 200 
    x = np.linspace(-L_plot, L_plot, N_grid)
    y = np.linspace(-L_plot, L_plot, N_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) # z is 0 everywhere on this slice
    pts_grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)  # Flatten grid and convert to tensor for the network
    pts_grid_t = torch.tensor(pts_grid, dtype=torch.float32)
    # Predict Psi on the grid 
    model_trained.eval() # Set to evaluation mode
    with torch.no_grad():
        # Get psi predictions
        psi_pred_flat = model_trained(pts_grid_t).numpy().flatten()
        prob_density_flat = psi_pred_flat**2
        prob_density_2d = prob_density_flat.reshape(N_grid, N_grid) # Reshape back to 2D grid 

    log_prob = np.log10(prob_density_2d + 1e-10) # Add a small epsilon to avoid log(0

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

