import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys 
import time 
from skopt.sampler import Hammersly 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from  utils.memory_usage import get_memory_usage
from utils.filemanager import load_config

# We store the data in YAML files so that it will be easier to track the different parameters 
filepath = os.path.join(parent_dir, "configs", "1D_H_atom.yaml")
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


# Normalization: we create a separate grid just for normalization.
# A fixed uniform grid (Riemann Sum)
R_max = 20.0 
eps = 1e-5   # Small shift to avoid div by zero
N_norm_grid = 2000
r_grid_norm = torch.linspace(eps, R_max, N_norm_grid, device=device).view(-1, 1)
# we pre-calculate the step size (dr) for the integral that we need for normalization
dr = (R_max - eps) / N_norm_grid

class HydrogenPINN1D(nn.Module):
    def __init__(self, width=num_dense_nodes, depth=num_dense_layers):
        super().__init__()
        layers = []
        # important: this is 1D 
        layers.append(nn.Linear(1, width))
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
        return  x * torch.exp(-x)*self.net(x)

def pde_residual(r, u, d2u, current_E):
    # Radial Schrodinger Equation: u'' + 2(E + 1/r)u = 0
    # Note: We pass current_E explicitly
    res = d2u + 2.0 * (current_E + 1.0 / r) * u
    return res

# The loss function is defined
def physics_loss(model, r):
    r.requires_grad_(True)
    u = model(r)
    # Compute first derivative
    du = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    # Compute second derivative
    d2u = torch.autograd.grad(du, r, torch.ones_like(du), create_graph=True)[0]
    res = pde_residual(r, u, d2u, model.E)
    loss_pde = (res**2).mean()
    # Normalization loss (make sure to use the grid one)
    u_norm_pred = model(r_grid_norm) 
    # Riemann Sum: Sum(height^2) * width
    # Note: We use torch.sum, not mean, because we are doing a Riemann integral
    integral_approx = torch.sum(u_norm_pred**2) * dr 
    loss_norm = (integral_approx - 1.0)**2
    # Total Loss
    loss = loss_pde + w_norm * loss_norm
    return  loss


def train1D(model, N_f, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Starting training...")
    print(f"The initialization energy is {model.E}")
    for epoch in range(epochs):
        optimizer.zero_grad()
        if sampling_strategy == "Hammersley":
            sampler = Hammersly(min_skip=1, max_skip=1000)
            space = [(eps, R_max)]
            raw_points = sampler.generate(space, N_f)
            r_c = torch.tensor(raw_points, dtype=torch.float32, device=device)
            
        elif sampling_strategy == "Gaussian":
            # 1. Generate Uniform Hammersley points in [0, 1]
            sampler = Hammersly(min_skip=1, max_skip=1000)
            space = [(0.0, 1.0)] # Normalized space
            raw_points = sampler.generate(space, N_f) 
            u_tensor = torch.tensor(raw_points, dtype=torch.float32, device=device)
            # 2. Map [0, 1] -> Half-Normal Distribution
            # This bunches the points near r=0 while keeping the "smoothness" of Hammersley
            # Formula: r = sigma * sqrt(2) * erfinv(u)
            sigma = 2.5 # Controls the spread
            u_tensor = torch.clamp(u_tensor, min=0.0, max=0.999999)
            r_c = sigma * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(u_tensor)
            r_c = torch.clamp(r_c, min=eps, max=R_max) # don´t go above R_max
            
        else: # Smart Sampling
            split_ratio = 0.3
            split_point = max(R_max * split_ratio, 5.0)
            split_point = min(split_point, R_max - 0.1)
            
            r_near = (split_point - eps) * torch.rand(N_f // 2, 1, device=device) + eps
            r_far = (R_max - split_point) * torch.rand(N_f // 2, 1, device=device) + split_point
            r_c = torch.cat([r_near, r_far], dim=0)
            
        r_c = r_c.to(device) # r_c must be on device (remember!)
        loss = physics_loss(model, r_c)
        loss.backward()
        optimizer.step()
        model.E_history.append(model.E.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Energy: {model.E.item():.5f} Ha")         
    return model

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    initial_memory = get_memory_usage()
    start_time = time.time()
    N_f_label =  "N_f= " + str(N_f)
    model = HydrogenPINN1D(width=num_dense_nodes, depth=num_dense_layers).to(device)
    model_trained = train1D(model, N_f=N_f, epochs=epochs)
    E_hist = np.array(model_trained.E_history)
    epochs_hist = np.arange(len(E_hist))
    r_plot = np.linspace(eps, 15.0, 500) # Define Radial Grid (avoid r=0 to prevent division by zero)
    r_plot_t = torch.tensor(r_plot, dtype=torch.float32, device=device).view(-1, 1)

    # model.eval() # do we need this?
    with torch.no_grad():
        u_pred = model_trained(r_plot_t).numpy().flatten()
    # Psi(r) = u(r) / r (what is u(r) here is R(r) in Paolos thesis)
    Psi_pred = u_pred / r_plot
    Psi_exact = 2.0 * np.exp(-r_plot)
    # Exact solution: u(r) = 2 * r * exp(-r)
    u_exact = 2.0 * r_plot * np.exp(-r_plot)
    u_exact_squared = u_exact**2
    u_pred_squared = u_pred**2 
    memory_used = get_memory_usage() - initial_memory
    print(f"Total memory used: {memory_used:.2f} MB")

    # Compute the Squared Error Term: Error = | |Psi_pred|^2 - |Psi_exact|^2 |
    # Psi here corresponds to R(r) in Paolos thesis
    abs_error_sq = np.abs(Psi_pred**2 - Psi_exact**2)
    mae_scalar = np.mean(abs_error_sq)
    print(f"Mean Absolute Error (MAE) on Wavefunction Squared: {mae_scalar:.6f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(epochs_hist, E_hist, label="Learned E")
    ax[0].axhline(y=-0.5, color='r', linestyle='--', label='Exact (-0.5)')
    ax[0].set_title("Energy Convergence")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Energy (Ha)")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(r_plot, u_pred**2, 'b-', linewidth=2, label=r"PINN $R(r)^2$")
    ax[1].plot(r_plot, u_exact**2, 'r--', linewidth=2, label=r"Exact $R(r)^2$")
    ax[1].set_title(r"Radial Wavefunction $R^2$ (log scale)=(l=0)")
    ax[1].set_xlabel("r (Bohr)")
    ax[1].legend()
    ax[1].set_yscale('log')
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the MAE error as a function of the radial function 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_plot, abs_error_sq, color='#4472C4', linewidth=2.5, label=r'$|R_{pred}|^2 - |R_{exact}|^2|$')
    ax.set_xlabel(r'$r \ [a_0]$', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Abs. error on $\Psi^2 \ [a_0^{-1}]$', fontsize=16, fontweight='bold')
    ax.set_title(r'Absolute Error of  $\Psi^2$', fontsize=14)
    ax.set_xlim(0, 10) 
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot the absolute error in the Radial function squared.
    abs_error_sq_u = np.abs(u_pred**2 - u_exact**2)
    mae_scalar_u = np.mean(abs_error_sq_u)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_plot, abs_error_sq_u, color='#4472C4', linewidth=2.5, label=r'$|R_{pred}|^2 - |R_{exact}|^2|$')
    ax.set_xlabel(r'$r \ [a_0]$', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Abs. error on $R^2 \ [a_0^{-1}]$', fontsize=16, fontweight='bold')
    ax.set_title(r'Absolute Error of $R^2$ (Radial function squared)', fontsize=14)
    ax.set_xlim(0, 10) 
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
