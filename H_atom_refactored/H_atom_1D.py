import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
import torch
import numpy as np
from skopt.sampler import Hammersly 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
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


""" class HydrogenPINN1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Using Tanh is good, but Sigmoid or Swish can sometimes be smoother for exp decays
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.E_history = [] 

    def forward(self, x):
        return self.net(x) """

class HydrogenPINN1D(nn.Module):
    def __init__(self, width=num_dense_nodes, depth=num_dense_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, width))
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
    
model = HydrogenPINN1D().to(device)
E = torch.tensor([E_init], dtype=torch.float32, device=device, requires_grad=True)

# We update the Optimizer for the Neural Network weights only
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
R_max = 20.0 # 
eps = 1e-5   # avoid div by zero

def u_ansatz(r):
    """
    We define the Ansatz: R(r) = r*e^(-r)*NN(r)
    1. 'r' handles the cusp at the nucleus (u(0)=0).
    2. 'e^(-r)' handles the decay at infinity.
    3. The NN learns the deviation(which is just a constant for 1s).
    """
    nn_out = model(r)
    return r * torch.exp(-r) * nn_out

def get_derivatives(r):
    r.requires_grad_(True)
    u = u_ansatz(r) # u(r) here is R(r) in Paolo's Thesis
    du = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, r, torch.ones_like(du), create_graph=True)[0]
    return u, du, d2u

def pde_residual(r, u, d2u, current_E):
    # Only difference with the other 1D case is that here the energy is fixed
    # Radial Schrodinger Equation: u'' + 2(E + 1/r)u = 0
    res = d2u + 2.0 * (current_E + 1.0 / r) * u
    return res

# Training Loop
loss_history = []
# Normalization: we create a separate grid just for normalization.
# A fixed uniform grid (Riemann Sum)
N_norm_grid = 2000
r_grid_norm = torch.linspace(eps, R_max, N_norm_grid, device=device).view(-1, 1)
# we pre-calculate the step size (dr) for the integral that we need for normalization
dr = (R_max - eps) / N_norm_grid

for epoch in range(epochs): 
    optimizer.zero_grad()
    split_ratio = 0.3 # can finetune it to 0.25 or other values 
    if sampling_strategy == "Hammersley":
        #print("We are using Hammersley for r sampling")
        # We initialize the Hammersly sampler,
        sampler = Hammersly(min_skip=1, max_skip=1000)
        # Define the space: 1 Dimension going from eps to R_max
        space = [(eps, R_max)]
        raw_points = sampler.generate(space, N_f) # convert from list to tensor
        r_c = torch.tensor(raw_points, dtype=torch.float32, device=device)
    elif sampling_strategy == "Gaussian":
        #  we generate Uniform Hammersley points between 0 and 1
        sampler = Hammersly(min_skip=1, max_skip=1000)
        space = [(0.0, 1.0)] 
        raw_points = sampler.generate(space, N_f) 
        u_tensor = torch.tensor(raw_points, dtype=torch.float32, device=device)
        # This basically bunches the poin near r=0 while keeping the "smoothness" of Hammersley
        # Formula: r = sigma * sqrt(2) * erfinv(u)
        sigma = 2.5 # spread
        u_tensor = torch.clamp(u_tensor, min=0.0, max=0.999999)
        r_c = sigma * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(u_tensor)
        r_c = torch.clamp(r_c, min=eps, max=R_max)
    else:
        # let´s call this smart sampling
        # We make the split dependent on R_max
        split_point = R_max * split_ratio
        split_point = max(split_point, 5.0)
        split_point = min(split_point, R_max - 0.1) # just for safety
        # 50% of points in the "Near" region
        r_near = (split_point - eps) * torch.rand(N_f // 2, 1, device=device) + eps
        # 50% of points in the"Far" region
        r_far = (R_max - split_point) * torch.rand(N_f // 2, 1, device=device) + split_point
        r_c = torch.cat([r_near, r_far], dim=0)

    r_c.requires_grad_(True) # Enable gradients for PINN training

    u, du, d2u = get_derivatives(r_c)
    # Right now we treat E as a constant in the loss 
    res = pde_residual(r_c, u, d2u, E)
    loss_pde = (res**2).mean()
    # Normalization loss
    u_norm_pred = u_ansatz(r_grid_norm) 
    integral_approx = torch.sum(u_norm_pred**2) * dr  # Simle way to Compute Integral: Sum(height^2) * width
    loss_norm = (integral_approx - 1.0)**2
    # Total Loss
    loss = loss_pde + w_norm * loss_norm
    loss.backward()
    optimizer.step()

    # Update the energy via the Rayleigh Quotient
    # E = <u|H|u> / <u|u>
    if epoch % 10 == 0:
        with torch.no_grad():
             # H u = -0.5 u'' - (1/r) u
             # We reuse the derivatives we calculated (detached from graph)
             H_u = -0.5 * d2u - (1.0 / r_c) * u
             
             numerator = torch.sum(u * H_u)
             denominator = torch.sum(u * u)
             E_new = numerator / (denominator + 1e-8)
    
             # Soft update(moving average)
             E.data = 0.9 * E.data + 0.1 * E_new
             model.E_history.append(E.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Energy: {E.item():.5f} Ha")


E_hist = np.array(model.E_history)
epochs_hist = np.arange(len(E_hist)) * 10

r_plot = np.linspace(eps, 15.0, 500)
r_plot_t = torch.tensor(r_plot, dtype=torch.float32, device=device).view(-1, 1)

r_eval = np.linspace(1e-5, 15.0, 500)
r_eval_t = torch.tensor(r_eval, dtype=torch.float32, device=device).view(-1, 1)
model.eval()
with torch.no_grad():
    # Model predicts u(r) = r * R(r)
    u_pred = u_ansatz(r_eval_t).cpu().numpy().flatten()
# Psi(r) = u(r) / r
Psi_pred = u_pred / r_eval
Psi_exact = 2.0 * np.exp(-r_eval)
# Exact solution: u(r) = 2* r * exp(-r)
u_exact = 2.0 * r_plot * np.exp(-r_eval)
u_exact_squared = u_exact**2
u_pred_squared = u_pred**2 

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(epochs_hist, E_hist, label="Learned E")
ax[0].axhline(y=-0.5, color='r', linestyle='--', label='Exact (-0.5)')
ax[0].set_title("Energy Convergence")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Energy (Ha)")
ax[0].legend()
ax[0].grid(True)
ax[1].plot(r_eval, u_pred**2, 'b-', linewidth=2, label=r"PINN $R(r)^2$")
ax[1].plot(r_eval, u_exact**2, 'r--', linewidth=2, label=r"Exact $R(r)^2")
ax[1].set_title("Radial Wavefunction (l=0)")
ax[1].set_xlabel("r (Bohr)")
ax[1].legend()
ax[1].grid(True)
plt.tight_layout()
plt.show()

# 5. Compute the Squared Error Term (from Eq 4.1)
# Error = | |Psi_pred|^2 - |Psi_exact|^2 |
# Here Psi corresponds to the radial part R(r)
abs_error_sq = np.abs(Psi_pred**2 - Psi_exact**2)
mae_scalar = np.mean(abs_error_sq)
print(f"Mean Absolute Error (MAE) on Wavefunction Squared: {mae_scalar:.6f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_eval, abs_error_sq, color='#4472C4', linewidth=2.5, label=r'$|Psi_{pred}|^2 - |Psi_{exact}|^2|$')
ax.set_xlabel(r'$r \ [a_0]$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'Abs. error on $Psi^2 \ [a_0^{-3}]$', fontsize=16, fontweight='bold')
ax.set_title(r'Absolute Error of $Psi^2$', fontsize=14)
ax.set_xlim(0, 10) 
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()


abs_error_sq_u = np.abs(u_pred**2 - u_exact**2)
mae_scalar_u = np.mean(abs_error_sq_u)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_eval, abs_error_sq_u, color='#4472C4', linewidth=2.5, label=r'$|R_{pred}|^2 - |R_{exact}|^2|$')
ax.set_xlabel(r'$r \ [a_0]$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'Abs. error on $R^2 \ [a_0^{-3}]$', fontsize=16, fontweight='bold')
ax.set_title('Absolute Error of R^2', fontsize=14)
ax.set_xlim(0, 10) 
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()


