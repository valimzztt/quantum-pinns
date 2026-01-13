import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from  utils.memory_usage import get_memory_usage
# We aim to solve the H atom in 3D dimensions
class HydrogenPINN(nn.Module):
    """ def __init__(self):
        super(HydrogenPINN, self).__init__()
        
        # Standard MLP: 1 deep layer with 32 neurons
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # This is the trainable energy parameter: we tell pytorch to Treat this tensor as a weight of the model
        # We initialize it randomly to -1 
        self.E = nn.Parameter(torch.tensor([-1.0])) """
    def __init__(self, width=32, depth=2):
        super().__init__()
        
        # Dynamic Architecture Construction
        layers = []
        # Input layer (3 -> width)
        layers.append(nn.Linear(3, width))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
            
        # Output layer (width -> 1)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        
        # This is the trainable energy parameter: we tell pytorch to Treat this tensor as a weight of the model
        # We initialize it randomly to -1 
        self.E = nn.Parameter(torch.tensor([-1.0])) 
    def forward(self, x):
        # Calculate distance r
        r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
        # Ansatz: e^(-r). This helps the network drastically, but the NN
        # still needs to learn the normalization scaling factor.
        ansatz = torch.exp(-1.0 * r)
        return ansatz * (1.0 + self.net(x))

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
    
    # Normalization Constraint (Required to fix the scale!)
    # We penalize if the mean value of Psi^2 is not consistent with the domain volume
    # For simplicity in this demo, we just anchor the center: Psi(0) should be approx 1/sqrt(pi) = 0.56
    # Or simpler: Just ensure it's not zero.
    center_point = torch.zeros(1, 3)
    psi_center = model(center_point)
    # Target value at origin for 1s orbital is 1/sqrt(pi) approx 0.564
    norm_loss = (psi_center - 0.564)**2 
    
    return torch.mean(residual**2) + norm_loss

def train(N_f, epochs):
    print(f"\n--- Training with N_f = {N_f} points ---")
    model = HydrogenPINN(width=32, depth=2).to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Training Loop
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        # Sample random points in 3D box [-5, 5]
        # More points (N_f) = Better resolution of the space
        inputs = (torch.rand(N_f, 3) * 10.0 - 5.0).float().requires_grad_(True)
        
        loss = physics_loss(model, inputs)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.5f}, Energy={model.E.item():.4f}")

    return model

if __name__ == "__main__":
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    start_time = time.time()
    N_f_label =  "N_f=3k"
    model_trained = train(N_f=3000, epochs=1000)

    # Create z-axis points for evaluation
    z_vals = np.linspace(-5, 5, 200)

    # Create input tensor: x=0, y=0, z varies
    inputs = np.zeros((200, 3))
    inputs[:, 2] = z_vals # Set z column
    inputs_torch = torch.tensor(inputs, dtype=torch.float32)

    # Get Simulation Prediction
    with torch.no_grad():
        psi_pred = model_trained(inputs_torch).numpy().flatten()
    # probability density
    prob_density_sim = psi_pred**2

    # Calculate Analytical Solution: Psi = (1/sqrt(pi)) * e^(-r)
    # r = |z| along the z-axis
    r_vals = np.abs(z_vals)
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

    print(f"Current CPU Tensor Memory: {torch.cuda.memory_allocated('cpu') / 1024**2:.2f} MB")

    # Plot the wavefunction squared as a function of x and y
    # 1. Create a 2D grid in the x-y plane at z=0
    L_plot = 10.0 # Range from -10 to 10
    N_grid = 200 

    x = np.linspace(-L_plot, L_plot, N_grid)
    y = np.linspace(-L_plot, L_plot, N_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) # z is 0 everywhere on this slice
    # 2. Flatten grid and convert to tensor for the network
    pts_grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    pts_grid_t = torch.tensor(pts_grid, dtype=torch.float32)

    # 3. Predict Psi on the grid
    model_trained.eval() # Set to evaluation mode
    with torch.no_grad():
        # Get psi predictions
        psi_pred_flat = model_trained(pts_grid_t).numpy().flatten()
        # Calculate Probability Density |Psi|^2
        prob_density_flat = psi_pred_flat**2
        
        # Reshape back to 2D grid for plotting
        prob_density_2d = prob_density_flat.reshape(N_grid, N_grid)

    # 4. Take Log10 for better visualization (like the reference image)
    # Add a small epsilon to avoid log(0)
    log_prob = np.log10(prob_density_2d + 1e-10)
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


    model_trained = train_and_plot(N_f=3000, epochs=1000)

    end_time = time.time()
    final_memory = get_memory_usage()

    print("="*40)
    print(f"Training Duration: {end_time - start_time:.2f} seconds")
    print(f"Peak Memory (Approx): {final_memory:.2f} MB")
    print("="*40)

