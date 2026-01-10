import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# We aim to solve the H atom in 3D dimensions
class HydrogenPINN(nn.Module):
    def __init__(self):
        super(HydrogenPINN, self).__init__()
        
        # Standard MLP: this time we take 
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # This is the trainable energy parameter
        # Random init: -1.0
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

# 
def train_and_plot(N_f=3000, epochs=1000):
    print(f"\n--- Training with N_f = {N_f} points ---")
    
    model = HydrogenPINN()
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


N_f_label =  "N_f=30k"
model_trained = train_and_plot(N_f=30000, epochs=3000)

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
plt.ylim(1e-5, 0.5) # Match the y-limits of your image
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()