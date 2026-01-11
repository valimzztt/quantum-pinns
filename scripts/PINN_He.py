import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def compute_laplacian(psi, x):
    """
    Computes the Laplacian of psi with respect to input coordinates x.
    This corresponds to the Kinetic Energy operator.
    """
    # First derivative (Gradient)
    grads = torch.autograd.grad(psi, x, 
                                grad_outputs=torch.ones_like(psi), 
                                create_graph=True, 
                                retain_graph=True)[0]
    
    # Second derivative (Divergence of Gradient)
    # We sum the 2nd derivatives across all 6 input dimensions
    laplacian = 0
    for i in range(x.shape[1]): # Iterate over x1, y1, z1, x2, y2, z2
        grad_i = grads[:, i].view(-1, 1)
        grad_2 = torch.autograd.grad(grad_i, x, 
                                     grad_outputs=torch.ones_like(grad_i), 
                                     create_graph=True, 
                                     retain_graph=True)[0]
        laplacian += grad_2[:, i].view(-1, 1)
        
    return laplacian

def physics_loss(model, x):
    """
    Calculates the residual of the Schrodinger equation:
    H_psi - E_psi = 0
    """
    x.requires_grad = True
    psi = model(x)
    
    # --- 1. Kinetic Energy Term (-0.5 * Laplacian) ---
    # In atomic units, mass = 1, h_bar = 1
    laplacian = compute_laplacian(psi, x)
    kinetic = -0.5 * laplacian

    # --- 2. Potential Energy Term (V) ---
    # Extract coordinates to calculate distances
    r1 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
    r2 = torch.sqrt(x[:, 3:4]**2 + x[:, 4:5]**2 + x[:, 5:6]**2 + 1e-6)
    
    # Distance between electrons (r12)
    r12 = torch.sqrt((x[:, 0:1]-x[:, 3:4])**2 + 
                     (x[:, 1:2]-x[:, 4:5])**2 + 
                     (x[:, 2:3]-x[:, 5:6])**2 + 1e-6)

    Z = 2.0 # Helium nucleus charge
    potential = - (Z / r1) - (Z / r2) + (1.0 / r12)
    
    # --- 3. Schrödinger Residual ---
    # H * Psi = (Kinetic + Potential) * Psi
    # Residual = (H - E) * Psi
    h_psi = kinetic + (potential * psi)
    residual = h_psi - (model.E * psi)
    
    # Mean Squared Error of the residual
    loss_f = torch.mean(residual**2)
    
    # --- 4. Normalization / Trivial Solution Penalty ---
    # We punish the network if the integral of Psi^2 is too small or if Psi is zero.
    # A simple approach for training is fixing Psi at a reference point or 
    # ensuring mean(Psi^2) is close to 1 (if sampling is uniform).
    norm_loss = (torch.mean(psi**2) - 1.0)**2
    
    return loss_f + norm_loss

# --- Training Setup ---


import torch
import torch.nn as nn
import torch.optim as optim

class HeliumAnsatzPINN(nn.Module):
    def __init__(self):
        super(HeliumAnsatzPINN, self).__init__()
        
        # Standard MLP to learn the correlation term (electron interactions)
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )
        
        # Trainable Energy Parameter
        # Starting closer to -2.903 helps, but we can start generic (-2.0)
        self.E = nn.Parameter(torch.tensor([-2.5])) 
        
        # Trainable 'Effective Charge' Z 
        # While Z=2 is the physical charge, treating it as a parameter allows 
        # the network to find the optimal 'shielded' charge (variational parameter).
        self.Z_param = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x):
        # 1. Calculate distances r1 and r2 from input coordinates
        # x shape: [batch_size, 6] -> (x1, y1, z1, x2, y2, z2)
        r1 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
        r2 = torch.sqrt(x[:, 3:4]**2 + x[:, 4:5]**2 + x[:, 5:6]**2 + 1e-6)
        
        # 2. Compute the Physics Ansatz (Hydrogenic product state)
        # Psi_approx = e^(-Z*r1) * e^(-Z*r2)
        # We enforce Z to be positive using absolute value or softplus if strictly training it
        Z = torch.abs(self.Z_param) 
        ansatz = torch.exp(-Z * r1) * torch.exp(-Z * r2)
        
        # 3. Compute Neural Network Correction
        # The NN learns the deviation from the simple independent electron model
        nn_correction = self.net(x)
        
        # 4. Combine them
        # Result = Ansatz * (1 + NN_output)
        # The (1 + ...) form ensures that if NN is zero, we still have a valid orbital.
        psi = ansatz * (1.0 + nn_correction)
        
        return psi

# --- Usage Example ---

# Initialize
model = HeliumAnsatzPINN()

# Define Optimizer (Now learning weights, Energy E, and effective Z)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Function to run one training step (conceptual)
def train_step(model, optimizer, batch_size=1024):
    optimizer.zero_grad()
    
    # Sample points (Importance sampling is better, but normal dist works for demo)
    inputs = (torch.randn(batch_size, 6) * 1.0).float().requires_grad_(True)
    
    # Use the same physics_loss function defined in the previous snippet
    # The 'physics_loss' function calls model(inputs), which now uses the Ansatz
    loss = physics_loss(model, inputs) 
    
    loss.backward()
    optimizer.step()
    return loss.item(), model.E.item()

# Quick test run
print("Starting Training with Ansatz...")
for i in range(1000):
    l, e = train_step(model, optimizer)
    if i % 100 == 0:
        print(f"Iter {i}: Loss={l:.5f}, Energy={e:.4f}, Eff. Charge={model.Z_param.item():.3f}")