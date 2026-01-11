# This script aims to use Physics-Informed Neural Networks (PINNs) to solve the Schrödinger
# equation for the hydrogen atom. 

import torch
import torch.nn as nn
import torch.optim as optim

class HydrogenPINN(nn.Module):
    def __init__(self):
        super(HydrogenPINN, self).__init__()
        
        # Input: Just 3 coordinates (x, y, z)
        self.net = nn.Sequential(
            nn.Linear(3, 32), # Smaller network needed
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # We start E at a random guess (e.g., -1.0). Target is -0.5.
        initial_E = -0.25
        self.E = nn.Parameter(torch.tensor([initial_E]))

    def forward(self, x):
        # Calculate distance r from origin
        # x shape: [batch_size, 3]
        r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
        
        # Ansatz: e^(-r). 
        # For Hydrogen, Z=1. We can hardcode this or let the NN learn it.
        # Here we use the ansatz to enforce the boundary condition at infinity.
        ansatz = torch.exp(-1.0 * r)
        
        # Network prediction
        nn_out = self.net(x)
        
        # Final Wavefunction
        psi = ansatz * (1.0 + nn_out)
        return psi

def compute_laplacian_3d(psi, x):
    # Same logic as before, but only iterating over 3 dimensions
    grads = torch.autograd.grad(psi, x, 
                                grad_outputs=torch.ones_like(psi), 
                                create_graph=True, 
                                retain_graph=True)[0]
    
    laplacian = 0
    for i in range(3): # x, y, z
        grad_i = grads[:, i].view(-1, 1)
        grad_2 = torch.autograd.grad(grad_i, x, 
                                     grad_outputs=torch.ones_like(grad_i), 
                                     create_graph=True, 
                                     retain_graph=True)[0]
        laplacian += grad_2[:, i].view(-1, 1)
    return laplacian

def h_atom_loss(model, x):
    x.requires_grad = True
    psi = model(x)
    
    # 1. Kinetic Energy (-0.5 * Laplacian)
    laplacian = compute_laplacian_3d(psi, x)
    kinetic = -0.5 * laplacian
    
    # 2. Potential Energy (-1/r)
    r = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + x[:, 2:3]**2 + 1e-6)
    potential = - (1.0 / r)
    
    # 3. Residual
    # (T + V)Psi = E Psi  ->  (T + V - E)Psi = 0
    residual = (kinetic + potential * psi) - (model.E * psi)
    
    # 4. Normalization constraint
    # Simple Monte Carlo integration approximation for normalization
    # (Assuming uniform sampling volume V, but here we just pin mean square)
    norm_loss = (torch.mean(psi**2) - 1.0)**2
    
    return torch.mean(residual**2) + norm_loss

# --- Training ---
model = HydrogenPINN()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print(f"Start Energy: {model.E.item()}")

epochs = 501 # Number of training epochs

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Sample 3D points
    inputs = (torch.randn(1000, 3) * 2.0).float()
    
    loss = h_atom_loss(model, inputs)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.5f}, Energy {model.E.item():.5f}")

# You should see Energy converge very close to -0.5