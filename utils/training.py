import torch
import torch.nn as nn
import torch.optim as optim

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

def train_model(model,N_f, epochs, device='cpu', learning_rate = 0.005):
    """
    Trains a given model instance and returns the final loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Pre-calculate box volume for normalization (L=6.0)
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
            print(f"Epoch {epoch}: Loss={loss.item():.10f}, Energy={model.E.item():.4f}")

    return model