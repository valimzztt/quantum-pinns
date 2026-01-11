import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class H2PlusPINN(nn.Module):
    def __init__(self, bond_length=2.0):
        super(H2PlusPINN, self).__init__()
        
        # Geometry: Protons at +z and -z
        self.R = bond_length
        self.z_shift = self.R / 2.0
        
        # Standard MLP
        self.net = nn.Sequential(
            nn.Linear(3, 64),       # Increased neurons slightly for complexity
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Trainable Energy Parameter
        # For H2+, exact electronic energy at R=2.0 is approx -1.1 atomic units
        self.E = nn.Parameter(torch.tensor([-1.2]))

    def forward(self, x):
        # Calculate distances to both nuclei
        # Nucleus 1 at (0, 0, -z_shift)
        r1 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + (x[:, 2:3] + self.z_shift)**2 + 1e-6)
        
        # Nucleus 2 at (0, 0, +z_shift)
        r2 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + (x[:, 2:3] - self.z_shift)**2 + 1e-6)
        
        # LCAO Ansatz: sum of 1s orbitals on both centers
        # This creates the two peaks naturally
        ansatz = torch.exp(-1.0 * r1) + torch.exp(-1.0 * r2)
        
        # Prediction
        return ansatz * (1.0 + self.net(x))

# --- 2. Loss Function ---
def physics_loss(model, x):
    x.requires_grad = True
    psi = model(x)
    
    # --- Kinetic Energy (Laplacian) ---
    grads = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    laplacian = 0
    for i in range(3):
        grad_2 = torch.autograd.grad(grads[:, i], x, torch.ones_like(grads[:, i]), create_graph=True)[0]
        laplacian += grad_2[:, i].view(-1, 1)
        
    kinetic = -0.5 * laplacian
    
    # --- Potential Energy (Two Centers) ---
    # V = -1/r1 - 1/r2
    r1 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + (x[:, 2:3] + model.z_shift)**2 + 1e-6)
    r2 = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + (x[:, 2:3] - model.z_shift)**2 + 1e-6)
    potential = (-1.0 / r1) - (1.0 / r2)
    
    # --- Residual: (H - E)Psi = 0 ---
    # Note: We are solving H_elec. Total E = E_elec + 1/R (nuclear repulsion)
    residual = (kinetic + potential * psi) - (model.E * psi)
    
    # --- Normalization Constraint ---
    # Prevent trivial solution (Psi=0). 
    # We anchor the midpoint (0,0,0) to be non-zero (Bonding orbital has density at center)
    center_point = torch.zeros(1, 3)
    psi_center = model(center_point)
    # Roughly target 0.5 to keep scale reasonable
    norm_loss = (psi_center - 0.5)**2 
    
    return torch.mean(residual**2) + norm_loss

# --- 3. Training Function ---
def train_h2plus(N_f=5000, epochs=2000):
    print(f"\n--- Training H2+ PINN with N_f = {N_f} points ---")
    
    model = H2PlusPINN(bond_length=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    loss_history = []

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # Sample random points in 3D box [-5, 5]
        inputs = (torch.rand(N_f, 3) * 10.0 - 5.0).float().requires_grad_(True)
        
        loss = physics_loss(model, inputs)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            # Calculate Total Energy = Electronic E + Nuclear Repulsion (1/R)
            elec_E = model.E.item()
            total_E = elec_E + (1.0 / model.R)
            print(f"Epoch {epoch}: Loss={loss.item():.5f}, Elec_E={elec_E:.4f}, Total_E={total_E:.4f}")
            loss_history.append(loss.item())

    return model

# --- 4. Plotting Function ---
def plot_h2_density(model):
    z_vals = np.linspace(-4, 4, 300)
    
    # Create input tensor: x=0, y=0, z varies
    inputs = np.zeros((300, 3))
    inputs[:, 2] = z_vals 
    inputs_torch = torch.tensor(inputs, dtype=torch.float32)
    
    with torch.no_grad():
        psi_pred = model(inputs_torch).numpy().flatten()
    
    prob_density = psi_pred**2
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(z_vals, prob_density, label='PINN Density ($|\Psi|^2$)', linewidth=3, color='blue')
    
    # Mark Proton Positions
    plt.axvline(x=-model.z_shift, color='red', linestyle='--', alpha=0.5, label='Proton 1')
    plt.axvline(x=model.z_shift, color='red', linestyle='--', alpha=0.5, label='Proton 2')
    
    plt.xlabel(r'$z$ axis (atomic units)', fontsize=14)
    plt.ylabel(r'Probability Density', fontsize=14)
    plt.title(f'H2+ Molecular Ion Density (Bond Length R={model.R})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.show()

# --- 5. Execution ---
model_h2 = train_h2plus(N_f=5000, epochs=2500)
plot_h2_density(model_h2)