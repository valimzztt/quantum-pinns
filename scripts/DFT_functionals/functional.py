import torch
import torch.nn as nn
import math

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_exact_hydrogen_data(n_points=1000):
    # r is the root leaf
    r = torch.linspace(0.1, 5.0, n_points, device=device).view(-1, 1)
    r.requires_grad_(True)
    
    # Calculate rho and grad_rho inside the graph of r
    rho = (1.0 / math.pi) * torch.exp(-2.0 * r)
    
    # We need grad_rho to be part of the graph too
    # Method 1: Analytical (Keeps graph)
    grad_rho = -2.0 * (1.0 / math.pi) * torch.exp(-2.0 * r)
    
    # Method 2: Autograd (Alternative if formula unknown)
    # grad_rho = torch.autograd.grad(rho, r, torch.ones_like(rho), create_graph=True)[0]
    
    return r, rho, grad_rho

# --------- 3. Neural Network for Kinetic Energy Functional ---------
class KineticFunctionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 dims (rho, grad_rho) -> Output: 1 dim (Energy Density tau)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Softplus(), # Smooth activation is crucial for derivatives
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )
    
    def forward(self, rho, grad_rho):
        # We perform a log-transform on inputs to help numerical stability
        # (Since rho decays exponentially, inputs vary by orders of magnitude)
        x = torch.cat([torch.log(rho + 1e-16), torch.log(torch.abs(grad_rho) + 1e-16)], dim=1)
        tau = self.net(x)
        return tau

def compute_functional_derivative(model, rho, grad_rho, r):
    # We need 'rho' and 'grad_rho' to be part of the graph connected to 'r'
    # BUT we also need them to be leaves for the first derivative d(tau)/d(rho).
    
    # Trick: We create new variables that are equal to rho/grad_rho but are leaf nodes
    # for the first step, then we manually connect the gradients later or use
    # create_graph=True carefully.
    
    # Actually, a cleaner way for PyTorch in this specific context (Functional Derivative)
    # is to calculate gradients w.r.t inputs, then use grad_rho (which is d_rho/d_r)
    # to compute the divergence via chain rule explicitly.
    
    # 1. Enable grad tracking on inputs for the network
    rho.requires_grad_(True)
    grad_rho.requires_grad_(True)
    
    # 2. Forward pass
    tau = model(rho, grad_rho)
    
    # 3. First derivatives (Gradient of Network w.r.t its inputs)
    # We want d(tau)/d(rho) and d(tau)/d(grad_rho)
    grads = torch.autograd.grad(
        tau, 
        [rho, grad_rho], 
        grad_outputs=torch.ones_like(tau), 
        create_graph=True,
        retain_graph=True # Important to keep graph for the next derivative
    )
    dtau_drho = grads[0]       # Partial derivative w.r.t density
    dtau_dgrad = grads[1]      # Partial derivative w.r.t density gradient
    
    # 4. Divergence Term: div( dtau_dgrad )
    # We need d/dr ( dtau_dgrad ).
    # Since dtau_dgrad is a function of rho(r) and grad_rho(r), it is implicitly a function of r.
    # We can ask autograd to differentiate it w.r.t r directly because r is at the root.
    
    # However, 'dtau_dgrad' was computed using 'rho' and 'grad_rho' which were
    # passed in as arguments. If those arguments were detached or if the graph 
    # splits, this fails.
    
    # ROBUST FIX: Use the Chain Rule explicitly for the divergence.
    # d(f)/dr = (df/drho * drho/dr) + (df/dgrad * dgrad/dr)
    # But since we already have the full graph from r -> rho -> net, 
    # we can just run grad(dtau_dgrad, r).
    
    # The error happened because we used .detach() in the previous snippet.
    # In this version, ensure 'rho' and 'grad_rho' passed in are NOT detached 
    # from 'r' in the main loop.
    
    d_term_dr = torch.autograd.grad(
        dtau_dgrad, 
        r, 
        grad_outputs=torch.ones_like(dtau_dgrad), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # Radial divergence formula
    divergence = d_term_dr + (2.0/r) * dtau_dgrad
    
    v_kinetic = dtau_drho - divergence
    return v_kinetic

# --------- 4. Training Loop ---------
net = KineticFunctionalNet().to(device)
# We treat the chemical potential mu as a learnable parameter (Inverse Problem)
mu = torch.nn.Parameter(torch.tensor(-0.5, device=device)) 
opt = torch.optim.Adam(list(net.parameters()) + [mu], lr=0.001)

r, rho, grad_rho = get_exact_hydrogen_data()
v_ext = -1.0 / r  # Known external potential

print("Training Functional Discovery PINN...")
for step in range(4000):
    
    # 1. Predict Kinetic Potential via Functional Derivative
    # Note: We pass 'r' only to calculate the divergence term, not as input to the net!
    v_kin_pred = compute_functional_derivative(net, rho, grad_rho, r)
    
    # 2. The Physics Constraint (Euler-Lagrange)
    # The sum of potentials must equal the chemical potential constant
    # v_kin + v_ext = mu
    residual = v_kin_pred + v_ext - mu
    
    loss = torch.mean(residual**2)
    
    opt.zero_grad()
    loss.backward(retain_graph=True)
    opt.step()
    
    
    print(f"Step {step}: Loss = {loss.item():.6f}, Mu (Pred) = {mu.item():.4f}")

print("\nTarget Mu (Hydrogen Ground State): -0.5")
print(f"Learned Mu: {mu.item():.4f}")