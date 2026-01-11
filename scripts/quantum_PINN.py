# pip install pennylane torch

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# ---- ODE: u'(x) = -u(x), u(0)=1  ----
# We'll enforce ODE residual + boundary condition.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# --- Quantum device (simulator) ---
n_qubits = 1
dev = qml.device("default.qubit", wires=n_qubits)

# --- Quantum circuit: outputs expectation <Z> ---
@qml.qnode(dev, interface="torch", diff_method="backprop")
def q_circuit(x, weights):
    """
    x: scalar tensor
    weights: shape (L, 3) trainable rotation angles for L layers
    """
    # Angle encoding of input
    qml.RY(x, wires=0)

    # Trainable layers
    for w in weights:
        qml.RX(w[0], wires=0)
        qml.RY(w[1], wires=0)
        qml.RZ(w[2], wires=0)

    return qml.expval(qml.PauliZ(0))


class QuantumPINN(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        # Trainable quantum parameters
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, 3))

        # Optional scalar affine map to improve expressivity
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # q_circuit returns in [-1, 1], map to more flexible range
        # x is (N,1). We'll evaluate pointwise for simplicity.
        out = []
        for xi in x.view(-1):
            zi = q_circuit(xi, self.weights)  # scalar
            out.append(zi)
        z = torch.stack(out).view(-1, 1)
        return self.a * z + self.b


def derivative(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]


def train_quantum_pinn(steps=2000, lr=2e-2, n_colloc=64):
    model = QuantumPINN(n_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps + 1):
        # Collocation points in [0, 2]
        x = 2.0 * torch.rand(n_colloc, 1, device=device, requires_grad=True)

        u = model(x)
        du_dx = derivative(u, x)

        # ODE residual: u' + u = 0
        loss_ode = torch.mean((du_dx + u) ** 2)

        # BC: u(0)=1
        x0 = torch.zeros(1, 1, device=device, requires_grad=True)
        u0 = model(x0)
        loss_bc = (u0 - 1.0) ** 2

        loss = loss_ode + 10.0 * loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                # quick check at x=1
                x1 = torch.tensor([[1.0]], device=device)
                pred = model(x1).item()
                true = float(np.exp(-1.0))
                print(f"[QPINN] step={step:4d} loss={loss.item():.3e} u(1)={pred:.4f} true={true:.4f}")

    return model


if __name__ == "__main__":
    q_model = train_quantum_pinn()
