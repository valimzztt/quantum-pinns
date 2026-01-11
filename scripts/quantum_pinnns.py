# pip install pennylane torch

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The quantum layer plays the same role as a kernel "
"in kernel methods: it maps inputs into a higher-dimensional feature space where the PDE solution can be represented more simply."
# =========================
# 1) Quantum Circuit (QNN) g_{θ2}
# =========================
n_qubits = 2
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnn(encoded_features, q_weights):
    """
    encoded_features: shape (n_qubits,) or (n_qubits, ) torch tensor
    q_weights: shape (n_layers, n_qubits, 3) trainable angles
    """
    # --- Encoding (angle encoding) ---
    for i in range(n_qubits):
        qml.RY(encoded_features[i], wires=i)

    # --- Variational layers ---
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(q_weights[l, i, 0], wires=i)
            qml.RY(q_weights[l, i, 1], wires=i)
            qml.RZ(q_weights[l, i, 2], wires=i)
        # simple entanglement
        qml.CNOT(wires=[0, 1])

    # outputs: expectation values -> p-dimensional vector
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# =========================
# 2) Classical Pre/Post networks (θ1, θ3)
# =========================
class Preprocessor(nn.Module):
    """f_{θ1}: x -> q(x) in R^m (m = n_qubits here for simplicity)"""
    def __init__(self, width=32, depth=2, out_dim=2):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Postprocessor(nn.Module):
    """h_{θ3}: qhat(x) -> u(x)"""
    def __init__(self, width=32, depth=2, in_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, qhat):
        return self.net(qhat)


class QCPINN(nn.Module):
    """Full architecture: pre -> qnn -> post"""
    def __init__(self):
        super().__init__()
        self.pre = Preprocessor(out_dim=n_qubits)
        self.post = Postprocessor(in_dim=n_qubits)
        self.q_weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))

    def forward(self, x):
        # x: (N,1)
        q = self.pre(x)  # (N, n_qubits)

        # run quantum circuit per sample (simple, not batched)
        qhat_list = []
        for i in range(q.shape[0]):
            qhat_i = qnn(q[i], self.q_weights)           # list length n_qubits
            qhat_list.append(torch.stack(qhat_i))        # (n_qubits,)
        qhat = torch.stack(qhat_list, dim=0)             # (N, n_qubits)

        u = self.post(qhat)  # (N,1)
        return u


def derivative(u, x):
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


# =========================
# 3) Training loop with physics loss
# =========================
def train_qcpinn(steps=2000, lr=2e-3, n_colloc=64):
    model = QCPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps + 1):
        # collocation points in [0, 2]
        x = 2.0 * torch.rand(n_colloc, 1, device=device, requires_grad=True)

        u = model(x)
        du = derivative(u, x)

        # PDE residual: u' + u = 0
        loss_pde = torch.mean((du + u) ** 2)

        # BC: u(0)=1
        x0 = torch.zeros(1, 1, device=device, requires_grad=True)
        u0 = model(x0)
        loss_bc = (u0 - 1.0) ** 2

        loss = loss_pde + 10.0 * loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                x1 = torch.tensor([[1.0]], device=device)
                pred = model(x1).item()
                true = float(np.exp(-1.0))
                print(f"step={step:4d}  loss={loss.item():.3e}  u(1)={pred:.4f}  true={true:.4f}")

    return model


if __name__ == "__main__":
    model = train_qcpinn()
