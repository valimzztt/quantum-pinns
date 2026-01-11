"""
He Hartree PINN (1D radial, spherically symmetric) — PyTorch
-----------------------------------------------------------
Solves (atomic units) for Helium (Z=2) in Hartree mean-field (no exchange):

(1)  -1/2 (phi'' + 2/r phi') + (-Z/r + V_H) phi = eps_orb * phi
(2)   (V_H'' + 2/r V_H') = -8*pi*phi^2          (since rho = 2*phi^2 and ∇²V = -4πrho)

with:
- regularity: phi'(0)=0, V_H'(0)=0  (enforced at r=eps)
- far boundary: phi(Rmax)=0, V_H(Rmax)=2/Rmax
- normalization: 4π ∫ r^2 phi^2 dr = 1

Run:
  python he_hartree_pinn.py
"""

import math
import time
import argparse
import torch
import torch.nn as nn

# -----------------------------
# MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Utilities: derivatives
# -----------------------------
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]

def radial_laplacian(f, r):
    # f: (N,1), r: (N,1) requires_grad=True
    df = grad(f, r)
    d2f = grad(df, r)
    return d2f + (2.0 / r) * df

def trapz(y, x):
    # y, x: 1D tensors with same shape
    return torch.trapz(y, x)


# -----------------------------
# Sampling
# -----------------------------
def sample_r_colloc(n, eps, rmax, power=2.5, device="cpu"):
    # bias sampling toward r ~ 0: r = eps + (rmax-eps) * u^power
    u = torch.rand(n, 1, device=device)
    r = eps + (rmax - eps) * (u ** power)
    return r


# -----------------------------
# Main training
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmax", type=float, default=20.0)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--Z", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_colloc", type=int, default=2000)
    parser.add_argument("--n_grid", type=int, default=2000)
    parser.add_argument("--sample_power", type=float, default=2.5)
    parser.add_argument("--lam_norm", type=float, default=10.0)
    parser.add_argument("--lam_bc", type=float, default=1.0)
    parser.add_argument("--lam_prior", type=float, default=0.1)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    Z = args.Z
    rmax = args.rmax
    eps = args.eps

    # Networks
    phi_net = MLP(1, 1, width=args.width, depth=args.depth).to(device)
    V_net   = MLP(1, 1, width=args.width, depth=args.depth).to(device)

    # Trainable orbital eigenvalue (start negative)
    eps_orb = nn.Parameter(torch.tensor([-1.0], device=device, dtype=torch.float64))

    params = list(phi_net.parameters()) + list(V_net.parameters()) + [eps_orb]
    opt = torch.optim.Adam(params, lr=args.lr)

    # Optional: exponential envelope helps stability (set alpha>0 to enable)
    alpha = 0.0  # try 1.0 if training stalls

    """     def phi(r):
        # simple output; if alpha>0 -> decays
        out = phi_net(r)
        if alpha > 0:
            out = torch.exp(-alpha * r) * out
        return out """
    
    def phi(r):
        return (rmax - r) * phi_net(r)


    def Vh(r):
        return V_net(r)

    def pde_residuals(r):
        r.requires_grad_(True)
        ph = phi(r)
        V  = Vh(r)

        lap_phi = radial_laplacian(ph, r)
        lap_V   = radial_laplacian(V, r)

        # (A) Schrödinger-like residual
        R_phi = -0.5 * lap_phi + (-Z / r + V) * ph - eps_orb * ph

        # (B) Poisson residual: ∇² V_H + 8π φ^2 = 0  (since rho=2φ^2)
        R_V = lap_V + 8.0 * math.pi * (ph ** 2)

        return R_phi, R_V

    def loss_normalization(r_grid):
        ph = phi(r_grid)[:, 0]
        rr = r_grid[:, 0]
        integrand = 4.0 * math.pi * (rr ** 2) * (ph ** 2)
        norm = trapz(integrand, rr)
        return (norm - 1.0) ** 2

    def loss_origin_bc():
        r0 = torch.tensor([[eps]], device=device, dtype=torch.float64, requires_grad=True)
        ph0 = phi(r0)
        V0  = Vh(r0)
        dph0 = grad(ph0, r0)
        dV0  = grad(V0, r0)
        return (dph0 ** 2).mean() + (dV0 ** 2).mean()

    def loss_far_bc():
        rR = torch.tensor([[rmax]], device=device, dtype=torch.float64)
        phR = phi(rR)
        VR  = Vh(rR)
        return (phR ** 2).mean() + ((VR - 2.0 / rmax) ** 2).mean()

    def loss_energy_prior():
        # bound orbital should have eps_orb < 0
        return torch.relu(eps_orb) ** 2

    # fixed grid for normalization + diagnostics
    r_grid = torch.linspace(eps, rmax, args.n_grid, device=device, dtype=torch.float64).view(-1, 1)

    t0 = time.time()
    for it in range(args.iters + 1):
        opt.zero_grad()

        r_c = sample_r_colloc(args.n_colloc, eps, rmax, power=args.sample_power, device=device)
        Rphi, RV = pde_residuals(r_c)

        loss_pde  = (Rphi ** 2).mean() + (RV ** 2).mean()
        loss_norm = loss_normalization(r_grid)
        loss_bc   = loss_origin_bc() + loss_far_bc()
        loss_pr   = loss_energy_prior()

        loss = loss_pde + args.lam_norm * loss_norm + args.lam_bc * loss_bc + args.lam_prior * loss_pr
        loss.backward()
        opt.step()

        if it % args.print_every == 0:
            with torch.no_grad():
                # quick diagnostic: far-field V should approach ~2/r
                VR = Vh(torch.tensor([[rmax]], device=device, dtype=torch.float64)).item()
                phR = phi(torch.tensor([[rmax]], device=device, dtype=torch.float64)).item()
                print(
                    f"it={it:5d}  "
                    f"L={loss.item():.3e}  "
                    f"L_pde={loss_pde.item():.3e}  "
                    f"L_norm={loss_norm.item():.3e}  "
                    f"L_bc={loss_bc.item():.3e}  "
                    f"eps_orb={eps_orb.item(): .6f}  "
                    f"phi(Rmax)={phR:+.2e}  "
                    f"V(Rmax)={VR:.6f}"
                )

    print(f"\nDone. Elapsed: {time.time()-t0:.1f} s")
    print(f"Final eps_orb = {eps_orb.item():.8f} (Hartree)")

    # Save learned profiles on CPU for later use
    with torch.no_grad():
        rg = r_grid.detach().cpu().view(-1)
        ph = phi(r_grid).detach().cpu().view(-1)
        VH = Vh(r_grid).detach().cpu().view(-1)
        out = torch.stack([rg, ph, VH], dim=1)
        torch.save(
            {
                "r": rg,
                "phi": ph,
                "Vh": VH,
                "eps_orb": float(eps_orb.detach().cpu().item()),
                "params": vars(args),
            },
            "he_hartree_pinn_result.pt"
        )
    print("Saved: he_hartree_pinn_result.pt")

    if args.plot:
        import matplotlib.pyplot as plt
        # Plot phi(r) and Vh(r)
        rg = rg.numpy()
        ph = ph.numpy()
        VH = VH.numpy()

        plt.figure()
        plt.plot(rg, ph)
        plt.xlabel("r (a.u.)")
        plt.ylabel("phi(r)")
        plt.title("He Hartree PINN orbital")
        plt.grid(True)

        plt.figure()
        plt.plot(rg, VH)
        plt.plot(rg, 2.0 / rg, linestyle="--")  # asymptotic reference
        plt.xlabel("r (a.u.)")
        plt.ylabel("V_H(r)")
        plt.title("He Hartree PINN Hartree potential")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
