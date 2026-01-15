import torch
class HPinnBenchmark1D:
    def __init__(self, device='cpu', n_test=50000, max_r=20.0):
        self.device = device
        self.max_r = max_r
        # We start by generating a fixed test set (1D radial)
        # 1. GENERATE FIXED TEST SET (1D Radial)
        # Core: Half-Normal distribution (Gaussian folded at 0)
        pts_core = torch.abs(torch.randn(n_test // 2, 1, device=device)) * 2.0
        # Tail: Uniform distribution from 0 to max_r
        pts_tail = torch.rand(n_test // 2, 1, device=device) * max_r

        self.test_r, _ = torch.sort(torch.cat([pts_core, pts_tail], dim=0), dim=0)         # Sort them just for cleaner plotting
        self.test_r = torch.clamp(self.test_r, min=1e-6)
        # Exact Radial Wavefunction Psi(r) = 2 * exp(-r)
        self.Psi_exact  = 2.0 * torch.exp(-self.test_r)
        self.norm_exact = torch.norm(self.Psi_exact, p=2)  # Pre-calculate norm for relative error

    def evaluate(self, model, training_time_sec):
        model.eval()
        with torch.no_grad():
            # We get the prediction of u(r), which is R(r) (in Paolo´s thesis)
            u_pred = model(self.test_r)
            # Check for global phase flip (we keep positive as the default)
            if torch.mean(u_pred) < 0:
                u_pred = -u_pred
            Psi_pred = u_pred / self.test_r       # Psi(r) = u(r) / r
            # Metric 1: L2 Relative Error on the Wavefunction
            error_vec = Psi_pred - self.Psi_exact
            l2_error = torch.norm(error_vec, p=2) / self.norm_exact 
            # Metric 2: MAE on the Radial Density (R^2), metric that Paolo used
            mae_density = torch.mean(torch.abs(Psi_pred**2 - self.Psi_exact**2))
            # Metric 3: Energy Error
            e_pred = model.E.item()
            e_error = abs(e_pred - (-0.5))
            
            return {
                "L2_Error_R": l2_error.item(),
                "MAE_Density_R2": mae_density.item(),
                "Abs_Error_E": e_error,
                "Time_Sec": training_time_sec,
                "Final_Energy": e_pred
            }
        