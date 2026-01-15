import torch
import numpy as np


class HPinnBenchmark3D:
    def __init__(self, device='cpu', n_test=50000, box_size=10.0):
        self.device = device
        self.box_size = box_size
        # We start by generating a fixed test set 
        # We use a mix of near-field and far-field
        pts_core = torch.randn(n_test // 2, 3, device=device) * 1.5  # 50% points in core (r < 2)
        # 50% points in box (uniform)
        print(pts_core)
        pts_tail = (torch.rand(n_test // 2, 3, device=device) * 2 * box_size) - box_size
        print(pts_tail)
        self.test_pts = torch.cat([pts_core, pts_tail], dim=0)
        
        # Ground Truth Hydrogen 1s: psi = (1/sqrt(pi)) * exp(-r)
        r = torch.sqrt(torch.sum(self.test_pts**2, dim=1, keepdim=True))
        self.psi_exact = (1.0 / np.sqrt(np.pi)) * torch.exp(-r)
        self.norm_exact = torch.norm(self.psi_exact, p=2)

    def evaluate(self, model, training_time_sec):
        """
        Takes a trained model and returns a dictionary of metrics.
        """
        model.eval() # Set to evaluation mode
        with torch.no_grad():
            # Run prediction using the model's specific ansatz/forward pass
            r = torch.sqrt(torch.sum(self.test_pts**2, dim=1, keepdim=True) + 1e-8)
            # Assuming the model returns raw NN output and we apply ansatz:
            # Since our model.forward() already applies exp(-r): 
            psi_pred = torch.exp(-r) *model(self.test_pts) # this is wrong
            psi_pred = model(self.test_pts)
            error_vec = psi_pred**2 - self.psi_exact**2
            # Metric 1: L2 Relative Error on the Wavefunction
            l2_error = torch.norm(error_vec, p=2) / self.norm_exact
            # Metric 2: Energy Error
            e_pred = model.E.item()
            e_error = abs(e_pred - (-0.5)) # -0.5 known groundstate energy
            return {
                "L2_Error_Psi": l2_error.item(),
                "Abs_Error_E": e_error,
                "Time_Sec": training_time_sec,
                "Final_Energy": e_pred,
                "Params": sum(p.numel() for p in model.parameters())
            }