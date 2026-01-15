import torch
import numpy as np


class H2PlusPinnBenchmark:
    def __init__(self, device='cpu', n_test=50000, box_size=10.0):
        self.device = device
        self.box_size = box_size
        
        # 1. GENERATE FIXED TEST SET
        # We create these ONCE. All models will be tested on these exact same points.
        # We use a mix of near-field and far-field to be fair.
        
        # 50% points in core (r < 2)
        pts_core = torch.randn(n_test // 2, 3, device=device) * 1.5
        
        # 50% points in box (uniform)
        pts_tail = (torch.rand(n_test // 2, 3, device=device) * 2 * box_size) - box_size
        
        self.test_pts = torch.cat([pts_core, pts_tail], dim=0)
        
        # 2. COMPUTE EXACT SOLUTION ONCE
        # Ground Truth Hydrogen 1s: psi = (1/sqrt(pi)) * exp(-r)
        r = torch.sqrt(torch.sum(self.test_pts**2, dim=1, keepdim=True))
        self.psi_exact = (1.0 / np.sqrt(np.pi)) * torch.exp(-r)
        
        # Pre-calculate norm of exact solution for relative error
        self.norm_exact = torch.norm(self.psi_exact, p=2)

    def evaluate(self, model, training_time_sec):
        """
        Takes a trained model and returns a dictionary of metrics.
        """
        model.eval() # Set to evaluation mode
        
        with torch.no_grad():
            # Run prediction using the model's specific ansatz/forward pass
            # Note: We assume model has a method or logic for its ansatz.
            # If your ansatz is external, pass the ansatz function here instead of model.
            
            # Recreating the ansatz logic here for safety:
            r = torch.sqrt(torch.sum(self.test_pts**2, dim=1, keepdim=True) + 1e-8)
            nn_out = model(self.test_pts)
            # Assuming the model returns raw NN output and we apply ansatz:
            # If your model.forward() ALREADY applies exp(-r), just use: psi_pred = model(self.test_pts)
            # based on your previous code, let's assume forward() is just the NN:
            psi_pred = torch.exp(-r) * nn_out
            
            # --- METRIC 1: Relative L2 Error ---
            # We must handle the sign ambiguity! 
            # The NN might learn -Psi instead of +Psi. Both are valid.
            # We check correlation to flip sign if needed.
            corr = torch.sum(psi_pred * self.psi_exact)
            if corr < 0:
                psi_pred = -psi_pred
                
            error_vec = psi_pred - self.psi_exact
            l2_error = torch.norm(error_vec, p=2) / self.norm_exact
            
            # --- METRIC 2: Energy Error ---
            # Extract E from model
            e_pred = model.E.item()
            e_error = abs(e_pred - (-0.5))
            
            return {
                "L2_Error_Psi": l2_error.item(),
                "Abs_Error_E": e_error,
                "Time_Sec": training_time_sec,
                "Final_Energy": e_pred,
                "Params": sum(p.numel() for p in model.parameters())
            }