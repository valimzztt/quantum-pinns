import tensorflow as tf
import numpy as np

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class HeliumPINN(tf.keras.Model):
    def __init__(self):
        super(HeliumPINN, self).__init__()
        
        # --- The Neural Network ---
        # Inputs: (x, y, z) coordinates
        # Outputs: [psi (orbital), V (electron potential)]
        self.hidden_layers = [
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh')
        ]
        
        # Output layer: 2 values (psi, V_eff)
        # We don't use activation here
        self.output_layer = tf.keras.layers.Dense(2)
        
        # --- Trainable Eigenvalue ---
        # Initialize energy estimate (He ground state is approx -2.9 a.u.)
        self.E = tf.Variable(-2.5, dtype=tf.float32, name='Energy')

    def call(self, inputs):
        # inputs shape: (Batch_Size, 3)
        x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        r = tf.sqrt(x**2 + y**2 + z**2 + 1e-9) # Add epsilon for stability at 0
        
        # Forward pass through hidden layers
        net = inputs
        for layer in self.hidden_layers:
            net = layer(net)
        
        raw_output = self.output_layer(net)
        raw_psi = raw_output[:, 0:1]
        raw_V   = raw_output[:, 1:2]
        
        # --- PHYSICAL ANSATZ (Soft Constraints) ---
        # Force psi to decay at infinity. 
        # This acts as a "guide" so the NN focuses on the fluctuations.
        psi = raw_psi * tf.exp(-r)
        
        # We leave V unconstrained (or you could apply Coulomb decay 1/r)
        return psi, raw_V

# --- The Physics Engine (Derivatives & Loss) ---

@tf.function
def compute_loss(model, inputs_xyz):
    # Enable gradient tracking
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs_xyz)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(inputs_xyz)
            
            # Get predictions
            psi, V_eff = model(inputs_xyz)
            
        # First derivatives (gradients)
        grad_psi = tape1.gradient(psi, inputs_xyz)
        grad_V   = tape1.gradient(V_eff, inputs_xyz)
        
    # Second derivatives (Laplacians)
    # We sum the diagonal of the Hessian (d2/dx2 + d2/dy2 + d2/dz2)
    lap_psi = tf.reduce_sum(tape2.gradient(grad_psi, inputs_xyz), axis=1, keepdims=True)
    lap_V   = tf.reduce_sum(tape2.gradient(grad_V, inputs_xyz), axis=1, keepdims=True)
    
    del tape1, tape2 # Clear tape memory
    
    # --- 1. Physics Constants ---
    Z = 2.0  # Helium Nucleus charge
    x, y, z = inputs_xyz[:, 0:1], inputs_xyz[:, 1:2], inputs_xyz[:, 2:3]
    r = tf.sqrt(x**2 + y**2 + z**2 + 1e-9)
    V_ext = -Z / r
    
    # --- 2. Fock Equation Residual ---
    # (-0.5 * Laplacian + V_ext + V_eff) * psi = E * psi
    # Note: For He RHF, the electron sees V_ext and the potential from the *other* electron.
    kinetic = -0.5 * lap_psi
    potential = (V_ext + V_eff) * psi
    
    residual_fock = kinetic + potential - (model.E * psi)
    
    # --- 3. Poisson Equation Residual ---
    # Del^2 V_eff = -4 * pi * |psi|^2
    # The source is the charge density of the *other* electron (which is just |psi|^2)
    rho = psi ** 2
    residual_poisson = lap_V + (4.0 * np.pi * rho)
    
    # --- 4. Loss Combination ---
    loss_fock = tf.reduce_mean(tf.square(residual_fock))
    loss_poisson = tf.reduce_mean(tf.square(residual_poisson))
    
    return loss_fock, loss_poisson, psi

@tf.function
def compute_normalization_loss(psi, inputs_xyz, domain_volume):
    # Simple Monte Carlo Integration for Normalization
    # Integral(|psi|^2) = 1
    
    N = tf.cast(tf.shape(psi)[0], tf.float32)
    integral = tf.reduce_sum(psi**2) * (domain_volume / N)
    loss_norm = tf.square(integral - 1.0)
    return loss_norm

# --- Training Loop ---

def train():
    model = HeliumPINN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Sampling Domain (Cube from -5 to 5)
    L = 5.0
    domain_vol = (2*L)**3
    batch_size = 1000
    
    print(f"Starting Training. Initial Energy Guess: {model.E.numpy():.4f}")
    
    for epoch in range(5001):
        # 1. Sample Points
        # Mix of uniform sampling and sampling near nucleus (Gaussian)
        n_uniform = batch_size // 2
        n_normal = batch_size - n_uniform
        
        pts_uniform = tf.random.uniform((n_uniform, 3), -L, L)
        pts_normal = tf.random.normal((n_normal, 3), 0, 1.0) # Focus on nucleus
        inputs = tf.concat([pts_uniform, pts_normal], axis=0)
        
        with tf.GradientTape() as tape:
            # PDE Losses
            l_fock, l_poisson, psi = compute_loss(model, inputs)
            
            # Normalization Loss (Crucial: prevents trivial solution psi=0)
            l_norm = compute_normalization_loss(psi, inputs, domain_vol)
            
            # Weighted Sum
            total_loss = l_fock + l_poisson + (10.0 * l_norm)
            
        # Backprop
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.numpy():.5f} | Energy E: {model.E.numpy():.5f} Ha")
            # Expected He Energy is approx -2.86 to -2.90 Ha depending on basis limit

    return model

# Run it
if __name__ == "__main__":
    trained_model = train()