import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

class HeliumPINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Deeper, narrower network often captures cusps better
        self.hidden_layers = [
            tf.keras.layers.Dense(50, activation='tanh') for _ in range(5)
        ]
        self.output_layer = tf.keras.layers.Dense(2) # [psi, V_eff]
        
        # We NO LONGER train E via backprop. We store it just for tracking.
        self.E_history = [] 

    def call(self, inputs):
        x, y, z = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        r = tf.sqrt(x**2 + y**2 + z**2 + 1e-9)
        
        net = inputs
        for layer in self.hidden_layers:
            net = layer(net)
        
        raw_output = self.output_layer(net)
        raw_psi = raw_output[:, 0:1]
        raw_V   = raw_output[:, 1:2]
        
        # Improved Ansatz: 
        # Helium is approx e^(-1.69r). We start with e^(-r) and let NN correct it.
        psi = raw_psi * tf.exp(-r)
        
        return psi, raw_V

# --- Physics Engine ---

@tf.function
def get_derivatives(model, inputs):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(inputs)
            psi, V_eff = model(inputs)
        
        grad_psi = tape1.gradient(psi, inputs)
        grad_V   = tape1.gradient(V_eff, inputs)
        
    lap_psi = tf.reduce_sum(tape2.gradient(grad_psi, inputs), axis=1, keepdims=True)
    lap_V   = tf.reduce_sum(tape2.gradient(grad_V, inputs), axis=1, keepdims=True)
    
    return psi, V_eff, lap_psi, lap_V

def train_and_plot():
    model = HeliumPINN()
    # Lower LR for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) 
    
    # Track current Energy
    current_E = -2.0 
    
    # Training Config
    L = 6.0
    domain_vol = (2*L)**3
    batch_size = 2000
    epochs = 4000
    
    loss_history = []
    
    for epoch in range(epochs):
        # Sample points (Heavy focus on nucleus r < 2)
        n_near = int(batch_size * 0.7)
        n_far  = batch_size - n_near
        
        pts_near = tf.random.normal((n_near, 3), 0, 0.8) # Gaussian cluster
        pts_far  = tf.random.uniform((n_far, 3), -L, L)
        inputs = tf.concat([pts_near, pts_far], axis=0)

        with tf.GradientTape() as tape:
            psi, V_eff, lap_psi, lap_V = get_derivatives(model, inputs)
            
            # Physics Constants
            x, y, z = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3]
            r = tf.sqrt(x**2 + y**2 + z**2 + 1e-9)
            V_ext = -2.0 / r # Helium Z=2
            
            # 1. Poisson Residual: Del^2 V = -4pi * rho
            rho = psi**2
            res_poisson = lap_V + (4.0 * np.pi * rho)
            
            # 2. Fock Residual: (-0.5 Del^2 + V_total) psi = E * psi
            # Note: We use the manually updated current_E here
            H_psi = -0.5 * lap_psi + (V_ext + V_eff) * psi
            res_fock = H_psi - (current_E * psi)
            
            # 3. Normalization Loss (Integral psi^2 = 1)
            # Monte Carlo Integral
            integral_psi2 = tf.reduce_mean(psi**2) * domain_vol
            loss_norm = (integral_psi2 - 1.0)**2
            
            # Combine
            loss = tf.reduce_mean(tf.square(res_fock)) + \
                   tf.reduce_mean(tf.square(res_poisson)) + \
                   10.0 * loss_norm

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # --- KEY FIX: RAYLEIGH QUOTIENT UPDATE ---
        # Calculate E = <psi|H|psi> / <psi|psi>
        # We do this outside the gradient tape (no backprop through E)
        if epoch % 10 == 0:
            # Re-evaluate H_psi and psi on the batch
            numerator = tf.reduce_mean(psi * H_psi) * domain_vol
            denominator = tf.reduce_mean(psi * psi) * domain_vol
            
            # Update E (moving average for stability)
            new_E = numerator / (denominator + 1e-6)
            current_E = 0.9 * current_E + 0.1 * new_E
            model.E_history.append(current_E)

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.numpy():.4f} | Energy: {current_E:.4f}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy Convergence
    ax[0].plot(model.E_history)
    ax[0].axhline(y=-2.8617, color='r', linestyle='--', label='RHF Limit (-2.86)')
    ax[0].set_title("Energy Convergence")
    ax[0].set_xlabel("Iterations (x10)")
    ax[0].set_ylabel("Energy (Hartrees)")
    ax[0].legend()
    
    # Plot 2: Radial Wavefunction
    # Generate a line of points from r=0 to r=4 along x-axis
    r_test = np.linspace(0.01, 4.0, 100)
    x_test = r_test
    y_test = np.zeros_like(r_test)
    z_test = np.zeros_like(r_test)
    pts_test = np.stack([x_test, y_test, z_test], axis=1)
    
    pred_psi, _ = model(pts_test)
    
    ax[1].plot(r_test, pred_psi, label='PINN psi(r)')
    # Compare with rough analytical approx for He: psi ~ e^(-1.69 r)
    ax[1].plot(r_test, np.exp(-1.6875 * r_test) * (pred_psi[0]/1.0), 'g--', label='Sto-1G approx')
    ax[1].set_title("1s Orbital Shape")
    ax[1].set_xlabel("Distance r (Bohr)")
    ax[1].legend()
    
    plt.show()

if __name__ == "__main__":
    train_and_plot()