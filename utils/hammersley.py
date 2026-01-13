def get_hammersley_sequence(n_samples, scramble=True):
    """
    Generates a 2D Hammersley sequence of N points in [0, 1]^2.
    
    Args:
        n_samples (int): Number of points to generate.
        scramble (bool): If True, applies a random shift (Cranley-Patterson rotation).
                         This is CRITICAL for training loops to avoid using 
                         the exact same fixed grid every epoch.
    """
    # 1. Generate First Dimension (Linear i/N)
    # We create indices 1 to N
    idx = np.arange(1, n_samples + 1)
    x_dim = (idx - 0.5) / n_samples  # Shifted to centers
    
    # 2. Generate Second Dimension (Van der Corput base 2)
    # Efficient bit-reversal logic for base 2
    # We can use a simple loop or bitwise operations. 
    # For < 100k points, a loop is fast enough and readable.
    y_dim = []
    for i in range(n_samples):
        n = i + 1 # Use 1-based index
        q = 0.
        bk = 0.5 # 1/base
        while n > 0:
            q += (n % 2) * bk
            n //= 2
            bk /= 2
        y_dim.append(q)
    y_dim = np.array(y_dim)
    
    # Stack them
    points = np.stack([x_dim, y_dim], axis=1) # Shape (N, 2)
    
    # 3. Scrambling (Random Shift)
    # Without this, the points are identical every time you call the function.
    if scramble:
        shift = np.random.rand(2) # Random shift for (x, y)
        points = (points + shift) % 1.0 # Wrap around 1.0
        
    return torch.tensor(points, dtype=torch.float32)