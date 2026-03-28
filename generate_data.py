import numpy as np
import os

# Set random seed for reproducibility in hackathon demo
np.random.seed(42)

num_samples = 5000
sequence_length = 15
num_features = 6

print("Generating 6D Autoencoder time-series telemetry for S.T.R.I.D.E. 4.0...")

# Initialize the 3D tensor: (samples, timesteps, features)
data = np.zeros((num_samples, sequence_length, num_features))

for i in range(num_samples):
    # Establish a unique baseline anchor for this specific synthetic "session"
    base_flight = np.random.normal(200.0, 20.0)
    base_hold = np.random.normal(95.0, 8.0)
    base_mouse = np.random.normal(450.0, 150.0)
    base_error = np.random.normal(2.0, 1.0)
    base_mouse_accel = np.random.normal(5.0, 2.0)
    base_context_switch = np.random.normal(300.0, 50.0)
    
    # Introduce Sine-wave fluctuation logic to simulate organic human rhythm
    # People don't type like robots; they accelerate and decelerate in bursts.
    phase = np.random.uniform(0, 2 * np.pi)
    freq = np.random.uniform(0.5, 2.0)
    
    for t in range(sequence_length):
        # The Sine wave modifies the base metrics across the 15 memory timesteps
        rhythm_mod = np.sin(freq * t + phase)
        
        f = base_flight + (rhythm_mod * 15.0) + np.random.normal(0, 5.0)
        h = base_hold + (rhythm_mod * 5.0) + np.random.normal(0, 2.0)
        m = base_mouse + (rhythm_mod * 50.0) + np.random.normal(0, 20.0)
        er = base_error + (rhythm_mod * 1.0) + np.random.normal(0, 0.5)
        ma = base_mouse_accel + (rhythm_mod * 2.0) + np.random.normal(0, 1.0)
        csl = base_context_switch + (rhythm_mod * 20.0) + np.random.normal(0, 10.0)
        
        data[i, t, 0] = np.clip(f, 50.0, 500.0)
        data[i, t, 1] = np.clip(h, 30.0, 200.0)
        data[i, t, 2] = np.clip(m, 0.0, 2000.0)
        data[i, t, 3] = np.clip(er, 0.0, 20.0)
        data[i, t, 4] = np.clip(ma, -500.0, 500.0)
        data[i, t, 5] = np.clip(csl, 50.0, 5000.0)

np.save("stride_sequences.npy", data)
print(f"Successfully generated 3D tensor {data.shape} and exported to stride_sequences.npy.")
