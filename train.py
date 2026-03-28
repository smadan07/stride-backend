import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Cropping1D
import joblib
import os

print("Booting Keras Deep Learning Compiler for S.T.R.I.D.E. 4.0...")

# 1. Load the 3D Tensor array
filename = "stride_sequences.npy"
print(f"Loading {filename}...")
try:
    X_3D = np.load(filename)
except FileNotFoundError:
    print(f"Error: {filename} not found. Please run generate_data.py first.")
    exit(1)

num_samples, seq_length, num_features = X_3D.shape

# 2. Reshape and Normalize the data
print("Normalizing 3D Tensor data with StandardScaler...")
# Scikit-learn requires 2D arrays, so we flatten the timesteps to scale dynamically
X_2D = X_3D.reshape(-1, num_features)
scaler = StandardScaler()
X_2D_scaled = scaler.fit_transform(X_2D)
# Mathematically reconstruct the 3D sequence: (samples, timesteps, features)
X_3D_scaled = X_2D_scaled.reshape(num_samples, seq_length, num_features)

# 3. Build the 1D Convolutional Neural Network (Autoencoder)
print("Constructing 1D-CNN Autoencoder topology...")
model = Sequential()

# --- THE ENCODER (Compresses sequential rhythms into latent space) ---
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=(seq_length, num_features)))
model.add(MaxPooling1D(pool_size=2, padding='same')) # Down-samples length 15 -> 8

# --- THE DECODER (Attempts to reconstruct the original rhythm) ---
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling1D(size=2)) # Up-samples length 8 -> 16
model.add(Conv1D(filters=num_features, kernel_size=3, activation='linear', padding='same'))

# --- SHAPE CORRECTION ---
# Because upsampling 8 -> 16 overshoots our original length of 15, we crop 1 timestep from the end.
model.add(Cropping1D(cropping=(0, 1)))

# Compile the Neural Network targeting Mean Squared Error (MSE)
model.compile(optimizer='adam', loss='mse')

# 4. Train the Model (Input and Target are identical in Autoencoders)
print("Training CNN Autoencoder. This defines the robust generic continuous baseline.")
model.fit(X_3D_scaled, X_3D_scaled, epochs=10, batch_size=32, validation_split=0.1)

# 5. Extract the Latent Reconstruction Error to define the Anomaly Limit
print("Calculating Reconstruction Error Thresholds...")
X_pred = model.predict(X_3D_scaled)

# MSE Calculation per sample across timesteps and features
mse_scores = np.mean(np.square(X_3D_scaled - X_pred), axis=(1, 2))
# The 95th percentile marks the absolute limit of acceptable rhythm variation (false positives)
anomaly_threshold = np.percentile(mse_scores, 95)
print(f"Calculated 95th Percentile ANOMALY_THRESHOLD: {anomaly_threshold:.4f}")

# 6. Export all artifacts 
print("Exporting deep learning artifacts...")
model.save("stride_cnn.keras")
joblib.dump({"scaler": scaler, "anomaly_threshold": anomaly_threshold}, "stride_config.pkl")

print("Deep Learning Compilation Complete. S.T.R.I.D.E 4.0 Backend is ready.")
