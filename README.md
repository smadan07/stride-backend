# 🛡️ S.T.R.I.D.E. 
**Continuous Behavioral Biometric Authentication Engine**

[![Hackathon Submission](https://img.shields.io/badge/Status-MVP_Submitted-success)]()
[![Architecture](https://img.shields.io/badge/Architecture-Decoupled-blue)]()

## 🚨 The Problem: The "Front Door" Fallacy
Traditional cybersecurity assumes that if a user possesses the correct session cookie or password, they are the authorized user. This "front door" authentication ignores what happens *after* login. If a laptop is left unlocked, or a session token is hijacked, the system blindly trusts the attacker. 

## 💡 The Solution: S.T.R.I.D.E.
S.T.R.I.D.E. rebuilds identity verification using a **Zero-Trust Triad**. We don't just verify what you know; we continuously and invisibly verify **who you are, where you are, and what hardware you are using.**

### 🔑 The Zero-Trust Triad (Core Features)

1. **The Human (Behavioral Biometrics via CNN)**
   - Our native `sensor.js` engine silently captures 6 real-time features: *Keystroke Flight Time, Dwell Time, Mouse Velocity, Trajectory Curvature, Click Cadence, and Error Rates.*
   - **The Brain:** We pass this rolling time-series window into a **Convolutional Neural Network (CNN)**. By reshaping the data into a 2D matrix, the CNN's convolutional layers extract the hidden rhythms and spatial-temporal relationships of the user's movements, flagging anomalous behavior (like an attacker typing) in milliseconds.

2. **The Network (Haversine Contextual Gating)**
   - Before the heavy ML model even runs, we extract the user's IP.
   - We utilize the **Haversine formula** to calculate the exact great-circle geographic distance between the current and previous login. By measuring distance over time, we detect "Impossible Travel" (e.g., traveling 5,000 mph), instantly killing hijacked sessions originating from foreign servers.

3. **The Silicon (WebGL GPU Fingerprinting)**
   - We extract a cryptographic rendering vector using a hidden WebGL graphic. 
   - Because different GPUs and OS combinations calculate floating-point math differently, we can cryptographically verify the physical machine. If an attacker steals a token and uses it on their own laptop, the hardware mismatch triggers an instant lockout.

---

## 🏗️ Architecture & The Vercel Pivot

**Tech Stack:** `HTML/JS/WebGL` (Frontend) | `FastAPI / Python` (Backend) | `Scikit-Learn / TensorFlow` (ML)

### ⚠️ The Hackathon Hardware Challenge
We initially designed our Python Machine Learning backend to run as a serverless function on Vercel. However, loading our comprehensive CNN anomaly detection model (`.pkl`) exceeded Vercel's strict 512MB memory limit, resulting in repeated Out-Of-Memory (OOM) crashes.

**The Pivot:** Rather than dumbing down our security model or stripping features to fit a free-tier constraint, we re-architected the system mid-hackathon. 
- **Frontend:** Deployed statically via GitHub Pages for 60 FPS WebGL rendering and zero-latency event capturing.
- **Backend/Inference Node:** Decoupled into a dedicated FastAPI service capable of handling the heavy matrix math required by the CNN.

*(Note: For the purpose of the live video demo, the backend inference node is running on local hardware to bypass free-tier memory constraints, demonstrating the exact production logic without compromise).*

---

## 🚀 Running the Project Locally (Judges / Testing)

Because of the decoupled architecture, you can run the inference engine locally to test the behavioral model.

### 1. Boot the ML Backend
```bash
# Navigate to the backend directory
cd stride-backend

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI inference node
uvicorn main:app --reload
# The backend will start on [http://127.0.0.1:8000](http://127.0.0.1:8000)
