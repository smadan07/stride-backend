from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import math
import numpy as np
import datetime
import tensorflow as tf
import requests

IP_CACHE = {}

# --- S.T.R.I.D.E. 4.0: CNN Autoencoder Upgrade ---
# Architecture: Zero-Trust Continuous Authentication Platform
# Powered by a 1D Convolutional Neural Network processing time-series tensors.

app = FastAPI(title="S.T.R.I.D.E. 4.0 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables loaded via try/except loop
try:
    print("Loading stride_cnn.keras and stride_config.pkl...")
    GLOBAL_MODEL = tf.keras.models.load_model("stride_cnn.keras")
    
    config = joblib.load("stride_config.pkl")
    GLOBAL_SCALER = config['scaler']
    ANOMALY_THRESHOLD = config['anomaly_threshold']
    print(f"Deep Engine Online. Anomaly Limit: {ANOMALY_THRESHOLD:.4f}")
except Exception as e:
    print(f"CRITICAL WARNING: Base model failed to load. Ensure train.py has run. Error: {e}")
    GLOBAL_MODEL = None
    GLOBAL_SCALER = None
    ANOMALY_THRESHOLD = 999.0

# Global sessions database
sessions_db = {}

# --- DTOs ---
from typing import Optional

class TelemetryData(BaseModel):
    session_id: str = "default_session"
    flight_times: Optional[list[float]] = []
    hold_times: Optional[list[float]] = []
    mouse_trajectory: Optional[list[float]] = []
    error_rates: Optional[list[float]] = []
    mouse_acceleration: Optional[list[float]] = []
    context_switch_latency: Optional[list[float]] = []
    screen_width: Optional[int] = 1920
    hardware_concurrency: Optional[int] = 2
    gpu_hash: Optional[str] = "unknown"
    override_ip: Optional[str] = None

class Invariants(BaseModel):
    screen_width: Optional[int] = 1920
    hardware_concurrency: Optional[int] = 2
    gpu_hash: Optional[str] = "unknown"

    def __eq__(self, other):
        return (self.screen_width == other.screen_width and
                self.hardware_concurrency == other.hardware_concurrency and
                self.gpu_hash == other.gpu_hash)

# --- UTILS ---
def get_mock_coordinates(ip: str):
    import hashlib
    h = hashlib.md5(ip.encode()).hexdigest()
    lat = -90 + (int(h[:8], 16) / 0xffffffff) * 180
    lon = -180 + (int(h[8:16], 16) / 0xffffffff) * 360
    if ip == "127.0.0.1" or ip.startswith("10.") or ip.startswith("192.168."): return (40.7128, -74.0060) # NY
    if ip == "8.8.8.8": return (51.5074, -0.1278) # London (VPN Scenario)
    if ip == "4.4.4.4": return (35.6762, 139.6503) # Tokyo (Teleport Scenario)
    return (lat, lon)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def is_vpn(ip: str):
    return ip == "8.8.8.8"

# 1. The Setup & Cache
def get_ip_reputation(ip_address: str) -> dict:
    # Whitelist localhost to prevent pointless loopback lookups
    if ip_address in ("127.0.0.1", "localhost", "0.0.0.0") or ip_address.startswith("192.") or ip_address.startswith("10."):
        return {"isp": "Local Network", "proxy": False, "hosting": False}
        
    # The In-Memory Cache (0ms latency)
    if ip_address in IP_CACHE:
        return IP_CACHE[ip_address]
        
    try:
        url = f"http://ip-api.com/json/{ip_address}?fields=isp,proxy,hosting"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            context = {
                "isp": data.get("isp", "Unknown ISP"),
                "proxy": data.get("proxy", False),
                "hosting": data.get("hosting", False)
            }
            IP_CACHE[ip_address] = context
            return context
    except Exception:
        # Fails open: If the API crashes during the hackathon demo, we simply ignore it rather than killing the server.
        pass
        
    return {"isp": "Unknown", "proxy": False, "hosting": False}

# --- CORE ENDPOINTS ---

@app.get("/")
async def serve_demo_ui():
    return FileResponse("index.html")

@app.get("/sensor.js")
async def serve_stealth_agent():
    return FileResponse("sensor.js")

@app.post("/telemetry")
async def process_telemetry(payload: TelemetryData, request: Request):
    if GLOBAL_MODEL is None or GLOBAL_SCALER is None:
        return {"error": "CNN model offline - run train.py"}

    sid = payload.session_id
    current_ip = payload.override_ip if payload.override_ip else request.client.host
    now = datetime.datetime.now()
    
    current_invariants = Invariants(
        screen_width=payload.screen_width,
        hardware_concurrency=payload.hardware_concurrency,
        gpu_hash=payload.gpu_hash
    )
    
    # 1. State Management - Initialization
    if sid not in sessions_db:
        # Clone the Deep Learning CNN for absolute Zero-Trust Isolation per session
        personal_cnn = tf.keras.models.clone_model(GLOBAL_MODEL)
        personal_cnn.set_weights(GLOBAL_MODEL.get_weights())
        # Re-compile to allow online fine-tuning
        personal_cnn.compile(optimizer='adam', loss='mse')

        sessions_db[sid] = {
            "state": "active",
            "personal_model": personal_cnn,
            "rolling_window": [], # sequential queue of the last 15 seconds
            "last_ip": current_ip,
            "last_timestamp": now,
            "calibration_samples": 0,
            "personal_threshold": ANOMALY_THRESHOLD,
            "baseline_invariants": current_invariants,
            "risk_score": 0.0,
            "xai_reasoning": "Deep Learning sequence array initialized."
        }
        
    sess = sessions_db[sid]
    
    # The Sandbox Latch Fast Reject
    if sess["state"] == "sandboxed":
        sess["risk_score"] = 100.0
        return {"status": "sandboxed", "risk_score": 100.0, "xai_reasoning": "Session permanently locked by Neural Net."}
        
    # Variables Extraction for Time Series
    f = np.mean(payload.flight_times) if payload.flight_times else 0.0
    h = np.mean(payload.hold_times) if payload.hold_times else 0.0
    m = sum(payload.mouse_trajectory) if payload.mouse_trajectory else 0.0
    er = np.mean(payload.error_rates) if payload.error_rates else 0.0
    ma = np.mean(payload.mouse_acceleration) if payload.mouse_acceleration else 0.0
    csl = np.mean(payload.context_switch_latency) if payload.context_switch_latency else 0.0
    
    # 2. Structural Invariant Verification (WebGL + Hardware)
    if current_invariants != sess["baseline_invariants"]:
        # The user's GPU hardware signature or screen dimensions suddenly changed mid-session.
        # This is mathematically impossible unless a hacker stole the session cookie on another machine!
        sess["risk_score"] += 100.0
        sess["xai_reasoning"] = "CRITICAL: WebGL GPU Hardware Fingerprint Mismatch. Confirmed Session Hijack."
        sess["baseline_invariants"] = current_invariants # Prevent infinitely spamming the score
        
    # 3. Geovelocity Engine & Location Anomalies
    time_diff_hours = (now - sess["last_timestamp"]).total_seconds() / 3600.0
    if time_diff_hours > 0 and current_ip != sess["last_ip"]:
        
        # Datacenter Trap / ASN Reputation Matrix
        ip_context = get_ip_reputation(current_ip)
        if ip_context["hosting"] or ip_context["proxy"]:
            sess["risk_score"] += 100.0
            sess["xai_reasoning"] = f"CRITICAL ASN Anomaly: Traffic routed through Commercial Datacenter/Proxy ({ip_context['isp']})."
        else:
            lat1, lon1 = get_mock_coordinates(sess["last_ip"])
            lat2, lon2 = get_mock_coordinates(current_ip)
            speed = haversine(lat1, lon1, lat2, lon2) / time_diff_hours
            
            if speed > 1000:
                vpn_flag = is_vpn(current_ip)
                if not vpn_flag:
                    sess["risk_score"] += 100.0
                    sess["xai_reasoning"] = f"CRITICAL Geovelocity anomaly. Impossible Travel Speed: {speed:.0f}km/h."
                
    sess["last_ip"] = current_ip
    sess["last_timestamp"] = now
    
    # 3. 1D-CNN Encoder/Decoder Inference
    if f > 0.0 or er > 0.0 or csl > 0.0:
        if m == 0.0: m = 450.0  # Allow pure typing without artificial mouse penalties
        if csl == 0.0: csl = 300.0 # Baseline typical non-moving value
        
        sess["rolling_window"].append([f, h, m, er, ma, csl])
        
        if len(sess["rolling_window"]) >= 15:
            # We have enough sequence memory to evaluate the CNN!
            window = np.array(sess["rolling_window"][-15:])
            
            # The StandardScaler was fit on flat 2D data (15, 6), so we can directly transform it
            scaled_window = GLOBAL_SCALER.transform(window)
            
            # Reshape into Keras 3D sequence: (batch, timesteps, features)
            nn_input = scaled_window.reshape(1, 15, 6)
            
            # Autoencoder Reconstruction 
            # In latent compression, if the rhythm is unrecognized by the filters, 
            # it will spectacularly fail to rebuild the output sequence.
            reconstructed = sess["personal_model"].predict(nn_input, verbose=0)
            
            # Calculate Reconstruction Error (MSE) across the whole sequence
            mse = np.mean(np.square(nn_input - reconstructed))
            
            is_calibrating = sess.get("calibration_samples", 0) < 5
            
            if is_calibrating:
                # OVERFITTING FIX: Training a Neural Net for 10 epochs on a single tensor 
                # instantly collapses its variance, causing it to reject minor natural 
                # human fluctuations as massive anomalies. Dropping to epochs=2 lets it bend gracefully.
                sess["personal_model"].fit(nn_input, nn_input, epochs=2, verbose=0)
                sess["calibration_samples"] += 1
                
                # THRESHOLD FIX: Increase your variance elasticity to 150% above your maximum 
                # calibration noise level. This is the "Goldilocks Zone" between 
                # Catastrophic Overfitting (1.0x) and Loose Security (3.0x).
                if mse > sess["personal_threshold"]:
                    sess["personal_threshold"] = max(sess["personal_threshold"], mse * 1.5)
                    
                sess["risk_score"] = 0.0
                sess["xai_reasoning"] = f"Calibrating Neural Net Weights... ({5 - sess['calibration_samples']} cycles remain)"
                sess["rolling_window"].pop(0)
            else:
                pt = sess["personal_threshold"]
                if mse > pt:
                    # Anomaly Detected! Use aggressive proportional penalty to instantly flag erratic typing 
                    # while maintaining the proportional protection for minor false positives.
                    drift = abs(mse - pt)
                    penalty = min(50.0, (drift * 25.0) + 12.0)
                    sess["risk_score"] += penalty
                    sess["xai_reasoning"] = f"CNN Reconstruction Error ({mse:.2f} > {pt:.2f}). (+{penalty:.1f} Risk)"
                    sess["rolling_window"].pop(0) # slide boundary
                else:
                    # Recover points instantly
                    sess["risk_score"] = max(0.0, sess["risk_score"] - 6.0)
                    
                    # Security: Defeat 'Frog in Boiling Water' Data Poisoning
                    if mse < (pt * 0.5):
                        # Online learning for Deep Neural Network!
                        sess["personal_model"].fit(nn_input, nn_input, epochs=1, verbose=0)
                        if sess["risk_score"] < 5.0:
                            sess["xai_reasoning"] = f"Rhythm verified. Continuous auth active. (MSE: {mse:.2f})"
                        else:
                            sess["xai_reasoning"] = f"Borderline syntax. CNN memory locked against Poisoning. (MSE: {mse:.2f})"
                    else:
                        sess["xai_reasoning"] = f"Borderline syntax. CNN memory locked against Poisoning. (MSE: {mse:.2f})"
                    
                    # Slide the sliding window
                    sess["rolling_window"].pop(0)
        else:
            # We are currently loading the 15-tick sequence buffer
            sess["xai_reasoning"] = f"Buffering CNN Sequential Latency Array... [{len(sess['rolling_window'])}/15 frames]"
            
    # 4. The Sandbox Latch
    risk = sess["risk_score"]
    
    if risk < 36:
        sess["state"] = "seamless"
    elif risk < 70:
        sess["state"] = "mfa_challenge"
    else:
        sess["state"] = "sandboxed"
        sess["risk_score"] = 100.0 
        
    return {
        "status": sess["state"],
        "risk_score": sess["risk_score"],
        "xai_reasoning": sess["xai_reasoning"]
    }

@app.get("/risk-status/{session_id}")
async def fetch_risk_status(session_id: str):
    if session_id not in sessions_db:
        return {"error": "Session Not Found", "status": "unknown"}
    sess = sessions_db[session_id]
    return {
        "status": sess["state"],
        "risk_score": sess["risk_score"],
        "xai_reasoning": sess["xai_reasoning"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
