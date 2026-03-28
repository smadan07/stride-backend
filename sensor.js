/* 
 * sensor.js - The Stealth Telemetry Agent
 * Architecture: S.T.R.I.D.E. 3.0 (Swift Trust & Real-time Identity Dynamics Engine)
 * 
 * This agent collects background behavioral biometrics (flight times, hold times,
 * and mouse trajectory lengths) and ships them to the Inference Engine every 1000ms.
 */

// Core buffering components
let keydownTimes = {};
let flightTimes = [];
let holdTimes = [];
let lastKeyupTime = 0;

let lastMouseX = null;
let lastMouseY = null;
let mouseTrajectories = [];

// New S.T.R.I.D.E 4.0 Dimension Tracking
let windowErrorCount = 0;
let errorRates = [];

let lastMouseVelocity = 0;
let lastMouseTime = null;
let mouseAcceleration = [];

let contextSwitchLatency = [];
let expectingContextSwitch = false;
let lastTypingTime = 0;

// 1. Privacy Filters & Flight/Hold Calculation
document.addEventListener('keydown', (e) => {
    // Track Errors (Backspace / Delete)
    if (e.key === "Backspace" || e.key === "Delete") {
        windowErrorCount++;
    }

    // Privacy Filter: Ignore keys being held down (OS repeat spam) and non-character keys (e.g. Shift, Ctrl)
    if (e.repeat || e.key.length !== 1) return;
    
    const now = Date.now();
    keydownTimes[e.key] = now;
    
    // Flight time: duration between last keyup and current keydown
    if (lastKeyupTime !== 0) {
        let t = now - lastKeyupTime;
        // Ignore massive pauses between words so they don't corrupt the biometric average
        if (t < 1500) {
            flightTimes.push(t);
        }
    }
});

document.addEventListener('keyup', (e) => {
    // Maintain filter consistency
    if (e.key.length !== 1) return;
    
    const now = Date.now();
    lastKeyupTime = now;
    lastTypingTime = now;
    expectingContextSwitch = true;
    
    // Hold time: duration between keydown and keyup for this specific key
    if (keydownTimes[e.key]) {
        holdTimes.push(now - keydownTimes[e.key]);
        delete keydownTimes[e.key];
    }
});

// 2. Mouse Trajectory Tracker & Acceleration (Euclidean Distance & Velocity Delta)
document.addEventListener('mousemove', (e) => {
    const now = Date.now();
    
    // Context Switch Latency calculation
    if (expectingContextSwitch && lastTypingTime > 0) {
        let latency = now - lastTypingTime;
        if (latency < 5000) { // Filter out purely idle AFK pauses vs natural action-switches
            contextSwitchLatency.push(latency);
        }
        expectingContextSwitch = false; 
    }

    if (lastMouseX !== null && lastMouseY !== null && lastMouseTime !== null) {
        let dx = e.clientX - lastMouseX;
        let dy = e.clientY - lastMouseY;
        let dist = Math.sqrt(dx * dx + dy * dy);
        mouseTrajectories.push(dist);
        
        // Acceleration calculation (Delta V / Delta T)
        let timeDiff = now - lastMouseTime;
        if (timeDiff > 0) {
            let velocity = dist / timeDiff; // pixels per millisecond
            let accel = velocity - lastMouseVelocity; // change in velocity
            mouseAcceleration.push(accel);
            lastMouseVelocity = velocity;
        }
    }
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
    lastMouseTime = now;
});

// 3. Hardware Invariants Capture
const screenWidth = window.screen.width;
const hardwareConcurrency = navigator.hardwareConcurrency || 2;

// --- ZERO-TRUST FINGERPRINTING ---
// Fast, synchronous 53-bit string hasher (cyrb53)
const cyrb53 = (str, seed = 0) => {
    let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
    for(let i = 0, ch; i < str.length; i++) {
        ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1  = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
    h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2  = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
    h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return "gpu_" + (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16);
};

// The WebGL Hash exploits distinct GPU rendering hardware logic
// (anti-aliasing, color interpolation logic) to extract a mathematical signature.
let cachedGpuHash = null;

function generateWebGLFingerprint() {
    if (cachedGpuHash) return cachedGpuHash;
    
    try {
        const canvas = document.createElement("canvas");
        const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        if (!gl) return "gpu_unsupported";
        
        // Basic Vertex Shader
        const vShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vShader, "attribute vec2 attrVertex;varying vec2 varyinTexCoordinate;uniform vec2 uniformOffset;void main(){varyinTexCoordinate=attrVertex+uniformOffset;gl_Position=vec4(attrVertex,0,1);}");
        gl.compileShader(vShader);
        
        // Basic Fragment Shader forcing precision color interpolation
        const fShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fShader, "precision mediump float;varying vec2 varyinTexCoordinate;void main() {gl_FragColor=vec4(varyinTexCoordinate,0,1);}");
        gl.compileShader(fShader);
        
        const program = gl.createProgram();
        gl.attachShader(program, vShader);
        gl.attachShader(program, fShader);
        gl.linkProgram(program);
        gl.useProgram(program);
        
        // Draw a generic polygon triangle
        gl.program = program;
        const vertexPosBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexPosBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-0.2, -0.9, 0, 0.4, -0.26, 0, 0, 0.7321, 0]), gl.STATIC_DRAW);
        
        // Tie attributes and Render
        program.vertexPosAttrib = gl.getAttribLocation(program, "attrVertex");
        gl.enableVertexAttribArray(program.vertexPosAttrib);
        gl.vertexAttribPointer(program.vertexPosAttrib, 3, gl.FLOAT, false, 0, 0);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        
        // Base64 hash translation
        cachedGpuHash = cyrb53(canvas.toDataURL());
        return cachedGpuHash;
    } catch (e) {
        return "gpu_error";
    }
}

// Provide a stable synthetic session ID for the Hackathon lifecycle
const sessionId = "sess_demo_" + Math.random().toString(36).substr(2, 9);
window.hijack_ip = null; // Console Override for Hackathon Geo Demonstrations

// Helper function to calculate mean of an array safely
function getAverage(arr) {
    if (arr.length === 0) return 0.0;
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum / arr.length;
}

// Helper function to calculate sum
function getSum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}

// 4. Real-time Inference Engine Link (1000ms Heartbeat)
setInterval(async () => {
    // POST errorCount for this window
    if (windowErrorCount > 0) {
        errorRates.push(windowErrorCount);
        windowErrorCount = 0; // Reset for next window
    } else {
        errorRates.push(0);
    }

    // 5D Architecture Upgrade: We no longer flatten arrays on the client side. 
    // We send the raw arrays to Python so the Inference Engine can calculate 
    // variance and Standard Deviation.
    const payload = {
        session_id: sessionId,
        flight_times: flightTimes,
        hold_times: holdTimes,
        mouse_trajectory: mouseTrajectories,
        error_rates: errorRates,
        mouse_acceleration: mouseAcceleration,
        context_switch_latency: contextSwitchLatency,
        screen_width: screenWidth,
        hardware_concurrency: hardwareConcurrency,
        gpu_hash: generateWebGLFingerprint(),
        override_ip: window.hijack_ip
    };

    // SMOOTHING THE NOISE: Implement an Intelligent Rolling Window
    // Instead of brutally truncating the arrays every 1 second (which creates random volatile spikes),
    // we maintain the last 30 keystrokes to evaluate a sustained behavioral trend.
    if (Date.now() - lastKeyupTime > 15000) {
        // Stop analyzing if the user pauses for more than 15 seconds (True context switch)
        // This preserves the 30-stroke memory during natural reading/thinking pauses!
        flightTimes = [];
        holdTimes = [];
        errorRates = [];
        contextSwitchLatency = [];
    } else {
        // Keep a rolling moving average
        if (flightTimes.length > 30) flightTimes = flightTimes.slice(-30);
        if (holdTimes.length > 30) holdTimes = holdTimes.slice(-30);
        if (errorRates.length > 30) errorRates = errorRates.slice(-30);
        if (contextSwitchLatency.length > 30) contextSwitchLatency = contextSwitchLatency.slice(-30);
    }

    // Velocity measures distance per *second*, so this MUST be cleared every interval
    mouseTrajectories = [];
    mouseAcceleration = [];

    // POST Payload to FastAPI
    try {
        const response = await fetch('http://localhost:8000/telemetry', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            const data = await response.json();
            updateStrideUI(data.status, data.risk_score, data.xai_reasoning);
        }
    } catch (err) {
        // Silently fail if server is down to remain 'stealth'
    }

}, 1000);

// 5. Reactive UI Sandbox Latch Visualizer
function updateStrideUI(state, riskScore = 0, reason = "") {
    // 5a. Live HUD Updates (if they exist in the DOM)
    const stateVal = document.getElementById('val-state');
    const riskVal = document.getElementById('val-risk');
    const xaiLog = document.getElementById('xai-log');
    
    if (stateVal && riskVal && xaiLog) {
        stateVal.innerText = state.toUpperCase();
        riskVal.innerText = parseFloat(riskScore).toFixed(1) + " / 100";
        
        if (state === "seamless") stateVal.style.color = "var(--accent-green)";
        if (state === "mfa_challenge") stateVal.style.color = "var(--accent-warning)";
        if (state === "sandboxed") stateVal.style.color = "var(--accent-danger)";
        
        if (reason) {
            const entry = document.createElement("div");
            entry.innerText = `> [${parseFloat(riskScore).toFixed(1)}] ${reason}`;
            entry.style.color = state === "sandboxed" ? "var(--accent-danger)" : "#a0aab5";
            xaiLog.appendChild(entry);
            xaiLog.scrollTop = xaiLog.scrollHeight;
        }
    }

    // 5b. The Irreversible ML Kill State
    if (state === "sandboxed") {
        // Only inject if it doesn't already exist
        if (!document.getElementById("stride-sandbox-lockdown")) {
            // Apply visual classes (stealthily altering the DOM)
            document.body.classList.add("stride-sandboxed");
            
            // Overriding inline for demo ensurety
            document.body.style.transition = "all 0.5s ease";
            document.body.style.filter = "grayscale(100%) blur(2px)";
            document.body.style.pointerEvents = "none"; 
            
            // Build visual lockdown modal
            const overlay = document.createElement("div");
            overlay.id = "stride-sandbox-lockdown";
            overlay.style.position = "fixed";
            overlay.style.top = "0";
            overlay.style.left = "0";
            overlay.style.width = "100vw";
            overlay.style.height = "100vh";
            overlay.style.backgroundColor = "rgba(255, 0, 0, 0.15)";
            overlay.style.border = "8px solid red";
            overlay.style.zIndex = "2147483647"; 
            overlay.style.display = "flex";
            overlay.style.alignItems = "center";
            overlay.style.justifyContent = "center";
            overlay.style.pointerEvents = "all"; // block events underneath
            
            const banner = document.createElement("div");
            banner.style.backgroundColor = "white";
            banner.style.padding = "40px";
            banner.style.border = "4px solid red";
            banner.style.textAlign = "center";
            banner.style.fontFamily = "monospace";
            
            banner.innerHTML = `
                <h1 style="color:red; font-size:48px; margin:0; text-transform:uppercase;">S.T.R.I.D.E Sandbox Latch</h1>
                <h2 style="color:black; font-size:24px;">SESSION TERMINATED</h2>
                <p style="color:black;">Irreversible behavioral drift or hijacking detected.</p>
            `;
            
            overlay.appendChild(banner);
            document.documentElement.appendChild(overlay);
        }
    }
}
