import pandas as pd
import numpy as np
import json

def process_eeg_data(csv_path: str) -> dict:
    """
    Deterministically processes raw EEG data to extract mathematical features.
    This prevents the LLM from hallucinating patterns in the raw noise.
    """
    df = pd.read_csv(csv_path)
    timestamps = df["timestamp"].values
    fs = 1.0 / (timestamps[1] - timestamps[0]) # Sampling frequency
    
    channels = [col for col in df.columns if col != "timestamp"]
    
    features = {}
    
    for ch in channels:
        signal = df[ch].values
        
        # 1. Peak Amplitude (max absolute voltage)
        peak_amp = np.max(np.abs(signal))
        
        # 2. Latency of Peak (time in seconds)
        peak_idx = np.argmax(np.abs(signal))
        latency = timestamps[peak_idx]
        
        # 3. Dominant Frequency using FFT
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
        
        # Ignore DC component (0Hz)
        fft_vals[0] = 0
        dom_freq = fft_freqs[np.argmax(np.abs(fft_vals))]
        
        # Categorize frequency band
        band = "Unknown"
        if 8 <= dom_freq < 13:
            band = "Alpha (Relaxation)"
        elif 13 <= dom_freq < 30:
            band = "Beta (Active Concentration/Motor)"
        elif 30 <= dom_freq < 100:
            band = "Gamma (High-level Processing)"
        elif 4 <= dom_freq < 8:
            band = "Theta (Drowsiness)"
        elif 0.5 <= dom_freq < 4:
            band = "Delta (Deep Sleep)"
            
        features[ch] = {
            "peak_amplitude_uV": round(float(peak_amp), 2),
            "peak_latency_sec": round(float(latency), 3),
            "dominant_frequency_Hz": round(float(dom_freq), 2),
            "frequency_band": band
        }
        
    return {
        "file": csv_path,
        "duration_sec": round(float(timestamps[-1]), 2),
        "sampling_rate_Hz": round(float(fs), 1),
        "channel_features": features
    }

if __name__ == "__main__":
    import mock_data
    mock_data.generate_mock_eeg()
    res = process_eeg_data("data/simulated_eeg.csv")
    print(json.dumps(res, indent=2))
