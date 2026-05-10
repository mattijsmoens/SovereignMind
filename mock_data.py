import pandas as pd
import numpy as np
import os

def generate_mock_eeg():
    os.makedirs("data", exist_ok=True)
    
    # Simulate 2 seconds of EEG data at 250Hz sampling rate
    fs = 250
    t = np.linspace(0, 2, 2 * fs)
    
    # Background noise for channels
    channels = ["F3", "F4", "C3", "C4", "P3", "P4"]
    data = {ch: np.random.normal(0, 5, len(t)) for ch in channels}
    
    # Inject a specific "Beta wave burst" (15-30Hz) into C3 (Motor cortex, right hand)
    # This represents a cognitive intention to move the right hand
    beta_freq = 22 # Hz
    burst_start = int(0.5 * fs)
    burst_end = int(1.2 * fs)
    burst_signal = 15.0 * np.sin(2 * np.pi * beta_freq * t[burst_start:burst_end])
    
    data["C3"][burst_start:burst_end] += burst_signal
    
    df = pd.DataFrame({"timestamp": t, **data})
    df.to_csv("data/simulated_eeg.csv", index=False)
    print("[MockData] Generated simulated_eeg.csv with a 22Hz Beta burst in channel C3.")

if __name__ == "__main__":
    generate_mock_eeg()
