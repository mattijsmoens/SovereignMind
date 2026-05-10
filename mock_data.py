import json
import random
import requests
import urllib.request
import csv
import time

def generate_mock_telemetry():
    """Generates realistic autonomic biofeedback telemetry."""
    return {
        "pupil_dilation_mm": round(random.uniform(2.0, 8.0), 2),
        "heart_rate_bpm": random.randint(55, 120),
        "hrv_ms": round(random.uniform(20.0, 100.0), 2)
    }

def fetch_real_fmri_vector():
    """
    Downloads a public fMRI dataset and extracts a real BOLD biological vector.
    Extracts time-series data for subject 's0', region 'parietal', event 'stim'.
    """
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/fmri.csv"
    print(f"[*] Downloading real fMRI dataset from {url}...")
    
    response = urllib.request.urlopen(url)
    lines = [l.decode('utf-8') for l in response.readlines()]
    reader = csv.DictReader(lines)
    
    # We want to extract a clean time-series vector for a specific subject and region
    vector_data = []
    
    # Parse the CSV rows
    for row in reader:
        if row['subject'] == 's0' and row['region'] == 'parietal' and row['event'] == 'stim':
            # The CSV has multiple timepoints, we grab the signal float
            vector_data.append((int(row['timepoint']), float(row['signal'])))
            
    # Sort by timepoint to ensure the vector is sequential
    vector_data.sort(key=lambda x: x[0])
    
    # Extract just the signal values into our final array
    final_vector = [val[1] for val in vector_data]
    
    print(f"[*] Successfully extracted {len(final_vector)}-point biological vector.")
    return final_vector

def run_mock_pipeline():
    # 1. Fetch REAL fMRI data
    real_fmri_vector = fetch_real_fmri_vector()
    
    with open("real_fmri_vector.json", "w") as f:
        json.dump(real_fmri_vector, f, indent=4)
        
    print("\n[Simulator] Pushing REAL Biological fMRI vector to SovereignMind API...")
    payload = {"fmri_vector": real_fmri_vector}
    
    try:
        res = requests.post("http://localhost:8007/api/v1/decode_semantic_signals", json=payload)
        print(f"API Response: {res.status_code}")
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    run_mock_pipeline()
