import json
import random
import os

def generate_mock_telemetry(verified_hash: str = "abc123def456") -> dict:
    """
    Generates a simulated non-invasive biofeedback telemetry payload.
    This replaces the old simulated_eeg.csv generator.
    """
    payload = {
        "verified_hash": verified_hash,
        "pupil_dilation_mm": round(random.uniform(2.0, 8.0), 2),
        "heart_rate_bpm": random.randint(60, 100),
        "hrv_ms": round(random.uniform(20.0, 80.0), 1)
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/mock_telemetry.json", "w") as f:
        json.dump(payload, f, indent=2)
        
    print(f"[MockData] Generated mock telemetry payload: data/mock_telemetry.json")
    return payload

def generate_mock_fmri_vector(target_sum: float = 60.0) -> list:
    """
    Generates a simulated non-invasive fMRI latent vector.
    """
    vector = [random.uniform(0.1, 5.0) for _ in range(20)]
    # adjust vector to meet target sum for deterministic LLM testing
    current_sum = sum(vector)
    adjustment = target_sum / current_sum
    vector = [round(v * adjustment, 2) for v in vector]
    
    os.makedirs("data", exist_ok=True)
    with open("data/mock_fmri_vector.json", "w") as f:
        json.dump({"fmri_vector": vector}, f, indent=2)
        
    print(f"[MockData] Generated mock fMRI vector payload (sum={sum(vector):.2f}): data/mock_fmri_vector.json")
    return vector

if __name__ == "__main__":
    generate_mock_telemetry()
    generate_mock_fmri_vector()
