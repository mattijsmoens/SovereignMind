# SovereignMind: The BCI Research Firewall

SovereignMind is a conceptual BCI research platform that proves how **Sovereign Shield N-Model Consensus** can accelerate neurotechnology by eliminating AI hallucinations during neural data analysis.

Instead of trusting a single language model to decode chaotic brain waves, SovereignMind uses a two-step deterministic + consensus pipeline:

1. **Deterministic Signal Processing:** Raw EEG data is processed strictly using `numpy` and `pandas` to extract mathematical features (peak amplitude, latency, frequency bands).
2. **N-Model Verification:** The deterministic features are passed to a 3-model Sovereign Shield panel (GPT-4o, Gemini, Llama 3). The models must achieve perfect cryptographic consensus on what the "Cognitive Intention" is (e.g., "Motor Cortex: Right Hand Movement").
3. **Ground Truth:** Only verified neural mappings are stored in the SQLite dictionary.

## Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file and insert your OpenRouter API key:
```
OPENROUTER_API_KEY="sk-or-v1-..."
```

## Usage

1. **Generate Mock Data:** Run `python mock_data.py` to create a simulated EEG file with a 22Hz Beta burst in the C3 (motor) channel.
2. **Start the Engine:** Run `python server.py`.
3. **Test the Pipeline:** Go to `http://localhost:8000/docs` and use the `POST /analyze_eeg` endpoint with the payload `{"csv_path": "data/simulated_eeg.csv"}`.
