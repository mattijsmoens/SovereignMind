# SovereignMind: Neuro-Semantic BCI Translation Engine

**SovereignMind** is a revolutionary, non-invasive Brain-Computer Interface (BCI) module built on the **SovereignShield N-Model AI Consensus Architecture**. 

Unlike legacy BCI systems that rely on invasive surgical hardware (e.g., neural laces) or passive, low-resolution EEG caps, SovereignMind operates entirely through high-bandwidth natural input/output ports: **Audio, Visual, and Semantic Language pathways.**

By treating the human brain as a protected endpoint, SovereignMind utilizes cryptographic AI consensus to enforce strict safety boundaries on all generated sensory stimuli and all reconstructed neural decodings.

---

## 🔬 Research Background & Origins

The architecture of SovereignMind was conceived as a non-invasive alternative to surgical BCI solutions (like Neuralink). Rather than physically wiring conductors into the motor cortex, the system recognizes that the human brain already possesses incredibly high-bandwidth, biologically optimized input and output ports: our eyes, ears, and semantic processing centers.

The technological leap for SovereignMind is built on the convergence of two major breakthroughs in neuroscience and AI:
1. **Semantic Decoding from Non-Invasive Scans**: Inspired by the 2023 University of Texas at Austin research on using LLMs to reconstruct continuous language from fMRI BOLD signals, and Meta's recent work on decoding visual imagery from MEG signals in real-time. SovereignMind uses this exact principle to "read" from the brain without breaking the skin.
2. **Generative Sensory Environments**: Realizing that if an LLM can decode a thought from a neural vector, it can also *encode* a target thought into a sensory environment. By piping mathematically optimized audio frequencies (binaural beats) and visual cues into a VR headset (like an Oculus Quest 2), SovereignMind can "write" to the brain safely and effectively.

By combining these two concepts under the protective umbrella of the SovereignShield N-Model Consensus, we created a fully functional, bidirectional, fail-safe Holodeck.

---

## Core Architecture

SovereignMind operates as a bidirectional translation engine with two primary closed-loop systems:

### 1. The Sensory Stimulus Engineer (Write-Access)
Instead of passively reading brain waves, SovereignMind **actively writes** target cognitive states to the brain using mathematically optimized sensory environments. 
- **How it Works**: You provide a target cognitive state (e.g., "Deep Focus", "Sleep", "Creative Flow"). The SovereignShield Consensus panel calculates the exact auditory binaural/isochronic frequencies, visual hex color palettes, and semantic priming narratives required to induce that state safely.
- **VR Integration**: This JSON payload can be streamed directly to an Oculus Quest 2 or other VR headset to dynamically alter the user's environment in real-time.

### 2. The Semantic Signal Decoder (Read-Access)
Leveraging cutting-edge research in non-invasive fMRI BOLD and MEG decoding, SovereignMind translates raw latent neural vectors directly into human language and visual imagery.
- **How it Works**: A latent semantic vector (representing regional brain activations) is passed into the N-Model Consensus engine. The LLM panel decodes the latent space and reconstructs the exact inner monologue (sentence) and visual scene the user is experiencing.

---

## SovereignShield Safety & Enforcement

SovereignMind implements the core philosophies of **IntentShield** and **LogicShield** to act as a fail-safe firewall between raw AI generation and human cognition.

- **Enforcing the Action (Stimulus Boundaries)**: The mathematical parameters (audio frequencies, visual colors) are strictly bounded. If the AI attempts to generate a physiologically dangerous stimulus (e.g., strobe patterns that could induce seizures), the consensus fails and the action is cryptographically vetoed.
- **Enforcing the Data (Semantic Intent)**: The semantic priming scripts and decoded sentences are audited for psychological safety. Harmful, malicious, or distressing intents are flagged by the secondary consensus models and blocked.
- **Cryptographic Verification**: Every successful generation or decoding produces a deterministic hash. The BCI hardware (e.g., the VR headset) will only execute payloads that carry a verified SovereignShield cryptographic signature.

---

## 📂 Codebase File Structure

The project is organized into four core files, each handling a specific piece of the bidirectional translation pipeline:

### 1. `main.py` (The AI Consensus Engine)
This is the "brain" of the BCI, powered by the SovereignShield N-Model architecture.
- Contains the `OpenRouterMCPProvider` which connects to external LLMs (GPT-4o, Gemini).
- Houses the **Sensory Stimulus Engineer** logic (`generate_safe_stimulus()`) which maps target cognitive states to mathematically optimized audio frequencies, visual colors, and semantic scripts.
- Houses the **Semantic Decoder** logic (`decode_semantic_signal()`) which translates raw latent fMRI/MEG vectors directly into human-readable sentences and visual imagery.
- Enforces multi-model consensus via `ConsensusVerifier` to guarantee safety before returning any payload.

### 2. `server.py` (The API Router)
This is the FastAPI backend that handles all incoming real-time traffic from the VR headset or external hardware.
- Mounts the `/api/v1/generate_stimulus` endpoint to request new sensory environments.
- Mounts the `/api/v1/verify_feedback` endpoint to ingest physiological sensors (Heart Rate, Pupil Dilation) from the VR headset to track efficacy.
- Mounts the `/api/v1/decode_semantic_signals` endpoint to ingest raw brain vectors.
- Bridges the AI Engine (`main.py`) with the Ground Truth Database (`db.py`).

### 3. `db.py` (The Ground Truth Ledger)
Handles persistent SQLite storage (`mind_truth.db`) for all verified transactions.
- Maintains the `verified_stimulus_cues` table, logging every generated stimulus payload, its target state, and the cryptographic hash that approved it.
- Maintains the `verified_semantic_decodings` table, logging raw latent brain vectors alongside their successfully decoded semantic sentences.
- Ensures a complete audit trail of what the system requested, generated, and verified.

### 4. `mock_data.py` (The Simulator)
A utility script used for testing the system without requiring physical VR or BCI hardware.
- Generates `mock_telemetry.json`: Simulated biofeedback data (HRV, heart rate, pupil dilation) to test the efficacy loop.
- Generates `mock_fmri_vector.json`: A simulated latent brain vector array to test the semantic decoder engine.

---

## API Endpoints

SovereignMind exposes a FastAPI backend (`server.py`) for real-time integration.

### `POST /api/v1/generate_stimulus`
Generates cryptographically verified sensory parameters for a target state.
**Request Payload:**
```json
{
  "target_state": "Deep Focus"
}
```
**Response Payload:**
```json
{
  "status": "success",
  "verified_hash": "4179044f9b7a3cf6c38ad06e7fe469bcb0af210b3f4a09de8f7e1e76d9c862fe",
  "stimulus_parameters": {
    "audio_frequency_hz": 40,
    "visual_color_hex": "#0000FF",
    "semantic_priming_script": "Your mind is clear and sharp. Distractions fade away.",
    "safety_flag": "SAFE"
  }
}
```

### `POST /api/v1/verify_feedback`
Closes the loop by ingesting biological telemetry (e.g., Heart Rate, Heart Rate Variability (HRV), and Pupil Dilation) from the VR headset's sensors. This allows the system to measure the real-time physiological efficacy of the generated stimulus and dynamically adapt if the user's stress levels spike.

**Request Payload:**
```json
{
  "verified_hash": "4179044f9b7a3cf6c38ad06e7fe469bcb0af210b3f4a09de8f7e1e76d9c862fe",
  "pupil_dilation_mm": 5.5,
  "heart_rate_bpm": 65,
  "hrv_ms": 45.0
}
```
**Response Payload:**
```json
{
  "status": "success",
  "verified_hash": "4179044f9b7a3cf6c38ad06e7fe469bcb0af210b3f4a09de8f7e1e76d9c862fe",
  "efficacy_score": 1.0,
  "message": "Telemetry feedback logged."
}
```

### `POST /api/v1/decode_semantic_signals`
Translates a latent fMRI/MEG neural vector into human language.
**Request Payload:**
```json
{
  "fmri_vector": [10.0, 20.0, 30.0]
}
```
**Response Payload:**
```json
{
  "status": "success",
  "verified_hash": "d078535589f7a2fe4ee3d42802e1888d006493b2848dc3db46ddd887448c0ce4",
  "decoded_semantics": {
    "reconstructed_sentence": "I need a glass of water",
    "visual_scene_description": "A clear glass of water"
  }
}
```

---

## Ground Truth Database

All cryptographically approved parameters, decodings, and telemetry efficacy scores are logged in the local SQLite database (`mind_truth.db`).
- `verified_stimulus_cues`: Logs the target state, generated parameters, and confidence scores.
- `verified_semantic_decodings`: Logs the source latent vector alongside the reconstructed sentence and visual imagery.

---

## Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Configuration**:
   Ensure `OPENROUTER_API_KEY` is set in your `.env` file to power the LLM Consensus panel.
3. **Start the Engine**:
   ```bash
   python server.py
   ```
4. **Generate Mock Data**:
   Use `python mock_data.py` to generate simulated biofeedback payloads and latent fMRI vectors for testing the endpoints.
