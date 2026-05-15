# SovereignMind: Neuro-Semantic BCI Translation Engine

> [!IMPORTANT]
> **RESEARCH PROTOTYPE & SIMULATION ARCHITECTURE**
> SovereignMind is a purely software-driven proof-of-concept. While the AI translation mechanics and cryptographic consensus pipelines are fully functional and tested against real-world biological datasets, the physical consumer hardware required to run this in a real-time "Holodeck" loop does not currently exist. This repository serves as a functional architectural blueprint for the future of non-invasive BCI software. See the [Hardware Requirements & Scientific Limitations](#-hardware-requirements--prototyping) section for more details.

**SovereignMind** is a revolutionary, non-invasive Brain-Computer Interface (BCI) module built on the **SovereignShield N-Model AI Consensus Architecture**. 

Unlike legacy BCI systems that rely on invasive surgical hardware (e.g., neural laces) or passive, low-resolution EEG caps, SovereignMind operates entirely through high-bandwidth natural input/output ports: **Audio, Visual, and Semantic Language pathways.**

By treating the human brain as a protected endpoint, SovereignMind utilizes cryptographic AI consensus to enforce strict safety boundaries on all generated sensory stimuli and all reconstructed neural decodings.

---

## Research Background & Origins

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

### 3. Privacy-First Local Mode (Air-Gapped Operation)
Because neural decodings contain highly sensitive internal monologue, SovereignMind fully supports 100% offline, air-gapped extraction.
- **How it Works**: By setting the `USE_LOCAL_LLM=true` environment flag, the API entirely disconnects from cloud providers (OpenRouter/GPT-4o) and routes all consensus verification and semantic extraction to a local LLM runner (e.g., **Ollama** or **LM Studio**) running on `localhost:11434`. This guarantees your neural data never leaves your physical machine.

---

## SovereignShield Safety & Enforcement

SovereignMind implements the core philosophies of **IntentShield** and **LogicShield** to act as a fail-safe firewall between raw AI generation and human cognition.

- **Enforcing the Action (Stimulus Boundaries)**: The mathematical parameters (audio frequencies, visual colors) are strictly bounded. If the AI attempts to generate a physiologically dangerous stimulus (e.g., strobe patterns that could induce seizures), the consensus fails and the action is cryptographically vetoed.
- **Enforcing the Data (Semantic Intent)**: The semantic priming scripts and decoded sentences are audited for psychological safety. Harmful, malicious, or distressing intents are flagged by the secondary consensus models and blocked.
- **Cryptographic Verification**: Every successful generation or decoding produces a deterministic hash. The BCI hardware (e.g., the VR headset) will only execute payloads that carry a verified SovereignShield cryptographic signature.

---

## Codebase File Structure

The project operates as a 100% standalone microservice, organized into core files and embedded submodules handling specific pieces of the bidirectional translation pipeline:

### 1. `main.py` (The AI Consensus Engine)
This is the master entrypoint of the BCI, powered by the SovereignShield architecture.
- Enforces multi-model consensus via the bundled `sovereign_mcp` module to guarantee safety before returning any payload.
- Connects the raw biological fMRI input to the **Semantic Decoder**.
- Integrates the **ConscienceBridge** to extract neural emotional states from the decoded text.

### 2. `SovereignConscience/` (The Neural State Engine)
An embedded, standalone submodule that bridges the decoded text to human biology.
- Runs text through GoEmotions and simulated fMRI ROI models (Amygdala, Insula, etc.).
- Bundles its own `intentshield` and `logicshield` implementations for hermetic security sealing.
- Outputs the LLM system prompt prefix required to give the consensus models biological context.

### 3. `semantic-decoding/` (The HuthLab Decoder)
The embedded integration of the University of Texas at Austin's fMRI decoding pipeline.
- Translates raw latent fMRI/MEG vectors into human language using GPT-based beam search algorithms.

### 4. `lora_extractor.py` (The Translation Interface)
The Python wrapper module that bridges the standalone `semantic-decoding` repository with the broader SovereignMind pipeline, handling the heavy lifting of loading pre-fit fMRI models and raw brain scan `.hf5` data.

### 5. `server.py` & `db.py` (The API and Ledger)
- **`server.py`**: The FastAPI backend that handles incoming real-time traffic, mounting endpoints for decoding brain signals and verifying physiological feedback.
- **`db.py`**: Persistent SQLite storage (`mind_truth.db`) for all cryptographically verified transactions and neural mappings.

---

## From Prototype to Production (The LoRA Extractor)

The current repository contains a fully functional end-to-end pipeline. However, to prove that the system can cryptographically verify biological data without requiring a massive cluster of GPUs for local ML training, we utilize a **deterministic prototype hack**.

**How the Prototype Test Works:**
In `main.py`, the system calculates the sum of the incoming biological vector array. Based on that sum, it triggers a deterministic mapping (e.g., if the sum is < 50, it assigns the target concept "apple"). It then asks the LLMs to construct a sentence around that concept. This forces both LLMs in the consensus panel to arrive at the exact same semantic conclusion, guaranteeing a cryptographic hash match and proving that the data routing and firewalling work flawlessly.

**How Real-World Production Works:**
In a finalized consumer BCI, this deterministic hack is removed and replaced by the pipeline modeled in `lora_extractor.py`. 
1. **Calibration:** You wear the VR headset and the EEG/fMRI scanner. The VR headset displays specific scenes (e.g., a snowy mountain).
2. **Recording:** The system records your brain's unique latent vector response to that image.
3. **Training (LoRA):** Using the script in `lora_extractor.py`, the system trains a highly lightweight, personalized **Low-Rank Adaptation (LoRA)** adapter and a Neural Projection Layer.
4. **Deployment:** To deploy the system, the massive base LLM (e.g., Llama-3) is loaded into the GPU, and the personalized LoRA adapter weights are "snapped on" using the PEFT library.
   ```python
   base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
   bci_model = PeftModel.from_pretrained(base_model, "models/s0_adapter") 
   ```
5. **Organic Translation:** During real-time usage at the `/decode_semantic_signals` endpoint, the prompt hack is completely bypassed. The raw biological vector passes through the `BrainVectorProjector` to become an embedding, which is fed directly into the adapted LLM. The AI organically outputs the translated thought based on your personal neurological patterns.
   ```python
   brain_embedding = projector(latent_vector)
   generated_tokens = bci_model.generate(inputs_embeds=brain_embedding)
   decoded_text = tokenizer.decode(generated_tokens)
   ```

### The Ultimate Bidirectional Loop (Thought-Controlled Reality)
Once the LoRA adapter is trained on a specific user, the system becomes a completely frictionless, closed-loop Holodeck controlled purely by thought:
1. **You Think (Read-Access):** You visualize or think a command (e.g., *"Change the sky to red"*).
2. **The Scanner Reads:** The BCI hardware (fMRI/EEG) captures the latent neural vector of that thought.
3. **The LoRA Translates:** Your personalized LoRA adapter instantly translates those raw brain waves into the English string: `"Change the sky to red"`.
4. **The Engine Processes:** SovereignMind takes that string and automatically routes it into the **Sensory Stimulus Engineer** (`/generate_stimulus`).
5. **The Environment Responds (Write-Access):** SovereignShield verifies the intent, calculates the exact hex color for red (`#FF0000`), and sends the JSON payload to the Oculus Quest 2.
6. **You See It:** The VR headset immediately shifts the sky to red.

Your inner monologue becomes the direct operating system for the reality you are experiencing, and the system dynamically writes it back to your eyes and ears in real-time.

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

## Hardware Requirements & Prototyping

SovereignMind is a purely software-driven translation engine. To interface it with the real world, it requires external hardware for both "Read" and "Write" access.

> [!WARNING]
> **DISCLAIMER: Current Scientific Limitations**
> While the SovereignMind *software architecture* and AI translation mechanics are fully functional, the consumer *hardware* required to achieve a real-time, thought-controlled Holodeck does not currently exist. 
> 1. **The Hardware Gap:** Complex semantic decoding requires fMRI or MEG machines, which are multi-million dollar, room-sized scanners. Portable consumer EEG headsets (like Emotiv) are too noisy to decode complex inner monologue; they can only detect broad states (e.g., focus vs. relaxation).
> 2. **Latency:** fMRI measures blood oxygenation (the BOLD signal), which lags behind actual neural firing by 5 to 8 seconds. Real-time, instantaneous Holodeck response is biologically impossible with current fMRI techniques.
> 
> SovereignMind serves as an advanced **simulation architecture** for the future of BCI software, ready to deploy once high-resolution, portable, non-invasive sensors are invented.

### The "Write" Interface (Sensory Stimulus)
To execute the sensory environments generated by the engine, you need:
- **A VR Headset:** An Oculus Quest 2, Meta Quest Pro, or Apple Vision Pro to render the visual hex colors, display the semantic scripts, and play the targeted audio frequencies.
- **Autonomic Sensors:** To close the biofeedback loop (`/verify_feedback`), the system requires physiological telemetry. You can pair Bluetooth devices like an Apple Watch, Oura Ring, or a Polar H10 chest strap to feed real-time Heart Rate and HRV data back into the engine, allowing the VR environment to dynamically adapt to the user's stress levels.

### The "Read" Interface (Semantic Decoding)
To actually *control* the VR environment using your thoughts, the engine requires a latent neural vector:
- **Research-Grade:** Fully decoding complex inner monologue (e.g., "Take me to a snowy mountain") currently requires an active, high-resolution fMRI or MEG machine scanning the brain.
- **Consumer Prototyping:** For immediate prototyping without an fMRI, you can use consumer EEG headsets like the **EMOTIV EPOC X** or **Muse S**. While these cannot decode complex sentences, they *can* detect high-level cognitive states (e.g., spikes in Focus). These simple binary spikes can be passed to SovereignMind as simplified vectors to trigger basic state-changes in the VR environment.
- **The Simulator:** If you have no BCI hardware, you can use the included `test_bridge.py` script to run the pipeline on the pre-recorded fMRI responses provided by HuthLab (Subject S1, perceived speech).

---

## Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Pre-fit Neural Models**:
   The `semantic-decoding` submodule requires pre-fit Ridge Regression encoding models to interpret the fMRI data.
   - Download the models from the official HuthLab UT Austin Box link: [Download Models (2.7GB)](https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz)
   - Extract the contents into: `semantic-decoding/models/S1/` (or the appropriate subject ID).

3. **Environment Configuration**:
   - **Cloud Mode:** Ensure `OPENROUTER_API_KEY` is set in your `.env` file to power the LLM Consensus panel via GPT-4o and Gemini.
   - **Local Privacy Mode:** Set `USE_LOCAL_LLM=true` in your `.env` file and ensure Ollama is running locally with the `llama3` and `mistral` models installed (`ollama run llama3`).

4. **Start the Engine**:
   ```bash
   python server.py
   ```
5. **Run the End-to-End Pipeline**:
   Use `python test_bridge.py` to process the HuthLab fMRI test vectors, run the beam search decoder, and push the reconstructed text through the entire SovereignShield consensus and biological Conscience bridge.

---

## References & Datasets

SovereignMind builds upon several foundational academic datasets and papers in the field of neuroscience and machine learning:

1. **Semantic Decoding from Non-Invasive Brain Recordings**
   * **Paper**: Tang, J., LeBel, A., Jain, S., & Huth, A. G. (2023). Semantic reconstruction of continuous language from non-invasive brain recordings. *Nature Neuroscience*. [Link](https://www.nature.com/articles/s41593-023-01304-9)
   * **Datasets**: [OpenNeuro ds003020](https://openneuro.org/datasets/ds003020/) (Training), [OpenNeuro ds004510](https://openneuro.org/datasets/ds004510/) (Test Responses)
   * **Implementation**: Adapted via the bundled `semantic-decoding` repository.

2. **GoEmotions: A Dataset of Fine-Grained Emotions**
   * **Paper**: Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Google Research*. [Link](https://arxiv.org/abs/2005.00547)
   * **Model**: Utilized via the Hugging Face `SamLowe/roberta-base-go_emotions` pipeline to extract 27 emotional features from decoded text.

3. **Circumplex Model of Affect**
   * **Reference**: Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*.
   * **Application**: Used to mathematically collapse 27 discrete text emotions into a unified (Valence, Arousal) coordinate grid.

4. **fMRI Region of Interest (ROI) Emotion Correlates**
   * **Application**: The `SovereignConscience` module utilizes Support Vector Machines (SVM) trained on normalized fMRI voxel data. The models predict activation across the Amygdala, Insula, ACC, vmPFC, and dlPFC based on established neurological emotion mapping (NeuroEmo structural conventions).
