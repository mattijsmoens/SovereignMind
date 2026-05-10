import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sqlite3
from pathlib import Path

from db import init_db, save_verified_stimulus, save_verified_decoding
from main import generate_safe_stimulus, decode_semantic_signal

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SovereignMind")

app = FastAPI(title="SovereignMind Sensory Stimulus Engine")

init_db()

class StimulusRequest(BaseModel):
    target_state: str

class FeedbackRequest(BaseModel):
    verified_hash: str
    pupil_dilation_mm: float
    heart_rate_bpm: int
    hrv_ms: float

class SemanticDecodeRequest(BaseModel):
    fmri_vector: list

@app.post("/api/v1/decode_semantic_signals")
def decode_semantic_signals(req: SemanticDecodeRequest):
    """
    Translates a simulated non-invasive latent brain vector (fMRI/MEG)
    directly into semantic sentences and visual scenes using LLM consensus.
    """
    logger.info("==================================================")
    logger.info(f"Incoming Request: Decode Semantic Signal Vector of length {len(req.fmri_vector)}")
    
    # N-Model Consensus Semantic Decoding
    logger.info("[Server] Passing latent vector to Sovereign Shield AI Consensus Engine...")
    try:
        consensus_result = decode_semantic_signal(req.fmri_vector)
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
        
    # Save to Ground Truth Database
    logger.info("[Server] Consensus reached successfully. Saving decoded semantics to DB.")
    save_verified_decoding(
        source_vector=req.fmri_vector, 
        decoded_semantics=consensus_result["decoded_semantics"], 
        verified_hash=consensus_result["verified_hash"]
    )
    
    return {
        "status": "success",
        "verified_hash": consensus_result["verified_hash"],
        "decoded_semantics": consensus_result["decoded_semantics"]
    }

@app.post("/api/v1/generate_stimulus")
def generate_stimulus(req: StimulusRequest):
    """
    Generates a targeted sensory stimulus (audio/visual/semantic) using the
    N-Model Consensus architecture to safely trigger the desired cognitive state.
    """
    logger.info("==================================================")
    logger.info(f"Incoming Request: Generate Stimulus for '{req.target_state}'")
    
    # N-Model Consensus Stimulus Generation
    logger.info("[Server] Passing target state to Sovereign Shield AI Consensus Engine...")
    try:
        consensus_result = generate_safe_stimulus(req.target_state)
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
        
    # Save to Ground Truth Database
    logger.info("[Server] Consensus reached safely. Saving verified stimulus parameters to DB.")
    save_verified_stimulus(
        target_state=req.target_state, 
        stimulus_parameters=consensus_result["stimulus_parameters"], 
        verified_hash=consensus_result["verified_hash"]
    )
    
    return {
        "status": "success",
        "verified_hash": consensus_result["verified_hash"],
        "target_state": req.target_state,
        "stimulus_parameters": consensus_result["stimulus_parameters"]
    }

@app.post("/api/v1/verify_feedback")
def verify_feedback(req: FeedbackRequest):
    """
    Closes the loop by accepting bio-feedback telemetry to verify if the 
    generated stimulus successfully triggered the physiological markers of the state.
    """
    logger.info("==================================================")
    logger.info(f"Incoming Telemetry: Verifying efficacy for Hash {req.verified_hash[:8]}...")
    
    # In a real implementation, we would compare the HRV/Pupillometry against the expected baseline for the target state.
    efficacy_score = 1.0 if req.hrv_ms > 40.0 else 0.5
    
    return {
        "status": "success",
        "verified_hash": req.verified_hash,
        "efficacy_score": efficacy_score,
        "message": "Telemetry feedback logged."
    }

@app.get("/api/v1/database/cues")
def get_verified_cues():
    """Fetches the ground-truth stimulus dictionary."""
    db_path = Path(__file__).parent / "mind_truth.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM verified_stimulus_cues ORDER BY timestamp DESC LIMIT 20')
        rows = [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    
    for row in rows:
        if "stimulus_parameters" in row:
            row["stimulus_parameters"] = json.loads(row["stimulus_parameters"])
        
    return {"status": "success", "count": len(rows), "data": rows}

if __name__ == "__main__":
    logger.info("Starting SovereignMind Sensory Stimulus Engine on port 8007...")
    uvicorn.run(app, host="0.0.0.0", port=8007)
