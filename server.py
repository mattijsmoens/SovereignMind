import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sqlite3
from pathlib import Path

from db import init_db, save_verified_map
from signal_processing import process_eeg_data
from main import analyze_neural_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SovereignMind")

app = FastAPI(title="SovereignMind BCI Orchestrator")

init_db()

class EEGRequest(BaseModel):
    csv_path: str

@app.post("/analyze_eeg")
def analyze_eeg(req: EEGRequest):
    """
    Ingests an EEG CSV file, extracts deterministic features via NumPy,
    and runs a 3-model Sovereign Shield consensus to decode the cognitive intention.
    """
    logger.info("==================================================")
    logger.info(f"Incoming Request: Analyze EEG file {req.csv_path}")
    
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail="EEG CSV file not found.")
        
    # 1. Deterministic Signal Processing
    logger.info("[Server] Running deterministic Numpy signal processing...")
    try:
        features = process_eeg_data(req.csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process EEG data: {e}")
        
    logger.info(f"[Server] Extracted mathematical features for {len(features['channel_features'])} channels.")
    
    # 2. N-Model Consensus Decoder
    logger.info("[Server] Passing features to Sovereign Shield AI Decoder...")
    try:
        consensus_result = analyze_neural_features(features)
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
        
    # 3. Save to Ground Truth Database
    logger.info("[Server] Consensus reached. Saving verified mapping to DB.")
    save_verified_map(features, consensus_result["mapping"], consensus_result["verified_hash"])
    
    return {
        "status": "success",
        "verified_hash": consensus_result["verified_hash"],
        "cognitive_intention": consensus_result["mapping"]["cognitive_intention"],
        "deterministic_features": features
    }

@app.get("/database/mappings")
def get_verified_mappings():
    """Fetches the ground-truth brain dictionary."""
    db_path = Path(__file__).parent / "mind_truth.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM verified_neural_maps ORDER BY timestamp DESC LIMIT 20')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    for row in rows:
        row["raw_features"] = json.loads(row["raw_features"])
        
    return {"status": "success", "count": len(rows), "data": rows}

if __name__ == "__main__":
    logger.info("Starting SovereignMind Orchestrator on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
