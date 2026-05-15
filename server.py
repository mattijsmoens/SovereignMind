import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sqlite3
from pathlib import Path

from db import init_db, save_verified_decoding
from main import decode_semantic_signal

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SovereignMind")

app = FastAPI(title="SovereignMind Semantic Brain Decoder")

init_db()

class SemanticDecodeRequest(BaseModel):
    subject_id: str  # e.g., "S1", "S2", "S3"
    experiment: str   # e.g., "perceived_speech", "imagined_speech", "perceived_movies"
    task: str         # The specific task/story name

@app.post("/api/v1/decode_semantic_signals")
def decode_semantic_signals(req: SemanticDecodeRequest):
    """
    Decodes real fMRI brain recordings into semantic text using the HuthLab
    decoder (Tang et al., Nature Neuroscience 2023), then verifies the output
    via the Sovereign Shield N-Model Consensus.
    """
    logger.info("==================================================")
    logger.info(f"Incoming Request: Decode fMRI for subject={req.subject_id}, experiment={req.experiment}, task={req.task}")
    
    # N-Model Consensus Semantic Decoding via real HuthLab decoder
    logger.info("[Server] Passing brain recording to HuthLab Decoder + Sovereign Shield Consensus Engine...")
    try:
        consensus_result = decode_semantic_signal(req.subject_id, req.experiment, req.task)
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
        
    # Save to Ground Truth Database
    logger.info("[Server] Consensus reached successfully. Saving decoded semantics to DB.")
    save_verified_decoding(
        source_vector={"subject": req.subject_id, "experiment": req.experiment, "task": req.task}, 
        decoded_semantics=consensus_result["decoded_semantics"], 
        verified_hash=consensus_result["verified_hash"]
    )
    
    return {
        "status": "success",
        "verified_hash": consensus_result["verified_hash"],
        "decoded_semantics": consensus_result["decoded_semantics"],
        "raw_huthlab_output": consensus_result.get("raw_huthlab_output", {})
    }

@app.get("/api/v1/database/decodings")
def get_verified_decodings():
    """Fetches the ground-truth database of all verified brain signal decodings."""
    db_path = Path(__file__).parent / "mind_truth.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM verified_semantic_decodings ORDER BY timestamp DESC LIMIT 20')
        rows = [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    
    for row in rows:
        if "decoded_semantics" in row:
            row["decoded_semantics"] = json.loads(row["decoded_semantics"])
        if "source_vector" in row:
            row["source_vector"] = json.loads(row["source_vector"])
        
    return {"status": "success", "count": len(rows), "data": rows}

if __name__ == "__main__":
    logger.info("Starting SovereignMind Semantic Brain Decoder on port 8007...")
    uvicorn.run(app, host="0.0.0.0", port=8007)
