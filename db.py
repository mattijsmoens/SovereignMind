import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "mind_truth.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verified_neural_maps (
            hash TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_file TEXT,
            cognitive_intention TEXT,
            confidence_score REAL,
            raw_features JSON
        )
    ''')
    conn.commit()
    conn.close()

def save_verified_map(features: dict, extraction: dict, verified_hash: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO verified_neural_maps 
        (hash, source_file, cognitive_intention, confidence_score, raw_features)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        verified_hash,
        features.get("file", "unknown"),
        extraction.get("cognitive_intention"),
        1.0,
        json.dumps(features)
    ))
    conn.commit()
    conn.close()
