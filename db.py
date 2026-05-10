import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "mind_truth.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verified_stimulus_cues (
            hash TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            target_state TEXT,
            confidence_score REAL,
            stimulus_parameters JSON
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verified_semantic_decodings (
            hash TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_vector JSON,
            confidence_score REAL,
            decoded_semantics JSON
        )
    ''')
    conn.commit()
    conn.close()

def save_verified_stimulus(target_state: str, stimulus_parameters: dict, verified_hash: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO verified_stimulus_cues 
        (hash, target_state, confidence_score, stimulus_parameters)
        VALUES (?, ?, ?, ?)
    ''', (
        verified_hash,
        target_state,
        1.0,
        json.dumps(stimulus_parameters)
    ))
    conn.commit()
    conn.close()

def save_verified_decoding(source_vector: list, decoded_semantics: dict, verified_hash: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO verified_semantic_decodings 
        (hash, source_vector, confidence_score, decoded_semantics)
        VALUES (?, ?, ?, ?)
    ''', (
        verified_hash,
        json.dumps(source_vector),
        1.0,
        json.dumps(decoded_semantics)
    ))
    conn.commit()
    conn.close()
