import os
import sys

# Standalone execution

from main import decode_semantic_signal

if __name__ == "__main__":
    print("=== Testing SovereignMind with ConscienceBridge ===")
    try:
        # S1 is the actual subject ID, wheretheressmoke is the available test response
        result = decode_semantic_signal("S1", "perceived_speech", "wheretheressmoke")
        print("\n[SUCCESS] Pipeline executed.")
        print("\n--- Verified Hash ---")
        print(result["verified_hash"])
        print("\n--- Decoded Semantics (LLM Output) ---")
        import json
        print(json.dumps(result["decoded_semantics"], indent=2))
        print("\n--- Neural State (ConscienceBridge) ---")
        print(json.dumps(result["neural_state"], indent=2))
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
