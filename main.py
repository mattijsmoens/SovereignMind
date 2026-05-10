import os
import sys
import json
import logging
import requests

# Add the parent directory so we can import sovereign_mcp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SovereignMind")

os.environ["SOVEREIGN_MCP_SKIP_INTEGRITY"] = "1"

from sovereign_mcp import ToolRegistry, OutputGate, ConsensusVerifier
from sovereign_mcp.consensus import ModelProvider

class OpenRouterMCPProvider(ModelProvider):
    def __init__(self, model_id, api_key):
        super().__init__(model_id, temperature=0)
        self.api_key = api_key

    def extract_structured(self, content, schema):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        You are a highly advanced BCI neural decoder. 
        Analyze the following deterministic EEG mathematical features.
        If you see a Beta wave burst (13-30 Hz) with an amplitude > 10uV in the C3 channel, it indicates a "Motor Command: Right Hand".
        Otherwise, map to "Resting State".
        Return ONLY valid JSON matching this schema:
        {json.dumps(schema, indent=2)}
        
        Data:
        {content}
        """
        
        payload = {
            "model": self.model_id,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter Error: {response.text}")
            
        data = response.json()
        raw_output = data["choices"][0]["message"]["content"]
        
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_output)

output_schema = {
    "cognitive_intention": {"type": "string", "description": "The interpreted user intention (e.g. 'Motor Command: Right Hand')"},
    "key_channels_activated": {"type": "array", "items": {"type": "string"}}
}

def analyze_neural_features(features: dict) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set in .env")

    gpt4o = OpenRouterMCPProvider("openai/gpt-4o-mini", api_key=api_key)
    gemini = OpenRouterMCPProvider("google/gemini-2.0-flash-001", api_key=api_key)
    llama3 = OpenRouterMCPProvider("meta-llama/llama-3.3-70b-instruct", api_key=api_key)

    verifier = ConsensusVerifier(model_a=gpt4o, consensus_models=[gemini, llama3])

    registry = ToolRegistry()
    registry.register_tool(
        name="decode_neural_intent",
        description="Decode EEG features into cognitive intentions.",
        capabilities=["analyze_eeg", "extract_json"],
        input_schema={"type": "object"},
        output_schema=output_schema,
        risk_level="HIGH"
    )
    
    frozen_registry = registry.freeze()
    gate = OutputGate(frozen_registry=frozen_registry, consensus_verifier=verifier)

    feature_string = json.dumps(features, indent=2)

    logger.info("[Consensus] Asking Primary Model (GPT-4o) to decode intention...")
    extracted_draft = gpt4o.extract_structured(feature_string, output_schema)

    logger.info("[Consensus] Sending draft to Sovereign Shield for 3-model verification...")
    result = verifier.verify(
        tool_output=extracted_draft, 
        frozen_schema=output_schema, 
        verification_source=feature_string
    )

    if not result.match:
        logger.error(f"[Consensus] VETOED. Reason: {result.reason}")
        raise ValueError(f"Consensus VETOED. Models produced conflicting neural mappings. Hashes: {result.hashes}")
        
    logger.info("[Consensus] Sovereign Shield APPROVED. Cryptographic Hash: " + result.hashes[0][:16])
    
    return {
        "verified_hash": result.hashes[0],
        "mapping": extracted_draft
    }
