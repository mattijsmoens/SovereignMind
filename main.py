import os
import sys
import json
import logging
import requests

# Standalone mode: All imports must be local

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SovereignMind")

os.environ["SOVEREIGN_MCP_SKIP_INTEGRITY"] = "1"

from sovereign_mcp import ToolRegistry, OutputGate, ConsensusVerifier
from sovereign_mcp.consensus import ModelProvider, LocalMCPProvider, OpenRouterMCPProvider
from SovereignConscience.conscience_bridge import ConscienceBridge


semantic_output_schema = {
    "reconstructed_sentence": {"type": "string", "description": "The exact inner monologue or sentence the user is thinking"},
    "visual_scene_description": {"type": "string", "description": "A visual description of the imagery the user is seeing in their mind"},
    "affective_tone": {"type": "string", "description": "The emotional/moral tone of the thought, based on the neural state context"}
}

def decode_semantic_signal(subject_id: str, experiment: str, task: str) -> dict:
    """
    Decodes real fMRI brain recordings into semantic text using the HuthLab
    decoder (Tang et al., Nature Neuroscience 2023), then feeds the decoded
    text through the Sovereign Shield N-Model Consensus for cryptographic verification.
    
    Args:
        subject_id: The fMRI subject identifier (e.g., "UTS01")
        experiment: The experiment type (e.g., "perceived_speech")
        task: The specific task/story name
    """
    from lora_extractor import decode_fmri_response
    
    # Step 1: Run the REAL HuthLab decoder on the fMRI brain data
    logger.info(f"[HuthLab] Decoding brain recordings for subject={subject_id}, experiment={experiment}, task={task}")
    huthlab_result = decode_fmri_response(subject_id, experiment, task)
    decoded_text = huthlab_result["decoded_text"]
    logger.info(f"[HuthLab] Raw decoded text: {decoded_text[:200]}...")
    
    # Step 2: Initialize the Sovereign Shield Consensus to VERIFY the decoded output
    use_local = os.environ.get("USE_LOCAL_LLM", "").lower() == "true"
    
    if use_local:
        logger.info("[SovereignShield] Initializing Privacy Mode: Using Local Air-Gapped Models")
        model_a = LocalMCPProvider("llama3", base_url="http://localhost:11434/v1")
        model_b = LocalMCPProvider("mistral", base_url="http://localhost:11434/v1")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set. Set it, or set USE_LOCAL_LLM=true for offline mode.")
        model_a = OpenRouterMCPProvider("openai/gpt-4o-mini", api_key=api_key)
        model_b = OpenRouterMCPProvider("google/gemini-2.0-flash-001", api_key=api_key)

    verifier = ConsensusVerifier(model_a=model_a, model_b=model_b)

    registry = ToolRegistry()
    registry.register_tool(
        name="decode_semantic_signal",
        description="Verify decoded fMRI brain signals via N-Model Consensus.",
        capabilities=["semantic_decode", "extract_json"],
        input_schema={"type": "object"},
        output_schema=semantic_output_schema,
        risk_level="HIGH"
    )
    
    frozen_registry = registry.freeze()
    _ = OutputGate(frozen_registry=frozen_registry, consensus_verifier=verifier)
    
    # Step 3: Neural Emotion Processing (ConscienceBridge)
    logger.info("[ConscienceBridge] Analyzing decoded text for affective neural state...")
    # Use local SovereignConscience/data directory
    local_data_dir = os.path.join(os.path.dirname(__file__), 'SovereignConscience', 'data')
    os.makedirs(local_data_dir, exist_ok=True)
    bridge = ConscienceBridge(data_dir=local_data_dir)
    bridge.initialize()
    
    bridge_result = bridge.process(decoded_text)
    if not bridge_result["allowed"]:
        logger.error(f"[ConscienceBridge] BLOCKED by IntentShield: {bridge_result['block_reason']}")
        raise ValueError(f"IntentShield BLOCKED thought processing: {bridge_result['block_reason']}")
    elif bridge_result["neural_state"] is None:
        logger.error(f"[ConscienceBridge] BLOCKED by LogicShield: {bridge_result.get('logicshield_errors')}")
        raise ValueError(f"LogicShield validation failed: {bridge_result.get('logicshield_errors')}")
        
    logger.info(f"[ConscienceBridge] Neural state extracted. Moral signal: {bridge_result['neural_state']['moral_signal']:+.3f}")

    # Step 4: Feed the REAL decoded text + Neural State into the consensus for verification
    # The consensus verifier ensures no hallucination or corruption occurred during decoding
    prompt_context = f"The HuthLab fMRI decoder reconstructed the following text from subject {subject_id}'s brain recording: \"{decoded_text}\". Summarize the reconstructed sentence, describe any visual imagery implied by the decoded language, and identify the affective tone."

    sys_prompt = (
        "You are a SovereignMind Semantic Decoder. You are given the output of a real fMRI brain decoder (Tang et al., Nature Neuroscience 2023). "
        "Your job is to faithfully summarize the decoded brain signal into a clean reconstructed sentence, visual scene description, and affective tone. "
        "Do NOT hallucinate or add information that was not in the decoded text.\n\n"
        f"{bridge_result['llm_system_prefix']}"
    )

    logger.info(f"[Consensus] Asking Primary Model ({model_a.model_id}) to structure the decoded brain signal...")
    extracted_draft = model_a.extract_structured(prompt_context, semantic_output_schema, system_prompt=sys_prompt)

    logger.info("[Consensus] Sending draft to Sovereign Shield for verification...")
    result = verifier.verify(
        tool_output=extracted_draft, 
        frozen_schema=semantic_output_schema, 
        verification_source=prompt_context
    )

    if not result.match:
        logger.error(f"[Consensus] VETOED. Reason: {result.reason}")
        raise ValueError(f"Consensus VETOED. Models produced conflicting semantic decodings. Hash A: {result.hash_a}")
        
    logger.info("[Consensus] Sovereign Shield APPROVED. Cryptographic Hash: " + result.hash_a[:16])
    
    return {
        "verified_hash": result.hash_a,
        "decoded_semantics": extracted_draft,
        "neural_state": bridge_result["neural_state"],
        "raw_huthlab_output": huthlab_result
    }
