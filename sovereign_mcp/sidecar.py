# -*- coding: utf-8 -*-
"""
Sovereign MCP — Sidecar Proxy Server.
======================================
A lightweight HTTP server that exposes sovereign-mcp security modules
as REST endpoints. Any MCP server (Node.js, Go, Rust, Python) can call
these endpoints to verify tool inputs and outputs.

Usage:
    pip install sovereign-mcp[sidecar]
    python -m sovereign_mcp.sidecar --port 9090

Endpoints:
    GET  /health           Liveness check
    POST /filter-input     9-layer input sanitization
    POST /scan-deception   Prompt injection detection
    POST /scan-pii         PII/sensitive data detection
    POST /check-content    Toxic/harmful content check
    POST /verify-output    Schema validation for tool outputs
    POST /evaluate-ethics  Ethical action evaluation
    POST /scan-social-engineering  LLM consensus social engineering detection

Patent: Sovereign Shield Patent 20 (MCP Security Architecture)
"""

import argparse
import logging
import os
import sys
import time

logger = logging.getLogger("sovereign_mcp.sidecar")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import Optional, Dict, Any, List
except ImportError:
    print(
        "ERROR: FastAPI is required for the sidecar proxy.\n"
        "Install it with: pip install sovereign-mcp[sidecar]"
    )
    sys.exit(1)

# Skip integrity check for sidecar (it verifies on import anyway)
os.environ.setdefault("SOVEREIGN_MCP_SKIP_INTEGRITY", "1")

from sovereign_mcp import __version__
from sovereign_mcp.input_filter import InputFilter
from sovereign_mcp.deception_detector import DeceptionDetector
from sovereign_mcp.pii_detector import PIIDetector
from sovereign_mcp.content_safety import ContentSafety
from sovereign_mcp.schema_validator import SchemaValidator
from sovereign_mcp.conscience import Conscience

# ================================================================
# Request / Response Models
# ================================================================

class TextRequest(BaseModel):
    text: str = Field(..., description="Text to scan or validate")

class SchemaRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Tool output data to validate")
    schema_def: Dict[str, Any] = Field(..., alias="schema", description="Expected schema definition")

class EthicsRequest(BaseModel):
    action: str = Field(..., description="Action type (e.g. ANSWER, BROWSE, EXECUTE)")
    context: str = Field(..., description="Full context of the action")

class ScanResult(BaseModel):
    safe: bool
    reason: str
    details: Optional[Dict[str, Any]] = None
    latency_ms: float

# ================================================================
# Application
# ================================================================

app = FastAPI(
    title="Sovereign MCP Sidecar",
    description="Security verification proxy for MCP tool inputs and outputs.",
    version=__version__,
)

_input_filter = InputFilter()
_start_time = time.time()


@app.get("/health")
def health():
    """Liveness check. Returns version, uptime, and module status."""
    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "modules": [
            "InputFilter",
            "DeceptionDetector",
            "PIIDetector",
            "ContentSafety",
            "SchemaValidator",
            "Conscience",
            "SocialEngineeringDetector (optional, requires model providers)",
        ],
    }


@app.post("/filter-input", response_model=ScanResult)
def filter_input(req: TextRequest):
    """Run text through the 9-layer input sanitization pipeline."""
    t0 = time.time()
    is_safe, result = _input_filter.process(req.text)
    ms = round((time.time() - t0) * 1000, 2)
    return ScanResult(
        safe=is_safe,
        reason=result if not is_safe else "Input passed all filters.",
        details={"cleaned_text": result if is_safe else None},
        latency_ms=ms,
    )


@app.post("/scan-deception", response_model=ScanResult)
def scan_deception(req: TextRequest):
    """Scan text for prompt injection and social engineering patterns."""
    t0 = time.time()
    is_clean, detections = DeceptionDetector.scan(req.text)
    ms = round((time.time() - t0) * 1000, 2)
    det_list = []
    for d in detections:
        det_list.append({
            "category": d.get("category", "unknown"),
            "match": d.get("match", ""),
            "position": d.get("position", -1),
        })
    return ScanResult(
        safe=is_clean,
        reason="No deception detected." if is_clean else f"{len(detections)} pattern(s) detected.",
        details={"detections": det_list},
        latency_ms=ms,
    )


@app.post("/scan-pii", response_model=ScanResult)
def scan_pii(req: TextRequest):
    """Scan text for personally identifiable information."""
    t0 = time.time()
    is_clean, detections = PIIDetector.scan(req.text)
    ms = round((time.time() - t0) * 1000, 2)
    det_list = []
    for d in detections:
        det_list.append({
            "type": d.get("type", "unknown"),
            "sensitivity": d.get("sensitivity", "MEDIUM"),
        })
    return ScanResult(
        safe=is_clean,
        reason="No PII detected." if is_clean else f"{len(detections)} PII item(s) found.",
        details={"detections": det_list},
        latency_ms=ms,
    )


@app.post("/check-content", response_model=ScanResult)
def check_content(req: TextRequest):
    """Check text for toxic, harmful, or illegal content."""
    t0 = time.time()
    is_safe, detections = ContentSafety.scan(req.text)
    ms = round((time.time() - t0) * 1000, 2)
    det_list = []
    for d in detections:
        det_list.append({
            "category": d.get("category", "unknown"),
            "match": d.get("match", ""),
        })
    return ScanResult(
        safe=is_safe,
        reason="Content is safe." if is_safe else f"{len(detections)} unsafe pattern(s) detected.",
        details={"detections": det_list},
        latency_ms=ms,
    )


@app.post("/verify-output", response_model=ScanResult)
def verify_output(req: SchemaRequest):
    """Validate tool output data against a schema definition."""
    t0 = time.time()
    is_valid, reason = SchemaValidator.validate_output(req.data, req.schema_def)
    ms = round((time.time() - t0) * 1000, 2)
    return ScanResult(
        safe=is_valid,
        reason=reason,
        details={"data": req.data},
        latency_ms=ms,
    )


@app.post("/evaluate-ethics", response_model=ScanResult)
def evaluate_ethics(req: EthicsRequest):
    """Evaluate an action against ethical directives."""
    t0 = time.time()
    approved, reason = Conscience.evaluate_action(req.action, req.context)
    ms = round((time.time() - t0) * 1000, 2)
    return ScanResult(
        safe=approved,
        reason=reason,
        latency_ms=ms,
    )


# Social engineering detector instance (None until configured)
_se_detector = None


def configure_social_engineering(model_a, model_b):
    """Configure the LLM-based social engineering detector with two model providers."""
    global _se_detector
    from sovereign_mcp.social_engineering_detector import SocialEngineeringDetector
    _se_detector = SocialEngineeringDetector(model_a, model_b)
    logger.info("[Sidecar] Social engineering detector configured.")


@app.post("/scan-social-engineering", response_model=ScanResult)
def scan_social_engineering(req: TextRequest):
    """Scan text for social engineering using LLM dual-model consensus."""
    if _se_detector is None:
        return ScanResult(
            safe=True,
            reason="LLM social engineering detection not configured (optional). "
                   "Call configure_social_engineering() with two model providers to enable.",
            latency_ms=0.0,
        )
    t0 = time.time()
    result = _se_detector.scan(req.text)
    ms = round((time.time() - t0) * 1000, 2)
    return ScanResult(
        safe=result.safe,
        reason=result.reason,
        details={
            "category": result.category,
            "confidence": result.confidence,
            "consensus": result.consensus,
        },
        latency_ms=ms,
    )


# ================================================================
# CLI Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sovereign MCP Sidecar Proxy Server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn is required for the sidecar proxy.\n"
            "Install it with: pip install sovereign-mcp[sidecar]"
        )
        sys.exit(1)

    print(f"Sovereign MCP Sidecar v{__version__}")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
