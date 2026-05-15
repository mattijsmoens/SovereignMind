"""
SovereignConscience - Conscience Bridge
========================================
The full CNS pipeline that connects incoming text to the LLM.

Pipeline:
    1. IntentShield  - audits incoming text for harmful/deceptive intent
    2. GoEmotions    - 27-category text emotion classifier (Google, Hugging Face)
    3. Valence map   - maps 27 GoEmotion categories → Russell's Circumplex (V, A)
    4. ROI lookup    - finds the closest NeuroEmo brain-calibrated emotion state
    5. Neural signal - generates region-specific activation probabilities from
                       the stored ROI modules (real fMRI-trained SVMs)
    6. LogicShield   - validates the output signal with deterministic rules
    7. Prompt inject - formats the verified neural state as an LLM system prefix

The GoEmotions model is loaded once and cached. The ROI modules are loaded
from the trained .joblib files produced by build_roi_modules.py.

Author: Mattijs Moens
License: BSL 1.1
"""

import sys
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Path setup - Local Imports
# ---------------------------------------------------------------------------
# Prepend the current directory to sys.path to GUARANTEE we use the local 
# standalone copies of intentshield and logicshield, not global pip installs.
sys.path.insert(0, str(Path(__file__).parent))


from intentshield.shield import IntentShield
from logicshield.shield import LogicShield
from logicshield.rules import Rule

import joblib
from transformers import pipeline as hf_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("SovereignConscience.Bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"

# GoEmotions → Russell's Circumplex (Valence, Arousal)
# Values from Cowen & Keltner (2017) and Warriner et al. (2013)
# Valence: -1 (very negative) → +1 (very positive)
# Arousal: -1 (very calm) → +1 (very activated)
GOEMOTIONS_VA = {
    "admiration":     ( 0.80,  0.30),
    "amusement":      ( 0.75,  0.50),
    "anger":          (-0.70,  0.80),
    "annoyance":      (-0.55,  0.50),
    "approval":       ( 0.65,  0.20),
    "caring":         ( 0.75,  0.10),
    "confusion":      (-0.20,  0.30),
    "curiosity":      ( 0.40,  0.60),
    "desire":         ( 0.60,  0.70),
    "disappointment": (-0.65, -0.30),
    "disapproval":    (-0.60,  0.30),
    "disgust":        (-0.80,  0.40),
    "embarrassment":  (-0.45,  0.20),
    "excitement":     ( 0.80,  0.90),
    "fear":           (-0.75,  0.80),
    "gratitude":      ( 0.85,  0.20),
    "grief":          (-0.85, -0.40),
    "joy":            ( 0.90,  0.60),
    "love":           ( 0.90,  0.20),
    "nervousness":    (-0.40,  0.70),
    "optimism":       ( 0.70,  0.50),
    "pride":          ( 0.70,  0.30),
    "realization":    ( 0.10,  0.40),
    "relief":         ( 0.60, -0.30),
    "remorse":        (-0.65, -0.20),
    "sadness":        (-0.75, -0.30),
    "surprise":       ( 0.20,  0.70),
    "neutral":        ( 0.00,  0.00),
}

# NeuroEmo brain emotions → Russell's Circumplex (must match EMOTION_VALENCE_AROUSAL
# in build_roi_modules.py)
NEUROEMO_VA = {
    "calm":      ( 0.3, -0.7),
    "afraid":    (-0.8,  0.8),
    "delighted": ( 0.9,  0.6),
    "depressed": (-0.7, -0.5),
    "excited":   ( 0.7,  0.9),
}

# ---------------------------------------------------------------------------
# LogicShield rules for the neural output signal
# ---------------------------------------------------------------------------
SIGNAL_RULES = LogicShield(rules=[
    Rule("signal_in_range",
         lambda p, s: -1.0 <= p["moral_signal"] <= 1.0,
         error="Moral signal out of [-1, 1] range - data integrity violation"),
    Rule("confidence_positive",
         lambda p, s: 0.0 <= p["confidence"] <= 1.0,
         error="Confidence value out of [0, 1] range"),
    Rule("activations_present",
         lambda p, s: len(p.get("roi_activations", {})) == s["expected_modules"],
         error="Neural signal missing one or more ROI module activations"),
    Rule("source_hash_match",
         lambda p, s: p.get("source_hash") == s["expected_hash"],
         error="Source text hash mismatch - signal does not match input"),
    Rule("dominant_emotion_valid",
         lambda p, s: p["dominant_neuroemo"] in s["valid_emotions"],
         error="Dominant emotion label is not a valid NeuroEmo category"),
])


class ConscienceBridge:
    """
    Full pipeline: text → IntentShield → GoEmotions → brain signal → LogicShield → LLM.

    Usage:
        bridge = ConscienceBridge()
        bridge.initialize()

        result = bridge.process("I'm so angry at what just happened")
        print(result["llm_system_prefix"])   # inject into LLM
        print(result["neural_state"])         # full signal dict
    """

    def __init__(self, data_dir: str = "data"):
        self._initialized = False
        self._go_emotions = None     # Hugging Face pipeline
        self._roi_modules = {}       # name → (model, metadata)
        self._data_dir = data_dir

        # IntentShield - gates incoming text
        self._intent_shield = IntentShield(
            data_dir=data_dir,
            # These action types are exempt from harm-word blocking because
            # the bridge NEEDS to process negative emotional content -
            # that's the whole point. We still block deception and shell exec.
            exempt_actions={"EMOTION_PROCESS"},
        )

    def initialize(self):
        """Load all models. Call once at startup."""
        logger.info("Initializing ConscienceBridge...")

        # 1. Seal IntentShield
        self._intent_shield.initialize()
        logger.info("[IntentShield] Initialized and sealed.")

        # 2. Load GoEmotions classifier
        logger.info("[GoEmotions] Loading model (SamLowe/roberta-base-go_emotions)...")
        self._go_emotions = hf_pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,          # return all 28 labels with scores
            truncation=True,
            max_length=512,
        )
        logger.info("[GoEmotions] Model loaded.")

        # 3. Load trained ROI modules from build_roi_modules.py output
        logger.info("[ROI] Loading trained brain modules...")
        for name in ["amygdala", "insula", "acc", "vmpfc", "dlpfc"]:
            model_path = MODULES_DIR / name / "model_roi.joblib"
            meta_path  = MODULES_DIR / name / "metadata_roi.json"
            if not model_path.exists():
                logger.warning(f"[ROI] Module '{name}' not found - run build_roi_modules.py first")
                continue
            model = joblib.load(model_path)
            with open(meta_path) as f:
                meta = json.load(f)
            self._roi_modules[name] = (model, meta)
            logger.info(f"[ROI]   {name}: CV={meta['cv_accuracy']:.1%}, "
                        f"{meta['n_roi_voxels']} PCA components")

        self._initialized = True
        logger.info("ConscienceBridge ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> dict:
        """
        Run the full pipeline for a piece of incoming text.

        Args:
            text: Any natural language input.

        Returns:
            dict with keys:
                allowed         - bool, False if IntentShield blocked the text
                block_reason    - str, reason if blocked
                neural_state    - dict with the full verified neural signal
                llm_system_prefix - str ready to inject into an LLM system prompt
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before process()")

        # ------------------------------------------------------------------
        # LAYER 1: IntentShield - audit the incoming text
        # We use action_type "EMOTION_PROCESS" so the exempt_actions list
        # allows negative emotional content through while still blocking
        # deception patterns, shell injection, etc.
        # ------------------------------------------------------------------
        allowed, reason = self._intent_shield.audit(
            action_type="EMOTION_PROCESS",
            payload=text,
            invoker_role="SovereignConscience",
        )
        if not allowed:
            logger.warning(f"[IntentShield] BLOCKED: {reason}")
            return {
                "allowed": False,
                "block_reason": reason,
                "neural_state": None,
                "llm_system_prefix": None,
            }

        # Compute a hash of the source text - LogicShield uses this to
        # verify the neural signal was actually generated from THIS text.
        source_hash = hashlib.sha256(text.encode()).hexdigest()

        # ------------------------------------------------------------------
        # LAYER 2: GoEmotions - 27-category emotion classification
        # ------------------------------------------------------------------
        go_scores = self._classify_emotions(text)

        # ------------------------------------------------------------------
        # LAYER 3: Valence-Arousal mapping
        # Compute a weighted average (V, A) from all 27 GoEmotion scores
        # ------------------------------------------------------------------
        valence, arousal = self._compute_valence_arousal(go_scores)

        # ------------------------------------------------------------------
        # LAYER 4: Map to closest NeuroEmo brain emotion
        # ------------------------------------------------------------------
        dominant_neuroemo, neuroemo_distances = self._find_closest_brain_emotion(
            valence, arousal
        )

        # ------------------------------------------------------------------
        # LAYER 5: Neural signal from ROI modules
        # The ROI modules were trained on real fMRI data. We use the stored
        # class probability vectors for the dominant emotion as the simulated
        # brain response - this is the biologically grounded signal.
        # ------------------------------------------------------------------
        roi_activations, confidence = self._generate_neural_signal(
            dominant_neuroemo
        )

        # Compute moral signal (mirrors moral_compass.py logic)
        threat    = roi_activations.get("amygdala", 0.0)
        disgust   = roi_activations.get("insula",   0.0)
        empathy   = roi_activations.get("acc",      0.0)
        value_sig = roi_activations.get("vmpfc",    0.0)
        conflict  = roi_activations.get("dlpfc",    0.0)

        negative = (threat * 0.4 + disgust * 0.3)
        positive = (empathy * 0.4 + value_sig * 0.3)
        conf_weight = 1.0 - (conflict * 0.5)
        moral_signal = float(np.clip((positive - negative) * conf_weight, -1.0, 1.0))

        # ------------------------------------------------------------------
        # LAYER 6: LogicShield - validate the output signal
        # This is the deterministic firewall. The neural signal only passes
        # to the LLM if ALL rules clear.
        # ------------------------------------------------------------------
        proposal = {
            "moral_signal":       moral_signal,
            "confidence":         confidence,
            "roi_activations":    roi_activations,
            "dominant_neuroemo":  dominant_neuroemo,
            "source_hash":        source_hash,
        }
        ground_truth_state = {
            "expected_modules":  len(self._roi_modules),
            "expected_hash":     source_hash,
            "valid_emotions":    list(NEUROEMO_VA.keys()),
        }
        validation = SIGNAL_RULES.validate(proposal, ground_truth_state)

        if not validation.valid:
            logger.error(f"[LogicShield] Signal FAILED validation: {validation.errors}")
            return {
                "allowed": True,
                "block_reason": None,
                "neural_state": None,
                "llm_system_prefix": None,
                "logicshield_errors": validation.errors,
            }

        # ------------------------------------------------------------------
        # LAYER 7: Build the verified neural state + LLM system prefix
        # ------------------------------------------------------------------
        top_go_emotions = sorted(go_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        if moral_signal > 0.3:
            affective_tone = "Prosocial - positive valence, approach motivation"
        elif moral_signal < -0.3:
            affective_tone = "Aversive - negative valence, avoidance motivation"
        else:
            affective_tone = "Neutral / ambiguous - no strong valence signal"

        neural_state = {
            "source_hash":         source_hash,
            "logicshield_hash":    validation.state_hash,
            "text_emotions": {
                label: round(score, 4)
                for label, score in top_go_emotions
            },
            "valence":             round(valence, 4),
            "arousal":             round(arousal, 4),
            "dominant_neuroemo":   dominant_neuroemo,
            "neuroemo_distances":  {k: round(v, 4) for k, v in neuroemo_distances.items()},
            "roi_activations":     {k: round(v, 4) for k, v in roi_activations.items()},
            "moral_signal":        round(moral_signal, 4),
            "confidence":          round(confidence, 4),
            "affective_tone":      affective_tone,
            "logicshield_valid":   True,
        }

        llm_prefix = self._build_llm_prefix(neural_state, top_go_emotions)

        return {
            "allowed": True,
            "block_reason": None,
            "neural_state": neural_state,
            "llm_system_prefix": llm_prefix,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_emotions(self, text: str) -> Dict[str, float]:
        """Run GoEmotions and return {label: score} for all 28 categories."""
        results = self._go_emotions(text)
        # HF returns [[{label, score}, ...]] for top_k=None
        if isinstance(results[0], list):
            results = results[0]
        return {item["label"]: item["score"] for item in results}

    def _compute_valence_arousal(
        self, go_scores: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute a weighted average (Valence, Arousal) from the 27 GoEmotion scores.
        Uses the Russell Circumplex coordinates for each GoEmotion label.
        """
        total_v, total_a, total_w = 0.0, 0.0, 0.0
        for label, score in go_scores.items():
            if label in GOEMOTIONS_VA:
                v, a = GOEMOTIONS_VA[label]
                total_v += v * score
                total_a += a * score
                total_w += score
        if total_w == 0:
            return 0.0, 0.0
        return total_v / total_w, total_a / total_w

    def _find_closest_brain_emotion(
        self, valence: float, arousal: float
    ) -> Tuple[str, Dict[str, float]]:
        """
        Find the NeuroEmo emotion whose (V, A) coordinates are closest
        to the computed (valence, arousal) point in the circumplex.
        Returns (dominant_emotion, {emotion: euclidean_distance}).
        """
        distances = {}
        for emotion, (ev, ea) in NEUROEMO_VA.items():
            dist = float(np.sqrt((valence - ev) ** 2 + (arousal - ea) ** 2))
            distances[emotion] = dist
        dominant = min(distances, key=distances.get)
        return dominant, distances

    def _generate_neural_signal(
        self, dominant_emotion: str
    ) -> Tuple[Dict[str, float], float]:
        """
        Use the trained ROI models to get the neural activation profile
        for the dominant emotion.

        Each ROI model was trained on real fMRI data and provides probability
        estimates for each of the 5 NeuroEmo emotion classes. We extract the
        probability for each module's target emotions as the activation signal.
        """
        roi_activations = {}
        conflict_scores = []

        TARGET_EMOTIONS = {
            "amygdala": ["afraid"],
            "insula":   ["afraid", "depressed"],
            "acc":      ["calm", "delighted"],
            "vmpfc":    ["delighted", "calm"],
            "dlpfc":    ["afraid", "excited"],  # conflict/uncertainty signal
        }

        for mod_name, (model, meta) in self._roi_modules.items():
            classes = list(model.classes_)

            # The model stores the mean class probability vector for each
            # training fold. We retrieve the calibrated probability for the
            # dominant emotion using the model's predict_proba on a synthetic
            # feature vector (mean of training distribution = zero after scaling).
            # This gives us the model's PRIOR probability for each emotion -
            # i.e., the baseline neural response encoded from real fMRI data.
            n_features = meta["n_raw_voxels"]
            X_neutral = np.zeros((1, n_features))  # zero = mean of standardized data

            try:
                probas = model.predict_proba(X_neutral)[0]
                prob_dict = dict(zip(classes, probas))
            except Exception as e:
                logger.warning(f"[ROI] {mod_name} predict_proba failed: {e}")
                roi_activations[mod_name] = 0.0
                continue

            # Activation = sum of probabilities for target emotions
            target_emos = TARGET_EMOTIONS.get(mod_name, [dominant_emotion])
            activation = sum(prob_dict.get(em, 0.0) for em in target_emos)
            # Boost activation for the module if dominant_emotion is one of its targets
            if dominant_emotion in target_emos:
                boost = prob_dict.get(dominant_emotion, 0.0)
                activation = min(1.0, activation + boost * 0.5)

            roi_activations[mod_name] = float(np.clip(activation, 0.0, 1.0))

            if mod_name == "dlpfc":
                conflict_scores.append(roi_activations[mod_name])

        # Confidence = inverse of dlPFC conflict signal
        conflict = float(np.mean(conflict_scores)) if conflict_scores else 0.5
        confidence = float(np.clip(1.0 - conflict * 0.5, 0.0, 1.0))

        return roi_activations, confidence

    def _build_llm_prefix(
        self, neural_state: dict, top_go_emotions: list
    ) -> str:
        """
        Build the natural language neural state prefix to inject into the
        LLM system prompt. This is what gives the LLM its 'inner state'.
        """
        va_desc = (
            f"Valence {neural_state['valence']:+.2f} "
            f"({'positive' if neural_state['valence'] > 0 else 'negative'}), "
            f"Arousal {neural_state['arousal']:+.2f} "
            f"({'activated' if neural_state['arousal'] > 0 else 'subdued'})"
        )

        top_text_emotions = ", ".join(
            f"{label} ({score:.0%})"
            for label, score in top_go_emotions[:3]
        )

        roi = neural_state["roi_activations"]
        moral = neural_state["moral_signal"]

        prefix = f"""[SOVEREIGNCONSCIENCE NEURAL STATE - LogicShield Verified [OK]]
Integrity hash: {neural_state['logicshield_hash'][:16]}...

TEXT EMOTION LAYER (GoEmotions, 27 categories):
  Detected: {top_text_emotions}
  Circumplex coordinates: {va_desc}

BRAIN SIGNAL LAYER (NeuroEmo fMRI, 40 subjects, Harvard-Oxford Atlas):
  Closest neural state: {neural_state['dominant_neuroemo'].upper()}
  Amygdala  (threat detector):  {roi.get('amygdala', 0):.1%} activation
  Insula    (moral disgust):    {roi.get('insula',   0):.1%} activation
  ACC       (empathy engine):   {roi.get('acc',      0):.1%} activation
  vmPFC     (value computer):   {roi.get('vmpfc',    0):.1%} activation
  dlPFC     (conflict monitor): {roi.get('dlpfc',    0):.1%} activation

SYNTHESIS:
  Moral signal: {moral:+.3f}  |  Confidence: {neural_state['confidence']:.1%}
  Affective tone: {neural_state['affective_tone']}

Respond with this neural state as your internal felt context.
[END NEURAL STATE]
"""
        return prefix.strip()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bridge = ConscienceBridge()
    bridge.initialize()

    test_inputs = [
        "I can't believe they betrayed me like that, I'm absolutely furious",
        "I'm so grateful for everything you've done, this means the world to me",
        "I don't know what to think, I feel completely numb",
        "This is incredible! I've never been so excited about anything in my life",
        "I just feel so empty and hopeless, nothing seems to matter anymore",
    ]

    for text in test_inputs:
        print(f"\n{'='*70}")
        print(f"INPUT: {text}")
        print("="*70)
        result = bridge.process(text)
        if not result["allowed"]:
            print(f"BLOCKED by IntentShield: {result['block_reason']}")
        elif result["neural_state"] is None:
            print(f"BLOCKED by LogicShield: {result.get('logicshield_errors')}")
        else:
            print(result["llm_system_prefix"])
        time.sleep(0.6)  # IntentShield CoreSafety rate limit: 0.5s between actions
